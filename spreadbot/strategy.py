"""The trading logic: where to quote, whether it pays, and when to unwind.

Directional mandate, which the whole module assumes:

* maker venue (Lighter on Robinhood) -- **long only**. We rest a passive buy.
* hedge venue (Lighter Core) -- **short only**. We sell to hedge the fill.

So the bot is not a two-sided market maker. It has one trade: buy the maker
venue cheap into a thin book, sell the hedge venue immediately. It makes money
when the maker venue's book is gappy enough that a sweep fills us well below
where the hedge venue is bid.

Unwinding runs the same two legs in reverse, both ``reduce_only``: sell the
long on the maker venue, buy back the short on the hedge venue. Neither leg can
open exposure on the forbidden side.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional

from .book import OrderBook
from .config import Config
from .gaps import WallCandidate, find_wall_candidates
from .models import ZERO, MarketSpec, PairPosition, Quote, Side
from .volatility import scaled_window_seconds

BPS = Decimal(10_000)


@dataclass(frozen=True)
class EntryPlan:
    """A quote worth resting, together with the hedge it assumes."""

    symbol: str
    quote: Quote
    candidate: WallCandidate
    hedge_price: Decimal
    edge_bps: Decimal
    size: Decimal
    # Delayed mode only: where the take-profit rests while we hold the naked long.
    bounce_target: Optional[Decimal] = None


@dataclass(frozen=True)
class EntryDecision:
    plan: Optional[EntryPlan]
    reason: str
    best_edge_bps: Optional[Decimal] = None
    candidates: int = 0

    @property
    def ok(self) -> bool:
        return self.plan is not None


@dataclass(frozen=True)
class ExitPlan:
    symbol: str
    reason: str
    aggressive: bool          # True: take both legs now. False: rest a passive exit.
    size: Decimal
    maker_price: Optional[Decimal]     # sell price on the maker venue
    hedge_price: Optional[Decimal]     # buy-back price on the hedge venue
    pnl_bps: Decimal


class SpreadStrategy:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        # Hourly funding rates as fractions of notional, per venue, keyed by
        # symbol. Positive means longs pay. Refreshed by the engine.
        self.maker_funding: Dict[str, Decimal] = {}
        self.hedge_funding: Dict[str, Decimal] = {}

    # ---------------------------------------------------------------- funding

    def set_funding(self, maker: Dict[str, Decimal], hedge: Dict[str, Decimal]) -> None:
        self.maker_funding = maker
        self.hedge_funding = hedge

    def funding_cost_bps(self, symbol: str, *, hold_hours: Optional[Decimal] = None) -> Decimal:
        """Net funding we expect to pay over the hold, in bps of notional.

        Long on the maker venue, short on the hedge venue: we pay the maker
        venue's rate and receive the hedge venue's, so only the difference
        matters. Positive means it costs us.
        """
        cfg = self.cfg.funding
        if not cfg.enabled:
            return ZERO
        maker_rate = self.maker_funding.get(symbol)
        hedge_rate = self.hedge_funding.get(symbol)
        if maker_rate is None and hedge_rate is None:
            return ZERO
        net_hourly = (maker_rate or ZERO) - (hedge_rate or ZERO)
        hours = cfg.expected_hold_hours if hold_hours is None else hold_hours
        return net_hourly * hours * BPS

    # ------------------------------------------------------------------- edge

    def entry_edge_bps(
        self,
        entry_price: Decimal,
        hedge_price: Decimal,
        maker_spec: MarketSpec,
        hedge_spec: MarketSpec,
    ) -> Decimal:
        """Net bps captured by buying at ``entry_price`` and selling at ``hedge_price``.

        Maker fee on the entry leg, taker fee on the hedge leg (configurable),
        both charged on their own notional.
        """
        if entry_price <= ZERO:
            return ZERO
        maker_fee = entry_price * maker_spec.maker_fee_bps / BPS
        hedge_fee_bps = hedge_spec.taker_fee_bps if self.cfg.edge.hedge_is_taker else hedge_spec.maker_fee_bps
        hedge_fee = hedge_price * hedge_fee_bps / BPS
        net = hedge_price - entry_price - maker_fee - hedge_fee
        return net / entry_price * BPS

    def bounce_edge_bps(self, entry_price: Decimal, target: Decimal, maker_spec: MarketSpec) -> Decimal:
        """Net bps if the sweep mean-reverts and our take-profit fills.

        Both legs are maker orders on the same venue, so the fee is charged
        twice at the maker rate and no cross-venue price enters the sum.
        """
        if entry_price <= ZERO:
            return ZERO
        fees = (entry_price + target) * maker_spec.maker_fee_bps / BPS
        return (target - entry_price - fees) / entry_price * BPS

    def bounce_target_price(
        self, entry_price: Decimal, candidate: WallCandidate, maker_spec: MarketSpec
    ) -> Decimal:
        """Where to rest the take-profit for the bounce.

        ``hole_top`` aims at the near edge of the gap we were filled through:
        the sweep came from there, so that is where a reversion goes back to.
        A floor of ``min_target_bps`` keeps the target from collapsing onto the
        entry when the hole is narrow.
        """
        cfg = self.cfg.bounce
        floor = entry_price * (BPS + cfg.min_target_bps) / BPS
        if cfg.target == "hole_top":
            target = max(candidate.near_price, floor)
        else:
            target = max(entry_price * (BPS + cfg.target_bps) / BPS, floor)
        return maker_spec.quantize_price(target, side=Side.SELL)

    def min_entry_bps(self, symbol: str) -> Decimal:
        market = self.cfg.market(symbol)
        base = market.min_entry_bps if market.min_entry_bps is not None else self.cfg.edge.min_entry_bps
        return base + self.cfg.edge.extra_buffer_bps

    # ------------------------------------------------------------------ entry

    def plan_entry(
        self,
        symbol: str,
        maker_book: OrderBook,
        hedge_book: OrderBook,
        maker_spec: MarketSpec,
        hedge_spec: MarketSpec,
        *,
        capacity_base: Decimal,
    ) -> EntryDecision:
        """Pick the best place to rest a buy on the maker venue, if any."""
        market = self.cfg.market(symbol)
        size = maker_spec.quantize_size(min(market.order_base, capacity_base))
        if size <= ZERO or size < maker_spec.min_base_amount:
            return EntryDecision(None, "no capacity left for another clip")
        if not maker_book.ready or not hedge_book.ready:
            return EntryDecision(None, "book not ready")

        # What we could actually sell this clip for on the hedge venue right now.
        # In immediate mode this is the other half of the trade. In delayed mode
        # it is only the backstop, but it still has to be there: entering a scalp
        # we could not hedge if it goes wrong is how a small loss becomes a big
        # one, so the depth check applies in both modes.
        hedge_quote = hedge_book.executable_vwap(Side.SELL, size)
        if hedge_quote is None:
            return EntryDecision(None, "hedge book empty")
        hedge_price, fillable = hedge_quote
        if fillable < size:
            return EntryDecision(None, f"hedge book too thin ({fillable} of {size})")

        candidates: List[WallCandidate] = find_wall_candidates(
            maker_book,
            Side.BUY,
            maker_spec,
            self.cfg.gap,
            max_ahead_notional_usd=self.cfg.gap.max_ahead_notional_usd,
        )
        if not candidates:
            return EntryDecision(None, "no wall with a gap in front of it", candidates=0)

        threshold = self.min_entry_bps(symbol)
        delayed = self.cfg.hedge.is_delayed

        # Funding is charged on the position, not on the trade, so it comes off
        # every candidate equally - and on a hot token it can exceed the whole
        # edge on its own, in which case there is nothing to look for here.
        funding_bps = self.funding_cost_bps(symbol)
        if funding_bps > self.cfg.funding.max_cost_bps:
            return EntryDecision(
                None,
                f"funding costs {funding_bps:.2f}bps over the expected hold, "
                f"limit {self.cfg.funding.max_cost_bps}bps",
            )

        best_edge: Optional[Decimal] = None
        blocked: Optional[str] = None
        for candidate in candidates:
            # The two modes are paid by different things, so they are gated on
            # different numbers: the cross-venue spread for an immediate hedge,
            # the size of the bounce back up the hole for the scalp.
            target = (
                self.bounce_target_price(candidate.price, candidate, maker_spec)
                if delayed
                else None
            )
            edge = (
                self.bounce_edge_bps(candidate.price, target, maker_spec)
                if delayed
                else self.entry_edge_bps(candidate.price, hedge_price, maker_spec, hedge_spec)
            ) - funding_bps
            if best_edge is None or edge > best_edge:
                best_edge = edge
            if edge < threshold:
                # Candidates run from the touch outward, so a shallow wall that
                # misses the bar may still be beaten by a deeper one.
                continue
            if candidate.price * size < maker_spec.min_quote_amount:
                blocked = (
                    f"clip notional {candidate.price * size:.2f} below venue minimum "
                    f"{maker_spec.min_quote_amount}"
                )
                continue
            quote = Quote(
                venue=self.cfg.maker_venue.key,
                symbol=symbol,
                side=Side.BUY,
                price=candidate.price,
                size=size,
                wall_price=candidate.wall_price,
                gap_bps=candidate.hole_bps,
                edge_bps=edge,
                hedge_price=hedge_price,
            )
            return EntryDecision(
                EntryPlan(
                    symbol=symbol,
                    quote=quote,
                    bounce_target=target,
                    candidate=candidate,
                    hedge_price=hedge_price,
                    edge_bps=edge,
                    size=size,
                ),
                "ok",
                best_edge_bps=best_edge,
                candidates=len(candidates),
            )

        if blocked is not None:
            reason = blocked
        elif best_edge is not None:
            reason = f"best edge {best_edge:.2f}bps below threshold {threshold:.2f}bps"
        else:
            reason = "no priceable candidate"
        return EntryDecision(None, reason, best_edge_bps=best_edge, candidates=len(candidates))

    def should_requote(self, existing: Optional[Quote], plan: EntryPlan, maker_spec: MarketSpec) -> bool:
        tick = maker_spec.tick * Decimal(max(1, self.cfg.edge.requote_ticks))
        return plan.quote.differs_from(existing, tick=tick, size_tol=self.cfg.edge.requote_size_tol)

    # ---------------------------------------------------- naked window (scalp)

    def naked_window_seconds(self, vol_bps_per_minute: Optional[float]) -> float:
        """How long to hold the unhedged long, given current volatility."""
        cfg = self.cfg.hedge
        return scaled_window_seconds(
            vol_bps_per_minute,
            base_seconds=float(cfg.base_seconds),
            reference_vol_bps=float(cfg.reference_vol_bps_per_min),
            min_seconds=float(cfg.min_seconds),
            max_seconds=float(cfg.max_seconds),
        )

    def adverse_bps(self, entry_price: Decimal, hedge_price: Decimal) -> Decimal:
        """How far the hedge-able price has fallen below our entry, in bps.

        Negative means the trade is in our favour. This is measured against the
        price we could actually hedge at, not the mid, because that is the
        number the decision to bail is really about.
        """
        if entry_price <= ZERO:
            return ZERO
        return (entry_price - hedge_price) / entry_price * BPS

    def should_panic_hedge(self, entry_price: Decimal, hedge_price: Decimal) -> Optional[Decimal]:
        """Return the adverse move when it has blown through ``panic_bps``."""
        if not self.cfg.hedge.is_delayed:
            return None
        adverse = self.adverse_bps(entry_price, hedge_price)
        return adverse if adverse >= self.cfg.hedge.panic_bps else None

    # ------------------------------------------------------------------- exit

    def evaluate_exit(
        self,
        pair: PairPosition,
        maker_book: OrderBook,
        hedge_book: OrderBook,
        maker_spec: MarketSpec,
        hedge_spec: MarketSpec,
        *,
        now: float,
    ) -> Optional[ExitPlan]:
        """Decide whether to unwind the hedged pair, and how hard."""
        size = pair.matched
        if size <= ZERO:
            return None

        # Closing prices available right now: sell the long, buy back the short.
        maker_quote = maker_book.executable_vwap(Side.SELL, size)
        hedge_quote = hedge_book.executable_vwap(Side.BUY, size)
        if maker_quote is None or hedge_quote is None:
            return None
        maker_exit, maker_fillable = maker_quote
        hedge_exit, hedge_fillable = hedge_quote

        # Total P&L of the round trip, in bps of the long entry:
        #   entry captured (short_entry - long_entry) + exit (maker_exit - hedge_exit)
        if pair.long_entry <= ZERO:
            return None
        gross = (pair.short_entry - pair.long_entry) + (maker_exit - hedge_exit)
        fees = (
            pair.long_entry * maker_spec.maker_fee_bps
            + pair.short_entry * hedge_spec.taker_fee_bps
            + maker_exit * maker_spec.taker_fee_bps
            + hedge_exit * hedge_spec.taker_fee_bps
        ) / BPS
        pnl_bps = (gross - fees) / pair.long_entry * BPS

        # The part that is still open: how much the reverse spread has closed.
        exit_spread_bps = (maker_exit - hedge_exit) / pair.long_entry * BPS

        held_for = now - (pair.opened_at or now)
        unwind = self.cfg.unwind

        if pnl_bps <= -unwind.stop_loss_bps:
            return ExitPlan(
                pair.symbol, f"stop loss ({pnl_bps:.2f}bps)", True, size, maker_exit, hedge_exit, pnl_bps
            )
        if held_for >= unwind.max_hold_seconds:
            return ExitPlan(
                pair.symbol, f"max hold ({held_for:.0f}s)", True, size, maker_exit, hedge_exit, pnl_bps
            )
        if maker_fillable < size or hedge_fillable < size:
            # Not enough depth to exit aggressively; wait unless something else fires.
            return None
        if exit_spread_bps >= -self.cfg.edge.exit_bps:
            return ExitPlan(
                pair.symbol,
                f"spread converged ({exit_spread_bps:.2f}bps)",
                not unwind.passive_exit,
                size,
                maker_exit,
                hedge_exit,
                pnl_bps,
            )
        return None

    def passive_exit_price(
        self,
        pair: PairPosition,
        maker_book: OrderBook,
        maker_spec: MarketSpec,
    ) -> Optional[Decimal]:
        """Where to rest the reduce-only sell that closes the long.

        Mirror of the entry: sit one tick in front of the first ask-side wall,
        so the exit is itself a maker fill inside a gap.
        """
        candidates = find_wall_candidates(maker_book, Side.SELL, maker_spec, self.cfg.gap)
        for candidate in candidates:
            if candidate.price > pair.long_entry:
                return candidate.price
        best_ask = maker_book.best_ask
        if best_ask is None:
            return None
        price = maker_spec.quantize_price(best_ask - maker_spec.tick, side=Side.SELL)
        return price if price > pair.long_entry else None
