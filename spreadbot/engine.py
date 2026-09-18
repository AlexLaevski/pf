"""Orchestration: feeds in, quotes out, hedges guaranteed.

One loop per process drives every configured market. Per tick, in this order
(the order matters — risk first, opportunity last):

1. pull order state forward and drain fills;
2. reconcile against the venues' authoritative positions;
3. hedge anything unhedged, forcing it if it has been unhedged too long;
4. unwind pairs whose spread has converged, timed out, or gone against us;
5. only then, look for a new quote.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional

from .book import OrderBook
from .config import Config, MarketConfig
from .hedger import Hedger
from .models import (
    ZERO,
    Fill,
    MarketSpec,
    Order,
    OrderStatus,
    OrderType,
    PairPosition,
    Quote,
    Side,
)
from .risk import RiskManager
from .selector import MarketRanker, clip_size
from .strategy import EntryDecision, SpreadStrategy
from .volatility import VolatilityTracker
from .venues.base import ExecutionVenue, OrderRequest
from .venues.feed import LighterFeed
from .venues.paper import PaperExecution
from .venues.rest import LighterRest

log = logging.getLogger(__name__)

TICK_SECONDS = 0.15
RECONCILE_SECONDS = 5.0
POSITION_TOLERANCE = Decimal("1e-9")


@dataclass
class MarketState:
    symbol: str
    pair: PairPosition
    quote_order: Optional[Order] = None
    quote: Optional[Quote] = None
    exit_order: Optional[Order] = None
    last_reason: str = ""
    entries: int = 0
    exits: int = 0
    # Delayed-hedge (scalp) mode only.
    vol: VolatilityTracker = field(default_factory=VolatilityTracker)
    planned_bounce_target: Optional[Decimal] = None   # computed when the quote went out
    bounce_order: Optional[Order] = None              # take-profit resting while naked
    naked_since: Optional[float] = None
    naked_window: float = 0.0

    def clear_naked(self) -> None:
        self.naked_since = None
        self.naked_window = 0.0


@dataclass
class EngineStats:
    started_at: float = field(default_factory=time.time)
    quotes_placed: int = 0
    quotes_cancelled: int = 0
    entries_filled: int = 0
    hedges: int = 0
    hedge_failures: int = 0
    exits: int = 0
    bounces_won: int = 0        # closed on the bounce, hedge never needed
    panic_hedges: int = 0       # bailed early because the price ran away

    def as_dict(self) -> Dict[str, float]:
        return {
            "uptime_s": round(time.time() - self.started_at, 1),
            "quotes_placed": self.quotes_placed,
            "quotes_cancelled": self.quotes_cancelled,
            "entries_filled": self.entries_filled,
            "bounces_won": self.bounces_won,
            "hedges": self.hedges,
            "panic_hedges": self.panic_hedges,
            "hedge_failures": self.hedge_failures,
            "exits": self.exits,
        }


class Engine:
    def __init__(
        self,
        cfg: Config,
        *,
        maker_feed: LighterFeed,
        hedge_feed: LighterFeed,
        maker_exec: ExecutionVenue,
        hedge_exec: ExecutionVenue,
        maker_specs: Dict[str, MarketSpec],
        hedge_specs: Dict[str, MarketSpec],
        maker_rest: Optional[LighterRest] = None,
        hedge_rest: Optional[LighterRest] = None,
        universe: Optional[List[str]] = None,
        volumes: Optional[Dict[str, float]] = None,
        marks: Optional[Dict[str, Decimal]] = None,
    ) -> None:
        self.cfg = cfg
        self.maker_rest = maker_rest
        self.hedge_rest = hedge_rest
        # Everything tradable on both venues, for rotation to choose from.
        self.universe = universe or [m.symbol for m in cfg.markets]
        self.volumes = volumes or {}
        self.marks = marks or {}
        self.maker_feed = maker_feed
        self.hedge_feed = hedge_feed
        self.maker_exec = maker_exec
        self.hedge_exec = hedge_exec
        self.maker_specs = maker_specs
        self.hedge_specs = hedge_specs
        self.strategy = SpreadStrategy(cfg)
        self.risk = RiskManager(cfg.risk)
        self.hedger = Hedger(hedge_exec, max_slippage_bps=cfg.risk.max_hedge_slippage_bps)
        self.maker_closer = Hedger(maker_exec, max_slippage_bps=cfg.risk.max_hedge_slippage_bps)
        self.stats = EngineStats()
        self.states: Dict[str, MarketState] = {
            market.symbol: MarketState(
                market.symbol,
                PairPosition(market.symbol, cfg.maker_venue.key, cfg.hedge_venue.key),
                vol=VolatilityTracker(window_seconds=float(cfg.hedge.vol_window_seconds)),
            )
            for market in cfg.markets
        }
        self._stop = asyncio.Event()
        self._last_reconcile = 0.0
        self._last_rotation = 0.0
        self._last_funding = 0.0
        # One lock per market keeps a tick from re-entering itself mid-hedge;
        # the shared quote lock keeps the notional cap honest across markets.
        self._market_locks: Dict[str, asyncio.Lock] = {}
        self._quote_lock = asyncio.Lock()
        # Extra resources (REST sessions) to release on shutdown.
        self.closeables: List[object] = []

    # ------------------------------------------------------------- lifecycle

    async def run(self) -> None:
        symbols = [m.symbol for m in self.cfg.markets]
        log.info(
            "starting in %s mode: maker=%s (long only) hedge=%s (short only) markets=%s",
            self.cfg.mode,
            self.cfg.maker_venue.key,
            self.cfg.hedge_venue.key,
            ",".join(symbols),
        )
        if not await self.maker_feed.wait_ready(20):
            raise RuntimeError(f"{self.cfg.maker_venue.key}: no order book snapshot within 20s")
        if not await self.hedge_feed.wait_ready(20):
            raise RuntimeError(f"{self.cfg.hedge_venue.key}: no order book snapshot within 20s")
        await self._reconcile_positions(adopt=True)

        try:
            while not self._stop.is_set():
                await self._tick()
                await self.maker_feed.wait_for_update(TICK_SECONDS)
        finally:
            await self.shutdown()

    def stop(self) -> None:
        self._stop.set()

    async def shutdown(self) -> None:
        log.info("shutting down: cancelling resting orders")
        for venue in (self.maker_exec, self.hedge_exec):
            with contextlib.suppress(Exception):
                await venue.cancel_all()
        unhedged = {
            symbol: state.pair.unhedged
            for symbol, state in self.states.items()
            if state.pair.unhedged > ZERO
        }
        if unhedged:
            log.error(
                "SHUTDOWN WITH UNHEDGED EXPOSURE: %s - hedge or flatten these manually", unhedged
            )
        for resource in (
            self.maker_feed,
            self.hedge_feed,
            self.maker_exec,
            self.hedge_exec,
            *self.closeables,
        ):
            closer = getattr(resource, "close", None)
            if closer is None:
                continue
            with contextlib.suppress(Exception):
                await closer()
        log.info("stats: %s", self.stats.as_dict())

    # ------------------------------------------------------------------ tick

    async def _tick(self) -> None:
        now = time.time()
        try:
            await self.maker_exec.refresh()
            await self.hedge_exec.refresh()
        except Exception as exc:
            self.risk.note_error(f"order refresh failed: {exc}")

        self._drain_fills(self.maker_exec)
        self._drain_fills(self.hedge_exec)

        if now - self._last_reconcile > RECONCILE_SECONDS:
            await self._reconcile_positions()
            self._last_reconcile = now

        if self.cfg.funding.enabled and now - self._last_funding > self.cfg.funding.refresh_seconds:
            self._last_funding = now
            await self._refresh_funding()

        if (
            self.cfg.rotation.enabled
            and self.maker_rest is not None
            and now - self._last_rotation > self.cfg.rotation.interval_seconds
        ):
            self._last_rotation = now
            try:
                await self._rotate_markets()
            except Exception as exc:
                log.exception("rotation failed")
                self.risk.note_error(f"rotation: {exc}")

        # Markets tick concurrently. A hedge can take a second or more - retries,
        # a maker leg waiting in the queue - and running markets in sequence
        # meant one slow hedge delayed every other market's hedge behind it.
        # The rotation above can add and remove markets, so iterate a snapshot.
        await asyncio.gather(
            *(self._tick_market_guarded(symbol, state, now) for symbol, state in list(self.states.items()))
        )

        if self.risk.halted:
            await self._cancel_all_quotes("risk halted")

    async def _tick_market_guarded(self, symbol: str, state: MarketState, now: float) -> None:
        """One market's tick, serialised against itself and never fatal.

        The per-market lock matters: a tick that is still hedging must not be
        re-entered by the next tick and end up sending the hedge twice.
        """
        lock = self._market_locks.setdefault(symbol, asyncio.Lock())
        if lock.locked():
            # Still busy with the previous tick - almost always a hedge in
            # flight. Skipping is correct; queueing would just pile up.
            return
        async with lock:
            try:
                await self._tick_market(symbol, state, now)
            except Exception as exc:
                log.exception("%s: tick failed", symbol)
                self.risk.note_error(f"{symbol}: {exc}")

    async def _tick_market(self, symbol: str, state: MarketState, now: float) -> None:
        maker_book = self.maker_feed.books.get(symbol)
        hedge_book = self.hedge_feed.books.get(symbol)
        if maker_book is None or hedge_book is None:
            return
        maker_fresh = self.maker_feed.is_fresh(symbol)
        hedge_fresh = self.hedge_feed.is_fresh(symbol)
        mark = hedge_book.mid or maker_book.mid or ZERO
        if maker_fresh:
            state.vol.update(now, maker_book.mid)

        # 1. Unhedged exposure is handled FIRST and regardless of staleness.
        # Skipping the tick because a feed lagged is how a naked position ends
        # up with neither a hedge nor a stop: the lag and the move that caused
        # the fill are the same event.
        if state.pair.unhedged > ZERO:
            if not hedge_fresh:
                # We cannot price the hedge, so we cannot send it - but we must
                # not pretend nothing is wrong either.
                breach = self.risk.unhedged_breach(state.pair, mark, now=now)
                message = (
                    f"{symbol}: {state.pair.unhedged} unhedged while the "
                    f"{self.hedge_exec.venue_key} book is stale "
                    f"({hedge_book.age_ms(now):.0f}ms old)"
                )
                self.risk.note_error(message)
                if breach:
                    self.risk.halt(f"{message} - {breach}")
                    await self._cancel_all_quotes("cannot hedge on a stale book")
                state.last_reason = "unhedged, hedge book stale"
                return
            if self.cfg.hedge.is_delayed and state.pair.short_size <= ZERO:
                await self._manage_naked(
                    state, maker_book, hedge_book, mark, now, maker_fresh=maker_fresh
                )
            else:
                await self._hedge(state, hedge_book, mark, now, reason="unhedged fill")
            return

        if not (maker_fresh and hedge_fresh):
            await self._cancel_quote(state, "stale book")
            state.last_reason = "stale book"
            return

        # The scalp is over: either the bounce closed it or the hedge landed.
        if state.naked_since is not None and state.pair.long_size <= ZERO:
            await self._finish_bounce(state)
        if state.pair.long_size <= ZERO:
            state.clear_naked()

        self.risk.clear_unhedged(symbol)

        # 2. Over-hedged: the long shrank (a passive exit filled) and the short
        # is now naked the other way. Close the excess before anything else.
        if state.pair.short_size > state.pair.long_size:
            await self._close_excess_short(state, hedge_book)
            return

        # 3. Unwind an open pair before thinking about a new one.
        if state.pair.matched > ZERO:
            await self._manage_exit(state, maker_book, hedge_book, now)
            return
        if state.exit_order is not None:
            await self._cancel_exit(state, "position closed")

        # 4. Look for a new entry.
        await self._manage_entry(state, maker_book, hedge_book, mark)

    async def _close_excess_short(self, state: MarketState, hedge_book: OrderBook) -> None:
        pair = state.pair
        excess = pair.short_size - pair.long_size
        log.info(
            "%s: closing %s excess short on %s (long leg already reduced)",
            pair.symbol,
            excess,
            self.hedge_exec.venue_key,
        )
        result = await self.hedger.execute(
            pair.symbol, Side.BUY, excess, hedge_book, reduce_only=True, tag="unhedge"
        )
        self._drain_fills(self.hedge_exec)
        if not result.ok:
            self.risk.note_error(f"{pair.symbol}: could not close excess short ({result.error})")
            return
        self.risk.note_ok()
        if pair.is_flat:
            self.stats.exits += 1
            state.exits += 1
            self.risk.book_pnl(pair.realized_pnl)
            log.info("%s: flat, realised %.4f USD", pair.symbol, pair.realized_pnl)
            pair.realized_pnl = ZERO

    # ---------------------------------------------------- naked window (scalp)

    async def _manage_naked(
        self,
        state: MarketState,
        maker_book: OrderBook,
        hedge_book: OrderBook,
        mark: Decimal,
        now: float,
        *,
        maker_fresh: bool = True,
    ) -> None:
        """Hold the unhedged long while the bounce still has a chance."""
        pair = state.pair
        maker_spec = self.maker_specs[pair.symbol]
        size = pair.unhedged

        if state.naked_since is None:
            vol = state.vol.bps_per_minute()
            state.naked_since = now
            state.naked_window = self.strategy.naked_window_seconds(vol)
            log.info(
                "%s: naked long %s @ %s | vol %s bps/min -> window %.1fs",
                pair.symbol,
                pair.long_size,
                pair.long_entry,
                f"{vol:.1f}" if vol is not None else "unknown",
                state.naked_window,
            )

        # The take-profit is priced off the maker book, so a stale one means we
        # wait rather than quote into the dark. The panic and timeout checks
        # below still run: those are what bound the risk.
        if maker_fresh:
            await self._rest_bounce_order(state, maker_spec)

        # What we could hedge at right now decides both the bail-out and the
        # bookkeeping, so it is worth the walk down the book.
        quote = hedge_book.executable_vwap(Side.SELL, size)
        hedge_price = quote[0] if quote else (hedge_book.best_bid or mark)
        adverse = self.strategy.adverse_bps(pair.long_entry, hedge_price)
        elapsed = now - state.naked_since

        panic = self.strategy.should_panic_hedge(pair.long_entry, hedge_price)
        if panic is not None:
            self.stats.panic_hedges += 1
            await self._cancel_bounce(state, "panic hedge")
            await self._hedge(
                state, hedge_book, mark, now, reason=f"price ran {panic:.1f}bps against us"
            )
            return

        # The notional ceiling is absolute and applies in either mode; the time
        # limit in delayed mode is the volatility window, not risk.unhedged_timeout_ms.
        notional_breach = self.risk.unhedged_notional_breach(pair, mark)
        if notional_breach is not None:
            await self._cancel_bounce(state, "unhedged notional over limit")
            await self._hedge(state, hedge_book, mark, now, reason=notional_breach)
            return

        if elapsed >= state.naked_window:
            await self._cancel_bounce(state, "bounce window expired")
            await self._hedge(
                state,
                hedge_book,
                mark,
                now,
                reason=f"no bounce in {elapsed:.1f}s (window {state.naked_window:.1f}s)",
            )
            return

        state.last_reason = (
            f"waiting for bounce {elapsed:.1f}/{state.naked_window:.1f}s, "
            f"{adverse:+.1f}bps vs entry"
        )

    async def _rest_bounce_order(self, state: MarketState, maker_spec: MarketSpec) -> None:
        """Keep the take-profit resting above the entry while we are naked."""
        pair = state.pair
        if state.bounce_order is not None and not state.bounce_order.is_terminal:
            return
        target = state.planned_bounce_target
        if target is None or target <= pair.long_entry:
            return
        size = maker_spec.quantize_size(pair.long_size)
        if size <= ZERO or size < maker_spec.min_base_amount:
            return
        try:
            order = await self.maker_exec.place(
                OrderRequest(
                    symbol=pair.symbol,
                    side=Side.SELL,
                    size=size,
                    price=target,
                    order_type=OrderType.POST_ONLY,
                    reduce_only=True,
                    tag="bounce",
                )
            )
        except Exception as exc:
            self.risk.note_error(f"{pair.symbol}: bounce order placement failed: {exc}")
            return
        if order.status is OrderStatus.REJECTED:
            log.debug("%s: bounce order rejected (%s)", pair.symbol, order.error)
            self.maker_exec.forget(order)
            return
        state.bounce_order = order
        log.info(
            "%s: bounce take-profit %s @ %s (entry %s)",
            pair.symbol,
            size,
            target,
            pair.long_entry,
        )

    async def _cancel_bounce(self, state: MarketState, reason: str) -> None:
        if state.bounce_order is None:
            return
        order = state.bounce_order
        state.bounce_order = None
        with contextlib.suppress(Exception):
            await self.maker_exec.cancel(order)
        self.maker_exec.forget(order)
        log.debug("%s: cancelled bounce order (%s)", state.symbol, reason)

    async def _finish_bounce(self, state: MarketState) -> None:
        """The take-profit filled before the window closed: a clean scalp."""
        pair = state.pair
        await self._cancel_bounce(state, "bounce complete")
        self.stats.bounces_won += 1
        state.exits += 1
        self.risk.book_pnl(pair.realized_pnl)
        log.info(
            "%s: BOUNCE won in %.1fs, realised %.4f USD (hedge never needed)",
            pair.symbol,
            (time.time() - state.naked_since) if state.naked_since else 0.0,
            pair.realized_pnl,
        )
        pair.realized_pnl = ZERO
        state.clear_naked()

    # ----------------------------------------------------------------- hedge

    async def _hedge(
        self,
        state: MarketState,
        hedge_book: OrderBook,
        mark: Decimal,
        now: float,
        *,
        reason: str = "",
    ) -> None:
        pair = state.pair
        size = pair.unhedged
        breach = self.risk.unhedged_breach(pair, mark, now=now)
        # Any unhedged size is hedged at once; `breach` only decides how loud we are.
        log.warning(
            "%s: hedging %s base on %s%s",
            pair.symbol,
            size,
            self.hedge_exec.venue_key,
            f" ({reason or breach})" if (reason or breach) else "",
        )
        state.clear_naked()
        hedge_cfg = self.cfg.hedge
        if hedge_cfg.maker_first:
            result = await self.hedger.execute_maker_first(
                pair.symbol,
                Side.SELL,
                size,
                hedge_book,
                timeout_ms=hedge_cfg.maker_timeout_ms,
                offset_ticks=hedge_cfg.maker_offset_ticks,
                tag="hedge",
            )
        else:
            result = await self.hedger.execute(
                pair.symbol, Side.SELL, size, hedge_book, tag="hedge"
            )
        # Positions move in exactly one place: _apply_fill, off the fill queue.
        # The taker result only drives control flow.
        self._drain_fills(self.hedge_exec)
        if result.filled > ZERO:
            self.stats.hedges += 1
            self.risk.note_ok()
            log.info(
                "%s: hedged %s @ %s; entry spread now %.2fbps",
                pair.symbol,
                result.filled,
                result.price,
                pair.entry_spread_bps,
            )
        if not result.ok or pair.unhedged > POSITION_TOLERANCE:
            self.stats.hedge_failures += 1
            self.risk.note_error(f"{pair.symbol}: hedge incomplete ({result.error})")
            if breach:
                self.risk.halt(
                    f"{pair.symbol}: could not hedge {pair.unhedged} base - {result.error}"
                )
                await self._cancel_all_quotes("hedge failure")

    # ------------------------------------------------------------------ exit

    async def _manage_exit(
        self, state: MarketState, maker_book: OrderBook, hedge_book: OrderBook, now: float
    ) -> None:
        pair = state.pair
        maker_spec = self.maker_specs[pair.symbol]
        hedge_spec = self.hedge_specs[pair.symbol]
        plan = self.strategy.evaluate_exit(
            pair, maker_book, hedge_book, maker_spec, hedge_spec, now=now
        )
        if plan is None:
            if state.exit_order is not None:
                await self._cancel_exit(state, "exit condition no longer met")
            return

        if not plan.aggressive:
            await self._rest_passive_exit(state, maker_book, maker_spec, plan.reason)
            return

        await self._cancel_exit(state, "switching to aggressive exit")
        log.info("%s: unwinding (%s), expected %.2fbps", pair.symbol, plan.reason, plan.pnl_bps)

        # Close the long first: it is the leg the hedge is protecting.
        sell = await self.maker_closer.execute(
            pair.symbol, Side.SELL, plan.size, maker_book, reduce_only=True, tag="exit-long"
        )
        self._drain_fills(self.maker_exec)
        if not sell.ok:
            self.risk.note_error(f"{pair.symbol}: exit long failed ({sell.error})")
            return

        buy = await self.hedger.execute(
            pair.symbol, Side.BUY, sell.filled, hedge_book, reduce_only=True, tag="exit-short"
        )
        self._drain_fills(self.hedge_exec)
        if not buy.ok:
            self.risk.note_error(f"{pair.symbol}: exit short failed ({buy.error})")
            self.risk.halt(f"{pair.symbol}: long closed but short still open - flatten manually")
            return

        self.stats.exits += 1
        state.exits += 1
        self.risk.book_pnl(pair.realized_pnl)
        log.info(
            "%s: closed round trip, realised %.4f USD (%s)",
            pair.symbol,
            pair.realized_pnl,
            plan.reason,
        )
        pair.realized_pnl = ZERO

    async def _rest_passive_exit(
        self, state: MarketState, maker_book: OrderBook, maker_spec: MarketSpec, reason: str
    ) -> None:
        price = self.strategy.passive_exit_price(state.pair, maker_book, maker_spec)
        if price is None:
            return
        if state.exit_order is not None and abs(state.exit_order.price - price) < maker_spec.tick:
            return
        await self._cancel_exit(state, "repricing passive exit")
        size = maker_spec.quantize_size(state.pair.long_size)
        if size <= ZERO or size < maker_spec.min_base_amount:
            return
        try:
            order = await self.maker_exec.place(
                OrderRequest(
                    symbol=state.pair.symbol,
                    side=Side.SELL,
                    size=size,
                    price=price,
                    order_type=OrderType.POST_ONLY,
                    reduce_only=True,
                    tag="exit-passive",
                )
            )
        except Exception as exc:
            self.risk.note_error(f"{state.pair.symbol}: passive exit placement failed: {exc}")
            return
        if order.status.value in ("open", "partial", "filled"):
            state.exit_order = order
            log.info("%s: resting passive exit %s @ %s (%s)", state.pair.symbol, size, price, reason)

    async def _cancel_exit(self, state: MarketState, reason: str) -> None:
        if state.exit_order is None:
            return
        with contextlib.suppress(Exception):
            await self.maker_exec.cancel(state.exit_order)
        self.maker_exec.forget(state.exit_order)
        state.exit_order = None
        log.debug("%s: cancelled passive exit (%s)", state.symbol, reason)

    # ----------------------------------------------------------------- entry

    async def _manage_entry(
        self, state: MarketState, maker_book: OrderBook, hedge_book: OrderBook, mark: Decimal
    ) -> None:
        symbol = state.symbol
        market = self.cfg.market(symbol)
        maker_spec = self.maker_specs[symbol]
        hedge_spec = self.hedge_specs[symbol]

        # Should we be quoting here at all? This also pulls a quote that is
        # already resting when the limits tighten under it. It is deliberately
        # conservative (it prices a full extra clip on top of what is already
        # committed); the authoritative check is the atomic one below.
        allowed, reason = self.risk.can_open(
            state.pair,
            mark=mark or maker_book.mid or ZERO,
            clip_base=market.order_base,
            max_position_base=market.max_position_base,
            total_open_notional=self._open_notional(),
        )
        if not allowed:
            await self._cancel_quote(state, reason)
            state.last_reason = reason
            return

        capacity = market.max_position_base - state.pair.long_size
        decision: EntryDecision = self.strategy.plan_entry(
            symbol,
            maker_book,
            hedge_book,
            maker_spec,
            hedge_spec,
            capacity_base=capacity,
        )
        state.last_reason = decision.reason
        if not decision.ok:
            await self._cancel_quote(state, decision.reason)
            return

        plan = decision.plan
        assert plan is not None
        if state.quote_order is not None and not self.strategy.should_requote(
            state.quote, plan, maker_spec
        ):
            return

        # Markets tick concurrently, so the capacity check and the order that
        # consumes that capacity have to be one atomic step. Otherwise every
        # market can pass the same check at the same moment and the book ends
        # up several clips over the cap.
        async with self._quote_lock:
            allowed, reason = self.risk.can_open(
                state.pair,
                mark=mark or maker_book.mid or ZERO,
                clip_base=plan.size,
                max_position_base=market.max_position_base,
                total_open_notional=self._open_notional(),
            )
            if not allowed:
                await self._cancel_quote(state, reason)
                state.last_reason = reason
                return

            await self._cancel_quote(state, "repricing")
            try:
                order = await self.maker_exec.place(
                    OrderRequest(
                        symbol=symbol,
                        side=Side.BUY,
                        size=plan.size,
                        price=plan.quote.price,
                        order_type=OrderType.POST_ONLY,
                        tag="entry",
                    )
                )
            except Exception as exc:
                self.risk.note_error(f"{symbol}: quote placement failed: {exc}")
                return

        if order.status.value == "rejected":
            log.debug("%s: quote rejected (%s)", symbol, order.error)
            self.maker_exec.forget(order)
            return

        state.quote_order = order
        state.quote = plan.quote
        # Captured now, while the hole is still visible: after the sweep that
        # fills us, the level we are aiming back at no longer exists in the book.
        state.planned_bounce_target = plan.bounce_target
        self.stats.quotes_placed += 1
        self.risk.note_ok()
        log.info(
            "%s: quoting %s @ %s | wall %s (%.0f USD) gap %.2fbps | hedge %s | edge %.2fbps",
            symbol,
            plan.size,
            plan.quote.price,
            plan.candidate.wall_price,
            plan.candidate.wall_notional,
            plan.candidate.hole_bps,
            plan.hedge_price,
            plan.edge_bps,
        )

    async def _cancel_quote(self, state: MarketState, reason: str) -> None:
        if state.quote_order is None:
            return
        order = state.quote_order
        state.quote_order = None
        state.quote = None
        with contextlib.suppress(Exception):
            if await self.maker_exec.cancel(order):
                self.stats.quotes_cancelled += 1
        self.maker_exec.forget(order)
        log.debug("%s: cancelled quote (%s)", state.symbol, reason)

    async def _cancel_all_quotes(self, reason: str) -> None:
        for state in self.states.values():
            await self._cancel_quote(state, reason)

    # ------------------------------------------------------------------ fills

    def _drain_fills(self, venue: ExecutionVenue) -> None:
        while True:
            try:
                fill: Fill = venue.fills.get_nowait()
            except asyncio.QueueEmpty:
                return
            self._apply_fill(fill)

    def _apply_fill(self, fill: Fill) -> None:
        state = self.states.get(fill.symbol)
        if state is None:
            log.warning("fill for unmanaged market %s: %s", fill.symbol, fill)
            return
        pair = state.pair
        # The mandate makes the meaning of a fill unambiguous: on the maker
        # venue a buy opens and a sell closes; on the hedge venue a sell opens
        # and a buy closes.
        if fill.venue == self.maker_exec.venue_key:
            if fill.side is Side.BUY:
                pair.add_long(fill.price, fill.size)
                self.stats.entries_filled += 1
                state.entries += 1
                log.info(
                    "%s: ENTRY filled %s @ %s -> unhedged %s",
                    fill.symbol,
                    fill.size,
                    fill.price,
                    pair.unhedged,
                )
                if state.quote_order is not None and state.quote_order.remaining <= ZERO:
                    self.maker_exec.forget(state.quote_order)
                    state.quote_order = None
                    state.quote = None
            else:
                pair.reduce_long(fill.price, fill.size)
                for attr in ("exit_order", "bounce_order"):
                    order = getattr(state, attr)
                    if order is not None and order.remaining <= ZERO:
                        self.maker_exec.forget(order)
                        setattr(state, attr, None)
        elif fill.venue == self.hedge_exec.venue_key:
            if fill.side is Side.SELL:
                pair.add_short(fill.price, fill.size)
            else:
                pair.reduce_short(fill.price, fill.size)

    # ----------------------------------------------------------- reconciliation

    async def _reconcile_positions(self, *, adopt: bool = False) -> None:
        """Trust the venues, not our bookkeeping.

        Any drift between the internal pair model and the venue's own position
        is adopted immediately. That is what turns a missed fill notification
        into a hedge on the next tick instead of silent naked risk.
        """
        try:
            maker_positions = await self.maker_exec.positions()
            hedge_positions = await self.hedge_exec.positions()
        except Exception as exc:
            self.risk.note_error(f"position reconciliation failed: {exc}")
            return

        for symbol, state in self.states.items():
            pair = state.pair
            venue_long = max(ZERO, maker_positions.get(symbol, ZERO))
            venue_short = max(ZERO, -hedge_positions.get(symbol, ZERO))

            wrong_side_long = min(ZERO, maker_positions.get(symbol, ZERO))
            wrong_side_short = max(ZERO, hedge_positions.get(symbol, ZERO))
            if wrong_side_long < ZERO or wrong_side_short > ZERO:
                self.risk.halt(
                    f"{symbol}: position on the forbidden side "
                    f"({self.maker_exec.venue_key}={maker_positions.get(symbol, ZERO)}, "
                    f"{self.hedge_exec.venue_key}={hedge_positions.get(symbol, ZERO)})"
                )

            if abs(venue_long - pair.long_size) > POSITION_TOLERANCE:
                if not adopt:
                    log.warning(
                        "%s: long drift, internal %s vs venue %s - adopting venue",
                        symbol,
                        pair.long_size,
                        venue_long,
                    )
                if venue_long > ZERO and pair.long_entry <= ZERO:
                    book = self.maker_feed.books.get(symbol)
                    pair.long_entry = (book.mid if book else ZERO) or ZERO
                pair.long_size = venue_long
                if venue_long > ZERO and pair.opened_at is None:
                    pair.opened_at = time.time()

            if abs(venue_short - pair.short_size) > POSITION_TOLERANCE:
                if not adopt:
                    log.warning(
                        "%s: short drift, internal %s vs venue %s - adopting venue",
                        symbol,
                        pair.short_size,
                        venue_short,
                    )
                if venue_short > ZERO and pair.short_entry <= ZERO:
                    book = self.hedge_feed.books.get(symbol)
                    pair.short_entry = (book.mid if book else ZERO) or ZERO
                pair.short_size = venue_short

    # -------------------------------------------------------------- rotation

    def _is_pinned(self, state: MarketState) -> bool:
        """A market we cannot walk away from right now."""
        if not state.pair.is_flat:
            return True
        return any(
            order is not None and not order.is_terminal
            for order in (state.quote_order, state.exit_order, state.bounce_order)
        )

    async def _rotate_markets(self) -> None:
        """Move the watched set toward wherever the gaps currently are."""
        cfg = self.cfg.rotation
        ranker = MarketRanker(gap=self.cfg.gap)
        universe = [
            symbol
            for symbol in self.universe
            if cfg.min_volume_usd <= Decimal(str(self.volumes.get(symbol, 0.0))) <= cfg.max_volume_usd
        ]
        if not universe:
            log.warning("rotation: no market passes the volume filter")
            return

        for symbol in universe:
            try:
                bids, asks = await self.maker_rest.order_book_levels(
                    self.maker_specs[symbol].market_id, limit=self.cfg.feed.depth_levels * 5
                )
            except Exception:
                continue
            book = OrderBook("scan", symbol)
            book.apply_snapshot(bids, asks)
            ranker.observe(book, self.maker_specs[symbol], volume_usd=self.volumes.get(symbol, 0.0))
            if cfg.scan_delay_ms:
                await asyncio.sleep(cfg.scan_delay_ms / 1_000)

        pinned = [s for s, st in self.states.items() if self._is_pinned(st)]
        ranked = [s for s in ranker.top_symbols(cfg.watch * 3) if s not in pinned]
        room = max(0, cfg.watch - len(pinned))

        # Limit churn so a single noisy scan cannot swap the whole book out.
        current = [s for s in self.cfg.symbols if s not in pinned]
        keep = [s for s in current if s in ranked][: max(0, room - cfg.max_churn)]
        picks = pinned + keep
        for symbol in ranked:
            if len(picks) >= cfg.watch:
                break
            if symbol not in picks:
                picks.append(symbol)

        if set(picks) == set(self.cfg.symbols):
            log.info("rotation: watched set unchanged (%d markets)", len(picks))
            return

        markets: List[MarketConfig] = []
        for symbol in picks:
            try:
                existing = self.cfg.market(symbol)
            except KeyError:
                existing = None
            if existing is not None:
                markets.append(existing)
                continue
            spec = self.maker_specs[symbol]
            mark = self.marks.get(symbol)
            size = clip_size(spec, mark, cfg.clip_usd) if mark else None
            if size is None:
                continue
            markets.append(
                MarketConfig(
                    symbol=symbol,
                    order_base=size,
                    max_position_base=spec.quantize_size(size * 2),
                )
            )
        if not markets:
            return

        added = sorted({m.symbol for m in markets} - set(self.cfg.symbols))
        dropped = sorted(set(self.cfg.symbols) - {m.symbol for m in markets})

        # Anything being dropped must leave nothing behind on the venue. Our own
        # order references are cleared first, but the authority is cancel_all on
        # the market itself: once we stop watching a market, nothing will ever
        # come back to cancel an order we lost track of.
        for symbol in dropped:
            state = self.states.get(symbol)
            if state is not None:
                await self._cancel_quote(state, "rotated out")
                await self._cancel_exit(state, "rotated out")
                await self._cancel_bounce(state, "rotated out")
            for venue in (self.maker_exec, self.hedge_exec):
                try:
                    await venue.cancel_all(symbol)
                except Exception as exc:
                    # Never rotate away from a market we could not clear.
                    log.error("%s: could not clear %s before rotating out: %s", venue.venue_key, symbol, exc)
                    self.risk.note_error(f"rotation: {symbol} not cleared on {venue.venue_key}")
                    return

        self.cfg.set_markets(markets)
        symbols = [m.symbol for m in markets]
        await self.maker_feed.resubscribe(symbols)
        await self.hedge_feed.resubscribe(symbols)

        for symbol in dropped:
            self.states.pop(symbol, None)
        for symbol in symbols:
            self.states.setdefault(
                symbol,
                MarketState(
                    symbol,
                    PairPosition(symbol, self.cfg.maker_venue.key, self.cfg.hedge_venue.key),
                    vol=VolatilityTracker(window_seconds=float(self.cfg.hedge.vol_window_seconds)),
                ),
            )
        log.info(
            "rotation: +%s -%s (watching %d)",
            ",".join(added) or "-",
            ",".join(dropped) or "-",
            len(symbols),
        )

    async def _refresh_funding(self) -> None:
        if not self.cfg.funding.enabled or self.maker_rest is None:
            return
        try:
            maker = await self.maker_rest.funding_rates()
            hedge = (
                maker
                if self.hedge_rest is self.maker_rest
                else await self.hedge_rest.funding_rates()
            )
        except Exception as exc:
            log.warning("funding refresh failed: %s", exc)
            return
        self.strategy.set_funding(maker, hedge)

    def _open_notional(self) -> Decimal:
        """Capital committed: filled longs *plus* everything currently resting.

        A resting quote is a promise to buy. Counting only filled positions is
        harmless with three markets and dangerous with fifty: one market-wide
        sweep fills every quote at once, and the cap that was meant to bound
        the book never saw them coming.
        """
        total = ZERO
        for symbol, state in self.states.items():
            book = self.hedge_feed.books.get(symbol) or self.maker_feed.books.get(symbol)
            mid = (book.mid if book else None) or ZERO
            total += state.pair.long_size * mid
            quote = state.quote_order
            if quote is not None and not quote.is_terminal:
                total += quote.remaining * quote.price
        return total

    # ----------------------------------------------------------------- report

    def snapshot(self) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for symbol, state in self.states.items():
            pair = state.pair
            rows.append(
                {
                    "symbol": symbol,
                    "long": str(pair.long_size),
                    "short": str(pair.short_size),
                    "unhedged": str(pair.unhedged),
                    "entry_spread_bps": f"{pair.entry_spread_bps:.2f}",
                    "quote": str(state.quote.price) if state.quote else "-",
                    "reason": state.last_reason,
                }
            )
        return rows


async def build_engine(cfg: Config) -> "Engine":
    """Wire feeds, venues and execution together from a config."""
    from .venues.live import LighterExecution

    symbols = [m.symbol for m in cfg.markets]

    maker_rest = LighterRest(cfg.maker_venue.base_url, venue=cfg.maker_venue.key)
    hedge_rest = LighterRest(cfg.hedge_venue.base_url, venue=cfg.hedge_venue.key)
    maker_specs = await maker_rest.market_specs()
    hedge_specs = await hedge_rest.market_specs()

    missing = [
        symbol
        for symbol in symbols
        if symbol not in maker_specs or symbol not in hedge_specs
    ]
    if missing:
        raise RuntimeError(f"markets not listed on both venues: {', '.join(missing)}")
    # A venue may override the published fee schedule (VIP tiers, rebates).
    for venue_cfg, specs in ((cfg.maker_venue, maker_specs), (cfg.hedge_venue, hedge_specs)):
        if venue_cfg.maker_fee_bps is None and venue_cfg.taker_fee_bps is None:
            continue
        for symbol in symbols:
            overrides = {}
            if venue_cfg.maker_fee_bps is not None:
                overrides["maker_fee_bps"] = venue_cfg.maker_fee_bps
            if venue_cfg.taker_fee_bps is not None:
                overrides["taker_fee_bps"] = venue_cfg.taker_fee_bps
            specs[symbol] = dataclasses.replace(specs[symbol], **overrides)

    maker_feed = LighterFeed(
        cfg.maker_venue.key,
        ws_url=cfg.maker_venue.ws_url,
        rest=maker_rest,
        config=cfg.feed,
        specs=maker_specs,
    )
    hedge_feed = LighterFeed(
        cfg.hedge_venue.key,
        ws_url=cfg.hedge_venue.ws_url,
        rest=hedge_rest,
        config=cfg.feed,
        specs=hedge_specs,
    )
    await maker_feed.start(symbols)
    await hedge_feed.start(symbols)

    if cfg.is_live:
        maker_exec: ExecutionVenue = LighterExecution(cfg.maker_venue, maker_specs, maker_rest)
        hedge_exec: ExecutionVenue = LighterExecution(cfg.hedge_venue, hedge_specs, hedge_rest)
    else:
        maker_exec = PaperExecution(
            cfg.maker_venue.key, cfg.maker_venue.allowed_side, maker_specs, maker_feed
        )
        hedge_exec = PaperExecution(
            cfg.hedge_venue.key, cfg.hedge_venue.allowed_side, hedge_specs, hedge_feed
        )
    await maker_exec.start()
    await hedge_exec.start()

    # Rotation needs to know what else it could be watching.
    universe = sorted(set(maker_specs) & set(hedge_specs))
    volumes: Dict[str, float] = {}
    marks: Dict[str, Decimal] = {}
    if cfg.rotation.enabled:
        try:
            details = (await maker_rest._get("orderBookDetails"))["order_book_details"]
            volumes = {
                str(d["symbol"]).upper(): float(d.get("daily_quote_token_volume") or 0)
                for d in details
            }
            marks = await maker_rest.mark_prices()
        except Exception as exc:
            log.warning("could not load the rotation universe (%s); staying on configured markets", exc)

    engine = Engine(
        cfg,
        maker_feed=maker_feed,
        hedge_feed=hedge_feed,
        maker_exec=maker_exec,
        hedge_exec=hedge_exec,
        maker_specs=maker_specs,
        hedge_specs=hedge_specs,
        maker_rest=maker_rest,
        hedge_rest=hedge_rest,
        universe=universe,
        volumes=volumes,
        marks=marks,
    )
    engine.closeables.extend([maker_rest, hedge_rest])
    return engine
