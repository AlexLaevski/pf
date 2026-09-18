"""Measurement mode: what *would* have happened, recorded without trading.

Paper mode answers "does the plumbing work". This answers the only question
that decides whether the strategy is worth running:

* how often is there anywhere to rest a quote (time in market),
* how often does a sweep actually reach a resting quote (fill rate),
* when it does, does the price come back (bounce rate) or keep going
  (adverse selection),
* and what is the realised edge once those are combined.

Nothing here places an order. Every quote is virtual, tracked against the
public book, and written to JSONL when its life ends. The numbers are still
optimistic in one specific way — a virtual quote has no queue ahead of it
beyond what the book shows — but unlike a P&L guess, each row is a claim you
can check against the tape.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, TextIO

from .book import OrderBook
from .config import Config
from .models import ZERO, MarketSpec
from .strategy import SpreadStrategy

log = logging.getLogger(__name__)

BPS = Decimal(10_000)


@dataclass
class VirtualQuote:
    """One would-be order, from the moment we would have placed it."""

    symbol: str
    placed_at: float
    price: float
    size: float
    target: float
    edge_bps: float
    gap_bps: float
    distance_bps: float
    wall_price: float
    wall_notional: float
    ahead_notional: float
    mid_at_quote: float
    vol_bps_per_min: Optional[float]

    # Life of the quote.
    outcome: str = "resting"          # resting | cancelled | filled_pending | filled | expired
    rested_seconds: float = 0.0
    filled_at: Optional[float] = None
    # A quote only becomes measurable once the book is observed strictly above
    # it: until then it is at or through the touch, and "the best bid is below
    # our price" says nothing about whether anyone traded with us.
    armed: bool = False
    armed_at: Optional[float] = None

    # After a virtual fill.
    bounce_hit: bool = False
    seconds_to_bounce: Optional[float] = None
    max_adverse_bps: float = 0.0
    hedge_bps_at_end: Optional[float] = None
    realised_bps: Optional[float] = None
    window_seconds: float = 0.0

    def as_row(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MeasureConfig:
    horizon_seconds: float = 60.0      # how long a filled quote is tracked
    max_rest_seconds: float = 600.0    # a quote nobody touches is eventually written off
    requote_bps: float = 2.0           # reprice the virtual quote when the target moves this far


@dataclass
class MeasureStats:
    quotes: int = 0
    cancelled: int = 0
    fills: int = 0
    bounces: int = 0
    expired_unfilled: int = 0
    resting_seconds: float = 0.0
    watched_seconds: float = 0.0

    @property
    def fill_rate(self) -> float:
        done = self.fills + self.expired_unfilled + self.cancelled
        return self.fills / done if done else 0.0

    @property
    def bounce_rate(self) -> float:
        return self.bounces / self.fills if self.fills else 0.0


class Recorder:
    """Drives virtual quotes for every watched market."""

    def __init__(
        self,
        cfg: Config,
        strategy: SpreadStrategy,
        maker_specs: Dict[str, MarketSpec],
        hedge_specs: Dict[str, MarketSpec],
        *,
        measure: Optional[MeasureConfig] = None,
        sink: Optional[TextIO] = None,
    ) -> None:
        self.cfg = cfg
        self.strategy = strategy
        self.maker_specs = maker_specs
        self.hedge_specs = hedge_specs
        self.measure = measure or MeasureConfig()
        self.sink = sink
        self.stats = MeasureStats()
        self.open: Dict[str, VirtualQuote] = {}       # symbol -> resting virtual quote
        self.tracking: List[VirtualQuote] = []        # filled, inside the horizon
        self.rows: List[Dict[str, Any]] = []
        self._started = time.time()

    # ------------------------------------------------------------------ tick

    def tick(
        self,
        symbol: str,
        maker_book: OrderBook,
        hedge_book: OrderBook,
        *,
        now: float,
        vol_bps_per_min: Optional[float] = None,
    ) -> None:
        if not maker_book.ready or not hedge_book.ready:
            return
        self._advance_tracking(symbol, maker_book, hedge_book, now)
        self._check_fill(symbol, maker_book, now)
        self._refresh_quote(symbol, maker_book, hedge_book, now, vol_bps_per_min)

    # ------------------------------------------------------------- internals

    def _refresh_quote(
        self,
        symbol: str,
        maker_book: OrderBook,
        hedge_book: OrderBook,
        now: float,
        vol: Optional[float],
    ) -> None:
        market = self.cfg.market(symbol)
        maker_spec = self.maker_specs[symbol]
        hedge_spec = self.hedge_specs[symbol]

        decision = self.strategy.plan_entry(
            symbol,
            maker_book,
            hedge_book,
            maker_spec,
            hedge_spec,
            capacity_base=market.max_position_base,
        )
        existing = self.open.get(symbol)

        if not decision.ok or decision.plan is None:
            if existing is not None:
                self._close(existing, "cancelled" if existing.armed else "unmeasurable", now)
                self.stats.cancelled += 1
                del self.open[symbol]
            return

        plan = decision.plan
        mid = maker_book.mid or ZERO
        if existing is not None:
            moved = abs(Decimal(str(existing.price)) - plan.quote.price) / plan.quote.price * BPS
            if moved < Decimal(str(self.measure.requote_bps)):
                return
            self._close(existing, "cancelled" if existing.armed else "unmeasurable", now)
            self.stats.cancelled += 1

        target = plan.bounce_target or self.strategy.bounce_target_price(
            plan.quote.price, plan.candidate, maker_spec
        )
        quote = VirtualQuote(
            symbol=symbol,
            placed_at=now,
            price=float(plan.quote.price),
            size=float(plan.size),
            target=float(target),
            edge_bps=float(plan.edge_bps),
            gap_bps=float(plan.candidate.hole_bps),
            distance_bps=float(plan.candidate.distance_bps),
            wall_price=float(plan.candidate.wall_price),
            wall_notional=float(plan.candidate.wall_notional),
            ahead_notional=float(plan.candidate.ahead_notional),
            mid_at_quote=float(mid),
            vol_bps_per_min=vol,
            window_seconds=self.strategy.naked_window_seconds(vol),
        )
        self.open[symbol] = quote
        self.stats.quotes += 1

    def _check_fill(self, symbol: str, maker_book: OrderBook, now: float) -> None:
        quote = self.open.get(symbol)
        if quote is None:
            return
        if now - quote.placed_at > self.measure.max_rest_seconds:
            self._close(quote, "expired" if quote.armed else "unmeasurable", now)
            self.stats.expired_unfilled += 1
            del self.open[symbol]
            return
        best_bid = maker_book.best_bid
        if best_bid is None:
            return
        price = Decimal(str(quote.price))

        if not quote.armed:
            # Arm only once the touch is genuinely above us. A quote that
            # improves the best bid IS the touch, and public data cannot tell
            # whether it traded, so it is never counted as a fill.
            if best_bid > price:
                quote.armed = True
                quote.armed_at = now
            return
        if best_bid >= price:
            return
        # An armed quote that the book has now traded down through: a sweep
        # would have taken us.
        quote.outcome = "filled_pending"
        quote.filled_at = now
        quote.rested_seconds = now - quote.placed_at
        self.stats.fills += 1
        self.stats.resting_seconds += quote.rested_seconds
        del self.open[symbol]
        self.tracking.append(quote)
        log.info(
            "%s: virtual FILL @ %.8g after %.1fs resting (gap %.1fbps, edge %.1fbps)",
            symbol,
            quote.price,
            quote.rested_seconds,
            quote.gap_bps,
            quote.edge_bps,
        )

    def _advance_tracking(
        self, symbol: str, maker_book: OrderBook, hedge_book: OrderBook, now: float
    ) -> None:
        entry_side_done: List[VirtualQuote] = []
        for quote in self.tracking:
            if quote.symbol != symbol:
                continue
            entry = Decimal(str(quote.price))

            # Adverse excursion is measured on the price we could hedge at,
            # because that is the number that decides whether to bail.
            hedge_price = hedge_book.best_bid
            if hedge_price is not None and entry > ZERO:
                adverse = float((entry - hedge_price) / entry * BPS)
                quote.max_adverse_bps = max(quote.max_adverse_bps, adverse)
                quote.hedge_bps_at_end = float((hedge_price - entry) / entry * BPS)

            best_ask = maker_book.best_ask
            if (
                not quote.bounce_hit
                and best_ask is not None
                and best_ask > Decimal(str(quote.target))
            ):
                quote.bounce_hit = True
                quote.seconds_to_bounce = now - (quote.filled_at or now)
                self.stats.bounces += 1

            elapsed = now - (quote.filled_at or now)
            if quote.bounce_hit or elapsed >= self.measure.horizon_seconds:
                entry_side_done.append(quote)

        for quote in entry_side_done:
            self._settle(quote, now)
            self.tracking.remove(quote)

    def _settle(self, quote: VirtualQuote, now: float) -> None:
        """Book the outcome: the bounce we caught, or the hedge we would take."""
        if quote.bounce_hit:
            quote.realised_bps = (quote.target - quote.price) / quote.price * 10_000
        else:
            # No bounce inside the window: the hedge lands at whatever the other
            # venue is bid, which is exactly where adverse selection shows up.
            quote.realised_bps = quote.hedge_bps_at_end
        quote.outcome = "filled"
        self._close(quote, "filled", now)

    def _close(self, quote: VirtualQuote, outcome: str, now: float) -> None:
        if quote.outcome in ("resting", "cancelled"):
            quote.rested_seconds = now - quote.placed_at
            self.stats.resting_seconds += quote.rested_seconds
        quote.outcome = outcome
        row = quote.as_row()
        self.rows.append(row)
        if self.sink is not None:
            self.sink.write(json.dumps(row) + "\n")
            self.sink.flush()

    def finish(self, now: Optional[float] = None) -> None:
        """Flush everything still in flight so a short run is not silently empty."""
        now = now or time.time()
        for quote in list(self.open.values()):
            self._close(quote, "cut_short" if quote.armed else "unmeasurable", now)
        self.open.clear()
        for quote in list(self.tracking):
            self._settle(quote, now)
        self.tracking.clear()
        self.stats.watched_seconds = now - self._started
