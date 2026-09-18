"""Level-2 order book: snapshot + delta maintenance and the analytics the
strategy needs (executable VWAP, depth, distance-weighted density).

Lighter streams a full snapshot on ``subscribed/order_book`` and then price-level
deltas on ``update/order_book``; a delta with ``size == 0`` removes the level.
"""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .models import ZERO, Level, Side


class OrderBook:
    """A single market's book on a single venue."""

    __slots__ = (
        "venue",
        "symbol",
        "_bids",
        "_asks",
        "_bid_cache",
        "_ask_cache",
        "_dirty_bids",
        "_dirty_asks",
        "updated_at",
        "sequence",
        "ready",
    )

    def __init__(self, venue: str, symbol: str) -> None:
        self.venue = venue
        self.symbol = symbol
        self._bids: Dict[Decimal, Decimal] = {}
        self._asks: Dict[Decimal, Decimal] = {}
        self._bid_cache: List[Level] = []
        self._ask_cache: List[Level] = []
        self._dirty_bids = True
        self._dirty_asks = True
        self.updated_at: float = 0.0
        self.sequence: int = 0
        self.ready: bool = False

    # ------------------------------------------------------------------ writes

    def apply_snapshot(self, bids: Iterable[Tuple[Decimal, Decimal]], asks: Iterable[Tuple[Decimal, Decimal]]) -> None:
        self._bids = {p: s for p, s in bids if s > ZERO}
        self._asks = {p: s for p, s in asks if s > ZERO}
        self._dirty_bids = self._dirty_asks = True
        self.updated_at = time.time()
        self.sequence += 1
        self.ready = bool(self._bids and self._asks)

    def apply_delta(self, bids: Iterable[Tuple[Decimal, Decimal]], asks: Iterable[Tuple[Decimal, Decimal]]) -> None:
        for price, size in bids:
            if size <= ZERO:
                self._bids.pop(price, None)
            else:
                self._bids[price] = size
            self._dirty_bids = True
        for price, size in asks:
            if size <= ZERO:
                self._asks.pop(price, None)
            else:
                self._asks[price] = size
            self._dirty_asks = True
        self.updated_at = time.time()
        self.sequence += 1
        self.ready = bool(self._bids and self._asks)

    def clear(self) -> None:
        self._bids.clear()
        self._asks.clear()
        self._dirty_bids = self._dirty_asks = True
        self.ready = False

    # ------------------------------------------------------------------- reads

    @property
    def bids(self) -> Sequence[Level]:
        """Bids, best (highest) first."""
        if self._dirty_bids:
            self._bid_cache = [
                Level(p, self._bids[p]) for p in sorted(self._bids, reverse=True)
            ]
            self._dirty_bids = False
        return self._bid_cache

    @property
    def asks(self) -> Sequence[Level]:
        """Asks, best (lowest) first."""
        if self._dirty_asks:
            self._ask_cache = [Level(p, self._asks[p]) for p in sorted(self._asks)]
            self._dirty_asks = False
        return self._ask_cache

    def side(self, side: Side) -> Sequence[Level]:
        """The side of the book a resting order of ``side`` would join."""
        return self.bids if side is Side.BUY else self.asks

    def opposite(self, side: Side) -> Sequence[Level]:
        """The side an aggressive order of ``side`` would consume."""
        return self.asks if side is Side.BUY else self.bids

    @property
    def best_bid(self) -> Optional[Decimal]:
        bids = self.bids
        return bids[0].price if bids else None

    @property
    def best_ask(self) -> Optional[Decimal]:
        asks = self.asks
        return asks[0].price if asks else None

    @property
    def mid(self) -> Optional[Decimal]:
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2

    @property
    def spread_bps(self) -> Optional[Decimal]:
        bid, ask, mid = self.best_bid, self.best_ask, self.mid
        if bid is None or ask is None or not mid:
            return None
        return (ask - bid) / mid * Decimal(10_000)

    def age_ms(self, now: Optional[float] = None) -> float:
        if self.updated_at == 0.0:
            return float("inf")
        return ((now or time.time()) - self.updated_at) * 1_000

    def is_fresh(self, max_age_ms: float, now: Optional[float] = None) -> bool:
        return self.ready and self.age_ms(now) <= max_age_ms

    # --------------------------------------------------------------- analytics

    def executable_vwap(self, side: Side, size: Decimal) -> Optional[Tuple[Decimal, Decimal]]:
        """Average price of taking ``size`` with a market order of ``side``.

        Returns ``(vwap, fillable_size)``; ``None`` if the book is empty on the
        side being consumed. ``fillable_size`` is less than ``size`` when the
        visible book cannot absorb the whole order, and callers must treat that
        as "do not trade" rather than "trade smaller than asked".
        """
        levels = self.opposite(side)
        if not levels:
            return None
        remaining = size
        notional = ZERO
        for level in levels:
            take = min(remaining, level.size)
            notional += take * level.price
            remaining -= take
            if remaining <= ZERO:
                break
        filled = size - remaining
        if filled <= ZERO:
            return None
        return notional / filled, filled

    def worst_price(self, side: Side, size: Decimal) -> Optional[Decimal]:
        """Price of the last level a market order of ``size`` would touch."""
        levels = self.opposite(side)
        remaining = size
        last: Optional[Decimal] = None
        for level in levels:
            last = level.price
            remaining -= level.size
            if remaining <= ZERO:
                return last
        return None

    def depth_notional(self, side: Side, within_bps: Decimal) -> Decimal:
        """Resting notional on ``side`` within ``within_bps`` of the mid."""
        mid = self.mid
        if mid is None:
            return ZERO
        limit = mid * within_bps / Decimal(10_000)
        total = ZERO
        for level in self.side(side):
            if abs(level.price - mid) > limit:
                break
            total += level.notional
        return total

    def size_at_or_better(self, side: Side, price: Decimal) -> Decimal:
        """Base size resting at prices at least as good as ``price``.

        For a bid that is the queue ahead of us at higher prices; it is what a
        sweep has to clear before it reaches our order.
        """
        total = ZERO
        for level in self.side(side):
            better = level.price > price if side is Side.BUY else level.price < price
            if better or level.price == price:
                total += level.size
            else:
                break
        return total

    def snapshot_levels(self, depth: int = 10) -> Tuple[List[Level], List[Level]]:
        return list(self.bids[:depth]), list(self.asks[:depth])

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<OrderBook {self.venue}:{self.symbol} "
            f"bid={self.best_bid} ask={self.best_ask} "
            f"levels={len(self._bids)}/{len(self._asks)} age={self.age_ms():.0f}ms>"
        )
