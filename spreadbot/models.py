"""Domain models shared by every component.

Everything that touches a price or a size uses :class:`decimal.Decimal`.
Lighter returns prices/sizes as decimal strings and expects scaled integers on
the way in, so float rounding is never acceptable here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Iterable, Optional

ZERO = Decimal(0)


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY

    @property
    def is_ask(self) -> bool:
        """Lighter encodes direction as ``is_ask``."""
        return self is Side.SELL

    @property
    def sign(self) -> Decimal:
        """+1 when the side increases inventory, -1 when it decreases it."""
        return Decimal(1) if self is Side.BUY else Decimal(-1)


class OrderType(str, Enum):
    LIMIT = "limit"
    POST_ONLY = "post_only"
    IOC = "ioc"
    MARKET = "market"


class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


@dataclass(frozen=True)
class MarketSpec:
    """Static per-market metadata, taken from ``/api/v1/orderBookDetails``."""

    symbol: str
    market_id: int
    price_decimals: int
    size_decimals: int
    min_base_amount: Decimal
    min_quote_amount: Decimal
    maker_fee_bps: Decimal = ZERO
    taker_fee_bps: Decimal = ZERO

    @property
    def tick(self) -> Decimal:
        """Smallest price increment."""
        return Decimal(1).scaleb(-self.price_decimals)

    @property
    def lot(self) -> Decimal:
        """Smallest size increment."""
        return Decimal(1).scaleb(-self.size_decimals)

    def quantize_price(self, price: Decimal, *, side: Optional[Side] = None) -> Decimal:
        """Snap ``price`` onto the tick grid.

        With ``side`` given the price is rounded conservatively: a bid rounds
        down and an ask rounds up, so quantisation can never make a quote more
        aggressive than intended.
        """
        ticks = price / self.tick
        if side is Side.BUY:
            ticks = ticks.to_integral_value(rounding="ROUND_FLOOR")
        elif side is Side.SELL:
            ticks = ticks.to_integral_value(rounding="ROUND_CEILING")
        else:
            ticks = ticks.to_integral_value(rounding="ROUND_HALF_EVEN")
        return (ticks * self.tick).quantize(self.tick)

    def quantize_size(self, size: Decimal) -> Decimal:
        """Snap ``size`` down onto the lot grid (never over-order)."""
        lots = (size / self.lot).to_integral_value(rounding="ROUND_FLOOR")
        return (lots * self.lot).quantize(self.lot)

    def price_to_int(self, price: Decimal) -> int:
        return int((price.scaleb(self.price_decimals)).to_integral_value(rounding="ROUND_HALF_EVEN"))

    def size_to_int(self, size: Decimal) -> int:
        return int((size.scaleb(self.size_decimals)).to_integral_value(rounding="ROUND_HALF_EVEN"))


@dataclass(frozen=True)
class Level:
    price: Decimal
    size: Decimal

    @property
    def notional(self) -> Decimal:
        return self.price * self.size


@dataclass
class Quote:
    """A maker order the strategy wants to have resting on a venue."""

    venue: str
    symbol: str
    side: Side
    price: Decimal
    size: Decimal
    # Diagnostics, carried through to the logs so a quote can be explained.
    wall_price: Optional[Decimal] = None
    gap_bps: Optional[Decimal] = None
    edge_bps: Optional[Decimal] = None
    hedge_price: Optional[Decimal] = None

    def differs_from(self, other: Optional["Quote"], *, tick: Decimal, size_tol: Decimal) -> bool:
        if other is None:
            return True
        if self.side is not other.side:
            return True
        if abs(self.price - other.price) >= tick:
            return True
        if other.size == ZERO:
            return True
        return abs(self.size - other.size) / other.size >= size_tol


@dataclass
class Order:
    venue: str
    symbol: str
    side: Side
    price: Decimal
    size: Decimal
    order_type: OrderType
    client_order_index: int
    reduce_only: bool = False
    order_index: Optional[int] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_size: Decimal = ZERO
    avg_fill_price: Decimal = ZERO
    created_at: float = field(default_factory=time.time)
    error: Optional[str] = None

    @property
    def remaining(self) -> Decimal:
        return max(ZERO, self.size - self.filled_size)

    @property
    def is_terminal(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED)

    def apply_fill(self, size: Decimal, price: Decimal) -> None:
        if size <= ZERO:
            return
        notional = self.avg_fill_price * self.filled_size + price * size
        self.filled_size += size
        self.avg_fill_price = notional / self.filled_size
        self.status = OrderStatus.FILLED if self.remaining <= ZERO else OrderStatus.PARTIAL


@dataclass(frozen=True)
class Fill:
    venue: str
    symbol: str
    side: Side
    price: Decimal
    size: Decimal
    order_index: Optional[int] = None
    client_order_index: Optional[int] = None
    is_maker: bool = True
    ts: float = field(default_factory=time.time)

    @property
    def notional(self) -> Decimal:
        return self.price * self.size


@dataclass
class Position:
    """Signed inventory on a single venue. Positive means long."""

    venue: str
    symbol: str
    size: Decimal = ZERO
    entry_price: Decimal = ZERO
    realized_pnl: Decimal = ZERO

    def apply(self, fill: Fill) -> None:
        signed = fill.size * fill.side.sign
        new_size = self.size + signed
        if self.size == ZERO or (self.size > ZERO) == (signed > ZERO):
            # Opening or adding: weighted-average the entry.
            total = abs(self.size) + abs(signed)
            if total > ZERO:
                self.entry_price = (
                    self.entry_price * abs(self.size) + fill.price * abs(signed)
                ) / total
        else:
            # Reducing or flipping: bank the realised part.
            closed = min(abs(self.size), abs(signed))
            direction = Decimal(1) if self.size > ZERO else Decimal(-1)
            self.realized_pnl += (fill.price - self.entry_price) * closed * direction
            if abs(signed) > abs(self.size):
                self.entry_price = fill.price
        self.size = new_size
        if self.size == ZERO:
            self.entry_price = ZERO

    def unrealized_pnl(self, mark: Decimal) -> Decimal:
        if self.size == ZERO:
            return ZERO
        return (mark - self.entry_price) * self.size


@dataclass
class PairPosition:
    """The hedged pair: long on the maker venue, short on the hedge venue."""

    symbol: str
    long_venue: str
    short_venue: str
    long_size: Decimal = ZERO
    short_size: Decimal = ZERO
    long_entry: Decimal = ZERO
    short_entry: Decimal = ZERO
    opened_at: Optional[float] = None
    realized_pnl: Decimal = ZERO

    @property
    def unhedged(self) -> Decimal:
        """Long base that has no matching short yet. Must be driven to zero."""
        return self.long_size - self.short_size

    @property
    def matched(self) -> Decimal:
        return min(self.long_size, self.short_size)

    @property
    def is_flat(self) -> bool:
        return self.long_size <= ZERO and self.short_size <= ZERO

    @property
    def entry_spread_bps(self) -> Decimal:
        """Spread captured at entry, in bps of the long entry price."""
        if self.long_entry <= ZERO or self.matched <= ZERO:
            return ZERO
        return (self.short_entry - self.long_entry) / self.long_entry * Decimal(10_000)

    def add_long(self, price: Decimal, size: Decimal) -> None:
        total = self.long_size + size
        if total > ZERO:
            self.long_entry = (self.long_entry * self.long_size + price * size) / total
        self.long_size = total
        if self.opened_at is None:
            self.opened_at = time.time()

    def add_short(self, price: Decimal, size: Decimal) -> None:
        total = self.short_size + size
        if total > ZERO:
            self.short_entry = (self.short_entry * self.short_size + price * size) / total
        self.short_size = total

    def reduce_long(self, price: Decimal, size: Decimal) -> None:
        size = min(size, self.long_size)
        self.realized_pnl += (price - self.long_entry) * size
        self.long_size -= size
        if self.long_size <= ZERO:
            self.long_size = ZERO
            self.long_entry = ZERO

    def reduce_short(self, price: Decimal, size: Decimal) -> None:
        size = min(size, self.short_size)
        self.realized_pnl += (self.short_entry - price) * size
        self.short_size -= size
        if self.short_size <= ZERO:
            self.short_size = ZERO
            self.short_entry = ZERO
        if self.is_flat:
            self.opened_at = None


def sum_sizes(levels: Iterable[Level]) -> Decimal:
    return sum((level.size for level in levels), ZERO)
