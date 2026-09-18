"""Execution interface shared by the live and paper venues."""

from __future__ import annotations

import abc
import asyncio
import itertools
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Optional

from ..models import ZERO, Fill, MarketSpec, Order, OrderType, Side


class SideNotAllowed(RuntimeError):
    """Raised when an order would breach the venue's directional mandate."""


@dataclass
class OrderRequest:
    symbol: str
    side: Side
    size: Decimal
    price: Optional[Decimal] = None          # None only for MARKET
    order_type: OrderType = OrderType.POST_ONLY
    reduce_only: bool = False
    max_slippage_bps: Optional[Decimal] = None   # for IOC / MARKET
    client_order_index: Optional[int] = None
    tag: str = ""


_counter = itertools.count(int(time.time() * 1000) % 1_000_000_000)


def next_client_order_index() -> int:
    """Unique across markets, as Lighter requires."""
    return next(_counter)


class ExecutionVenue(abc.ABC):
    """One trading account on one venue.

    Every implementation enforces the directional mandate: the venue's
    ``allowed_side`` is the only side that may *open* exposure. The opposite
    side is permitted only with ``reduce_only=True``, which is how a position
    gets closed.
    """

    def __init__(self, venue_key: str, allowed_side: Side, specs: Dict[str, MarketSpec]) -> None:
        self.venue_key = venue_key
        self.allowed_side = allowed_side
        self.specs = specs
        self.fills: "asyncio.Queue[Fill]" = asyncio.Queue()
        self.open_orders: Dict[int, Order] = {}

    # ------------------------------------------------------------- lifecycle

    async def start(self) -> None:  # pragma: no cover - trivial
        return None

    async def close(self) -> None:  # pragma: no cover - trivial
        return None

    # -------------------------------------------------------------- mandate

    def check_side(self, request: OrderRequest) -> None:
        if request.side is self.allowed_side:
            return
        if request.reduce_only:
            return
        raise SideNotAllowed(
            f"{self.venue_key} is {self.allowed_side.value}-only; a {request.side.value} order "
            f"must be reduce_only (it may close, never open, the opposite exposure)"
        )

    def spec(self, symbol: str) -> MarketSpec:
        try:
            return self.specs[symbol.upper()]
        except KeyError as exc:
            raise KeyError(f"{self.venue_key}: unknown market {symbol}") from exc

    def normalise(self, request: OrderRequest) -> OrderRequest:
        """Snap price and size to the venue grid and validate minimums."""
        spec = self.spec(request.symbol)
        size = spec.quantize_size(request.size)
        if size < spec.min_base_amount or size <= ZERO:
            raise ValueError(
                f"{self.venue_key}:{request.symbol} size {request.size} below min_base_amount "
                f"{spec.min_base_amount}"
            )
        price = request.price
        if price is not None:
            price = spec.quantize_price(price, side=request.side)
            if price <= ZERO:
                raise ValueError(f"{self.venue_key}:{request.symbol} non-positive price {request.price}")
            if price * size < spec.min_quote_amount:
                raise ValueError(
                    f"{self.venue_key}:{request.symbol} notional {price * size} below "
                    f"min_quote_amount {spec.min_quote_amount}"
                )
        elif request.order_type is not OrderType.MARKET:
            raise ValueError("only MARKET orders may omit a price")
        return OrderRequest(
            symbol=request.symbol.upper(),
            side=request.side,
            size=size,
            price=price,
            order_type=request.order_type,
            reduce_only=request.reduce_only,
            max_slippage_bps=request.max_slippage_bps,
            client_order_index=request.client_order_index or next_client_order_index(),
            tag=request.tag,
        )

    # --------------------------------------------------------------- trading

    @abc.abstractmethod
    async def place(self, request: OrderRequest) -> Order:
        """Submit an order. Raises on rejection."""

    @abc.abstractmethod
    async def cancel(self, order: Order) -> bool:
        """Cancel a resting order. Returns False when it was already gone."""

    @abc.abstractmethod
    async def cancel_all(self, symbol: Optional[str] = None) -> int:
        """Cancel everything (optionally only one market). Returns the count."""

    @abc.abstractmethod
    async def positions(self) -> Dict[str, Decimal]:
        """Signed base position per symbol; positive long, negative short."""

    @abc.abstractmethod
    async def collateral(self) -> Decimal:
        """Account collateral in quote currency."""

    async def refresh(self, symbol: Optional[str] = None) -> None:
        """Pull order state forward and emit any new fills onto :attr:`fills`."""
        return None

    # --------------------------------------------------------------- helpers

    def _record(self, order: Order) -> Order:
        self.open_orders[order.client_order_index] = order
        return order

    def _emit_fill(self, fill: Fill) -> None:
        self.fills.put_nowait(fill)

    def forget(self, order: Order) -> None:
        self.open_orders.pop(order.client_order_index, None)

    def resting(self, symbol: Optional[str] = None) -> Dict[int, Order]:
        return {
            coi: order
            for coi, order in self.open_orders.items()
            if not order.is_terminal and (symbol is None or order.symbol == symbol.upper())
        }
