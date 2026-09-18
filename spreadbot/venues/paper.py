"""Paper execution: real market data, simulated fills.

Fill model, stated plainly so results are read with the right amount of salt:

* A resting buy fills when the public best bid drops strictly below its price —
  i.e. the book was swept through our level. A sweep that stops exactly at our
  price does not count, because in reality it would have to clear the queue
  already resting there first.
* A resting sell is the mirror image.
* IOC/market orders fill at the book VWAP for their size and are rejected when
  that VWAP is worse than ``max_slippage_bps`` from the touch.
* Fees come from the venue's published maker/taker rates.

Paper mode is for wiring, sizing and plumbing checks. It is not a backtest and
its P&L is an upper bound, not an estimate.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Dict, List, Optional

from ..models import ZERO, Fill, MarketSpec, Order, OrderStatus, OrderType, Position, Side
from .base import ExecutionVenue, OrderRequest

log = logging.getLogger(__name__)


class PaperExecution(ExecutionVenue):
    def __init__(
        self,
        venue_key: str,
        allowed_side: Side,
        specs: Dict[str, MarketSpec],
        feed,
        *,
        starting_collateral: Decimal = Decimal(10_000),
    ) -> None:
        super().__init__(venue_key, allowed_side, specs)
        self.feed = feed
        self.starting_collateral = starting_collateral
        self.positions_book: Dict[str, Position] = {}
        self.fees_paid: Decimal = ZERO
        self._resting: Dict[int, Order] = {}

    # --------------------------------------------------------------- trading

    async def place(self, request: OrderRequest) -> Order:
        self.check_side(request)
        request = self.normalise(request)
        spec = self.spec(request.symbol)
        book = self.feed.book(request.symbol)

        order = Order(
            venue=self.venue_key,
            symbol=request.symbol,
            side=request.side,
            price=request.price if request.price is not None else ZERO,
            size=request.size,
            order_type=request.order_type,
            client_order_index=request.client_order_index,
            reduce_only=request.reduce_only,
            order_index=request.client_order_index,
        )

        if request.order_type in (OrderType.IOC, OrderType.MARKET):
            self._take(order, book, spec, request.max_slippage_bps)
            return self._record(order)

        touch = book.best_ask if request.side is Side.BUY else book.best_bid
        crosses = touch is not None and (
            (request.side is Side.BUY and request.price >= touch)
            or (request.side is Side.SELL and request.price <= touch)
        )
        if crosses:
            if request.order_type is OrderType.POST_ONLY:
                order.status = OrderStatus.REJECTED
                order.error = "post-only would cross"
                return self._record(order)
            self._take(order, book, spec, request.max_slippage_bps)
            return self._record(order)

        order.status = OrderStatus.OPEN
        self._resting[order.client_order_index] = order
        return self._record(order)

    async def cancel(self, order: Order) -> bool:
        resting = self._resting.pop(order.client_order_index, None)
        if resting is None:
            return False
        resting.status = OrderStatus.CANCELED if resting.remaining > ZERO else OrderStatus.FILLED
        return True

    async def cancel_all(self, symbol: Optional[str] = None) -> int:
        targets = [
            order
            for order in list(self._resting.values())
            if symbol is None or order.symbol == symbol.upper()
        ]
        for order in targets:
            await self.cancel(order)
        return len(targets)

    async def positions(self) -> Dict[str, Decimal]:
        return {symbol: pos.size for symbol, pos in self.positions_book.items() if pos.size != ZERO}

    async def collateral(self) -> Decimal:
        realized = sum((pos.realized_pnl for pos in self.positions_book.values()), ZERO)
        return self.starting_collateral + realized - self.fees_paid

    # ---------------------------------------------------------------- engine

    async def refresh(self, symbol: Optional[str] = None) -> None:
        self.poll()

    def poll(self) -> List[Fill]:
        """Check resting orders against the current book. Call every tick."""
        fills: List[Fill] = []
        for coi, order in list(self._resting.items()):
            book = self.feed.books.get(order.symbol)
            if book is None or not book.ready:
                continue
            # The book has to trade *through* our price, not merely reach it:
            # an order resting at the touch shares the level with everyone else
            # already queued there, so equality is not a fill.
            touched = (
                book.best_bid is not None and book.best_bid < order.price
                if order.side is Side.BUY
                else book.best_ask is not None and book.best_ask > order.price
            )
            if not touched:
                continue
            size = order.remaining
            order.apply_fill(size, order.price)
            self._resting.pop(coi, None)
            fill = Fill(
                venue=self.venue_key,
                symbol=order.symbol,
                side=order.side,
                price=order.price,
                size=size,
                order_index=order.order_index,
                client_order_index=coi,
                is_maker=True,
            )
            self._apply(fill, self.spec(order.symbol).maker_fee_bps)
            fills.append(fill)
            self._emit_fill(fill)
        return fills

    # --------------------------------------------------------------- private

    def _take(
        self,
        order: Order,
        book,
        spec: MarketSpec,
        max_slippage_bps: Optional[Decimal],
    ) -> None:
        quote = book.executable_vwap(order.side, order.size)
        if quote is None:
            order.status = OrderStatus.REJECTED
            order.error = "empty book"
            return
        vwap, fillable = quote
        if fillable < order.size:
            order.status = OrderStatus.REJECTED
            order.error = f"book too thin: only {fillable} of {order.size} available"
            return
        touch = book.best_ask if order.side is Side.BUY else book.best_bid
        if max_slippage_bps is not None and touch:
            slip = (vwap - touch) if order.side is Side.BUY else (touch - vwap)
            if slip / touch * Decimal(10_000) > max_slippage_bps:
                order.status = OrderStatus.REJECTED
                order.error = f"slippage {slip / touch * Decimal(10_000):.2f}bps over limit"
                return
        # A limit price still binds an IOC order.
        if order.price > ZERO:
            if order.side is Side.BUY and vwap > order.price:
                order.status = OrderStatus.REJECTED
                order.error = "IOC limit price not reachable"
                return
            if order.side is Side.SELL and vwap < order.price:
                order.status = OrderStatus.REJECTED
                order.error = "IOC limit price not reachable"
                return

        order.apply_fill(order.size, vwap)
        fill = Fill(
            venue=self.venue_key,
            symbol=order.symbol,
            side=order.side,
            price=vwap,
            size=order.size,
            order_index=order.order_index,
            client_order_index=order.client_order_index,
            is_maker=False,
        )
        self._apply(fill, spec.taker_fee_bps)
        self._emit_fill(fill)

    def _apply(self, fill: Fill, fee_bps: Decimal) -> None:
        position = self.positions_book.setdefault(
            fill.symbol, Position(self.venue_key, fill.symbol)
        )
        position.apply(fill)
        self.fees_paid += fill.notional * fee_bps / Decimal(10_000)
        log.info(
            "PAPER FILL %s %s %s %s @ %s (%s)",
            self.venue_key,
            fill.symbol,
            fill.side.value,
            fill.size,
            fill.price,
            "maker" if fill.is_maker else "taker",
        )
