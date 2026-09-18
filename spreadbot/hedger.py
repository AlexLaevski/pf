"""Aggressive execution: hedging a fill, and unwinding a pair.

Everything here is a taker order with a bounded price. A hedge that does not go
through is worse than a hedge that costs a few bps, so the retry ladder widens
its price bound on each attempt and finishes with a bounded market order —
but it never widens past ``max_hedge_slippage_bps``, and it never silently
gives up: a failed hedge halts the bot.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from .book import OrderBook
from .models import ZERO, Order, OrderStatus, OrderType, Side
from .venues.base import ExecutionVenue, OrderRequest

log = logging.getLogger(__name__)

BPS = Decimal(10_000)


@dataclass(frozen=True)
class TakerResult:
    filled: Decimal
    price: Decimal
    attempts: int
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.filled > ZERO and self.error is None


class Hedger:
    def __init__(
        self,
        venue: ExecutionVenue,
        *,
        max_slippage_bps: Decimal,
        attempts: int = 3,
        retry_delay: float = 0.15,
    ) -> None:
        self.venue = venue
        self.max_slippage_bps = max_slippage_bps
        self.attempts = max(1, attempts)
        self.retry_delay = retry_delay

    async def execute_maker_first(
        self,
        symbol: str,
        side: Side,
        size: Decimal,
        book: OrderBook,
        *,
        timeout_ms: int,
        offset_ticks: int = 0,
        reduce_only: bool = False,
        tag: str = "hedge",
    ) -> TakerResult:
        """Try to hedge as a maker, then fall back to taking.

        Saves the spread when the book gives us a moment, but the fallback is
        not optional: whatever the post-only order has not filled by
        ``timeout_ms`` is cancelled and taken. A hedge that is still pending is
        not a hedge.
        """
        spec = self.venue.spec(symbol)
        size = spec.quantize_size(size)
        if size <= ZERO or size < spec.min_base_amount:
            return TakerResult(ZERO, ZERO, 0, f"size {size} below venue minimum")

        touch = book.best_ask if side is Side.SELL else book.best_bid
        if touch is None:
            return await self.execute(symbol, side, size, book, reduce_only=reduce_only, tag=tag)

        offset = spec.tick * Decimal(max(0, offset_ticks))
        price = spec.quantize_price(
            touch + offset if side is Side.SELL else touch - offset, side=side
        )
        order: Optional[Order] = None
        try:
            order = await self.venue.place(
                OrderRequest(
                    symbol=symbol,
                    side=side,
                    size=size,
                    price=price,
                    order_type=OrderType.POST_ONLY,
                    reduce_only=reduce_only,
                    tag=f"{tag}:maker",
                )
            )
        except Exception as exc:
            log.warning("%s: maker hedge placement failed (%s); taking", self.venue.venue_key, exc)

        if order is not None and order.status is not OrderStatus.REJECTED:
            deadline = time.monotonic() + max(0.0, timeout_ms / 1_000)
            while time.monotonic() < deadline and order.remaining > ZERO:
                await asyncio.sleep(min(0.05, max(0.01, timeout_ms / 20_000)))
                with contextlib.suppress(Exception):
                    await self.venue.refresh(symbol)
            if order.remaining <= ZERO:
                log.info(
                    "%s: %s filled as maker %s @ %s",
                    self.venue.venue_key,
                    tag,
                    order.filled_size,
                    order.avg_fill_price,
                )
                return TakerResult(order.filled_size, order.avg_fill_price, 1)
            with contextlib.suppress(Exception):
                await self.venue.cancel(order)
            self.venue.forget(order)

        filled_as_maker = order.filled_size if order is not None else ZERO
        remaining = size - filled_as_maker
        if remaining <= ZERO:
            return TakerResult(filled_as_maker, order.avg_fill_price, 1)

        log.info(
            "%s: %s maker leg left %s unfilled after %dms; taking the rest",
            self.venue.venue_key,
            tag,
            remaining,
            timeout_ms,
        )
        taken = await self.execute(
            symbol, side, remaining, book, reduce_only=reduce_only, tag=f"{tag}:taker"
        )
        total = filled_as_maker + taken.filled
        if total <= ZERO:
            return taken
        notional = filled_as_maker * (order.avg_fill_price if order else ZERO) + taken.filled * taken.price
        return TakerResult(total, notional / total, taken.attempts + 1, taken.error)

    async def execute(
        self,
        symbol: str,
        side: Side,
        size: Decimal,
        book: OrderBook,
        *,
        reduce_only: bool = False,
        tag: str = "hedge",
    ) -> TakerResult:
        """Take ``size`` on ``side``, widening the price bound on each retry."""
        spec = self.venue.spec(symbol)
        size = spec.quantize_size(size)
        if size <= ZERO or size < spec.min_base_amount:
            return TakerResult(ZERO, ZERO, 0, f"size {size} below venue minimum")

        remaining = size
        filled = ZERO
        notional = ZERO
        last_error: Optional[str] = None

        for attempt in range(1, self.attempts + 1):
            reference = book.best_ask if side is Side.BUY else book.best_bid
            if reference is None:
                last_error = "empty book"
                await asyncio.sleep(self.retry_delay)
                continue

            # Widen the bound with each attempt but stay inside the hard cap.
            bound_bps = min(
                self.max_slippage_bps,
                self.max_slippage_bps * Decimal(attempt) / Decimal(self.attempts),
            )
            limit = (
                reference * (BPS + bound_bps) / BPS
                if side is Side.BUY
                else reference * (BPS - bound_bps) / BPS
            )
            order_type = OrderType.IOC if attempt < self.attempts else OrderType.MARKET
            try:
                order: Order = await self.venue.place(
                    OrderRequest(
                        symbol=symbol,
                        side=side,
                        size=remaining,
                        price=spec.quantize_price(limit, side=side),
                        order_type=order_type,
                        reduce_only=reduce_only,
                        max_slippage_bps=self.max_slippage_bps,
                        tag=f"{tag}:{attempt}",
                    )
                )
            except Exception as exc:
                last_error = str(exc)
                log.warning("%s: %s attempt %d raised: %s", self.venue.venue_key, tag, attempt, exc)
                await asyncio.sleep(self.retry_delay)
                continue

            if order.filled_size > ZERO:
                filled += order.filled_size
                notional += order.filled_size * order.avg_fill_price
                remaining -= order.filled_size
            if order.status is OrderStatus.REJECTED:
                last_error = order.error or "rejected"
            if remaining <= ZERO:
                return TakerResult(filled, notional / filled, attempt)
            log.warning(
                "%s: %s attempt %d left %s unfilled (%s)",
                self.venue.venue_key,
                tag,
                attempt,
                remaining,
                last_error or "partial",
            )
            await asyncio.sleep(self.retry_delay)

        price = notional / filled if filled > ZERO else ZERO
        return TakerResult(
            filled,
            price,
            self.attempts,
            last_error or f"{remaining} left unfilled after {self.attempts} attempts",
        )
