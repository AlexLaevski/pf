"""Aggressive execution: hedging a fill, and unwinding a pair.

Everything here is a taker order with a bounded price. A hedge that does not go
through is worse than a hedge that costs a few bps, so the retry ladder widens
its price bound on each attempt and finishes with a bounded market order —
but it never widens past ``max_hedge_slippage_bps``, and it never silently
gives up: a failed hedge halts the bot.
"""

from __future__ import annotations

import asyncio
import logging
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
