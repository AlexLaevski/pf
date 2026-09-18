"""Live execution against a Lighter deployment via the official SDK.

The SDK is imported lazily: market data, the scanner and paper mode all work
without it installed.

Fill detection is deliberately belt-and-braces. ``sync_orders`` polls our own
active orders and turns a drop in ``remaining_base_amount`` into a fill, which
is the fast path. The authoritative path is position reconciliation in the
engine, which reads the public ``/account`` endpoint and hedges whatever
difference it finds. An order that disappears from the active list is *not*
assumed to be filled — it is marked unknown and left to the reconciler.
"""

from __future__ import annotations

import asyncio
import logging
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from ..config import VenueConfig
from ..models import ZERO, Fill, MarketSpec, Order, OrderStatus, OrderType
from .base import ExecutionVenue, OrderRequest
from .rest import LighterRest

log = logging.getLogger(__name__)

AUTH_TOKEN_TTL_SECONDS = 8 * 60


class LighterExecutionError(RuntimeError):
    pass


class LighterExecution(ExecutionVenue):
    def __init__(self, cfg: VenueConfig, specs: Dict[str, MarketSpec], rest: LighterRest) -> None:
        super().__init__(cfg.key, cfg.allowed_side, specs)
        self.cfg = cfg
        self.rest = rest
        self._signer: Any = None
        self._order_api: Any = None
        self._api_client: Any = None
        self._auth_token: Optional[str] = None
        self._auth_expires_at: float = 0.0
        self._lighter: Any = None

    # ------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        try:
            import lighter
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise LighterExecutionError(
                "live mode needs the Lighter SDK: pip install 'lighter-spread-bot[live]'"
            ) from exc

        self._lighter = lighter
        private_key = self.cfg.require_credentials()
        self._signer = lighter.SignerClient(
            url=self.cfg.base_url,
            account_index=self.cfg.account_index,
            api_private_keys={self.cfg.api_key_index: private_key},
        )
        err = self._signer.check_client()
        if err is not None:
            raise LighterExecutionError(f"{self.venue_key}: signer check failed: {err}")

        self._api_client = lighter.ApiClient(lighter.Configuration(host=self.cfg.base_url))
        self._order_api = lighter.OrderApi(self._api_client)
        log.info(
            "%s: live execution ready (account=%s, api_key_index=%s, %s-only)",
            self.venue_key,
            self.cfg.account_index,
            self.cfg.api_key_index,
            self.allowed_side.value,
        )

    async def close(self) -> None:
        for closer in (self._signer, self._api_client):
            if closer is None:
                continue
            try:
                result = closer.close()
                if hasattr(result, "__await__"):
                    await result
            except Exception:  # pragma: no cover - best effort teardown
                log.debug("%s: error closing client", self.venue_key, exc_info=True)
        self._signer = None
        self._api_client = None
        self._order_api = None

    # ---------------------------------------------------------------- trading

    async def place(self, request: OrderRequest) -> Order:
        self.check_side(request)
        request = self.normalise(request)
        spec = self.spec(request.symbol)
        signer = self._require_signer()
        lighter = self._lighter

        order = Order(
            venue=self.venue_key,
            symbol=request.symbol,
            side=request.side,
            price=request.price if request.price is not None else ZERO,
            size=request.size,
            order_type=request.order_type,
            client_order_index=request.client_order_index,
            reduce_only=request.reduce_only,
        )

        base_amount = spec.size_to_int(request.size)
        try:
            if request.order_type is OrderType.MARKET:
                if request.price is None:
                    raise ValueError("market orders still need an avg_execution_price bound")
                _, resp, err = await signer.create_market_order(
                    market_index=spec.market_id,
                    client_order_index=request.client_order_index,
                    base_amount=base_amount,
                    avg_execution_price=spec.price_to_int(request.price),
                    is_ask=request.side.is_ask,
                    reduce_only=request.reduce_only,
                    api_key_index=self.cfg.api_key_index,
                )
            else:
                time_in_force, order_expiry = self._tif(request.order_type)
                _, resp, err = await signer.create_order(
                    market_index=spec.market_id,
                    client_order_index=request.client_order_index,
                    base_amount=base_amount,
                    price=spec.price_to_int(request.price),
                    is_ask=request.side.is_ask,
                    order_type=lighter.SignerClient.ORDER_TYPE_LIMIT,
                    time_in_force=time_in_force,
                    reduce_only=request.reduce_only,
                    order_expiry=order_expiry,
                    api_key_index=self.cfg.api_key_index,
                )
        except Exception as exc:
            order.status = OrderStatus.REJECTED
            order.error = str(exc)
            log.error("%s: order submit raised: %s", self.venue_key, exc)
            return self._record(order)

        if err is not None:
            order.status = OrderStatus.REJECTED
            order.error = str(err)
            log.warning(
                "%s: %s %s %s @ %s rejected: %s",
                self.venue_key,
                request.symbol,
                request.side.value,
                request.size,
                request.price,
                err,
            )
            return self._record(order)

        order.status = OrderStatus.OPEN
        order.order_index = _extract_order_index(resp)
        if request.order_type in (OrderType.IOC, OrderType.MARKET):
            # An IOC never rests, so it will not show up in the active-order
            # poll. Settle it now, or the engine would keep believing the
            # position is unhedged.
            await self._settle_taker(order)
        log.info(
            "%s: placed %s %s %s @ %s (%s%s) coi=%s",
            self.venue_key,
            request.symbol,
            request.side.value,
            request.size,
            request.price,
            request.order_type.value,
            ", reduce_only" if request.reduce_only else "",
            request.client_order_index,
        )
        return self._record(order)

    def _tif(self, order_type: OrderType) -> Tuple[int, int]:
        sc = self._lighter.SignerClient
        if order_type is OrderType.POST_ONLY:
            return sc.ORDER_TIME_IN_FORCE_POST_ONLY, sc.DEFAULT_28_DAY_ORDER_EXPIRY
        if order_type is OrderType.IOC:
            return sc.ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL, sc.DEFAULT_IOC_EXPIRY
        return sc.ORDER_TIME_IN_FORCE_GOOD_TILL_TIME, sc.DEFAULT_28_DAY_ORDER_EXPIRY

    async def cancel(self, order: Order) -> bool:
        signer = self._require_signer()
        spec = self.spec(order.symbol)
        if order.order_index is None:
            await self.sync_orders(order.symbol)
        if order.order_index is None:
            # Never leave a quote resting because we lost its index: the venue
            # holds at most our own orders in this market, so cancel the market.
            log.warning(
                "%s: no order_index for coi=%s; cancelling all %s orders",
                self.venue_key,
                order.client_order_index,
                order.symbol,
            )
            await self.cancel_all(order.symbol)
            order.status = OrderStatus.CANCELED
            return True
        _, _, err = await signer.cancel_order(
            market_index=spec.market_id,
            order_index=order.order_index,
            api_key_index=self.cfg.api_key_index,
        )
        if err is not None:
            log.warning("%s: cancel failed for %s: %s", self.venue_key, order.order_index, err)
            return False
        order.status = OrderStatus.CANCELED
        return True

    async def cancel_all(self, symbol: Optional[str] = None) -> int:
        signer = self._require_signer()
        sc = self._lighter.SignerClient
        market_index = self.spec(symbol).market_id if symbol else sc.NIL_MARKET_INDEX
        _, _, err = await signer.cancel_all_orders(
            time_in_force=sc.CANCEL_ALL_TIF_IMMEDIATE,
            timestamp_ms=int(time.time() * 1_000),
            cancel_all_market_index=market_index,
            api_key_index=self.cfg.api_key_index,
        )
        if err is not None:
            raise LighterExecutionError(f"{self.venue_key}: cancel_all failed: {err}")
        count = 0
        for order in list(self.open_orders.values()):
            if order.is_terminal:
                continue
            if symbol is None or order.symbol == symbol.upper():
                order.status = OrderStatus.CANCELED
                count += 1
        return count

    # ------------------------------------------------------------ order state

    async def refresh(self, symbol: Optional[str] = None) -> None:
        await self.sync_orders(symbol)

    async def sync_orders(self, symbol: Optional[str] = None) -> List[Fill]:
        """Reconcile our tracked orders against the venue's active-order list.

        Returns the fills inferred from shrinking remaining size. Orders that
        vanished are marked ``PENDING`` with an explanatory error rather than
        being guessed as filled — the engine's position reconciler settles them.
        """
        if self._order_api is None:
            return []
        tracked = self.resting(symbol)
        if not tracked:
            return []

        token = await self._auth()
        market_id = self.spec(symbol).market_id if symbol else None
        try:
            result = await self._order_api.account_active_orders(
                authorization=token,
                account_index=self.cfg.account_index,
                market_id=market_id,
            )
        except Exception as exc:
            log.warning("%s: active order poll failed: %s", self.venue_key, exc)
            return []

        by_coi: Dict[int, Any] = {}
        for raw in getattr(result, "orders", None) or []:
            coi = _int_or_none(getattr(raw, "client_order_index", None))
            if coi is not None:
                by_coi[coi] = raw

        fills: List[Fill] = []
        for coi, order in tracked.items():
            raw = by_coi.get(coi)
            if raw is None:
                order.error = "not in active orders; awaiting position reconciliation"
                continue
            if order.order_index is None:
                order.order_index = _int_or_none(getattr(raw, "order_index", None))
            filled_now = _decimal_or_none(getattr(raw, "filled_base_amount", None))
            if filled_now is None:
                remaining = _decimal_or_none(getattr(raw, "remaining_base_amount", None))
                if remaining is None:
                    continue
                filled_now = max(ZERO, order.size - remaining)
            delta = filled_now - order.filled_size
            if delta <= ZERO:
                continue
            filled_quote = _decimal_or_none(getattr(raw, "filled_quote_amount", None))
            price = (
                filled_quote / filled_now
                if filled_quote and filled_now > ZERO
                else _decimal_or_none(getattr(raw, "price", None)) or order.price
            )
            order.apply_fill(delta, price)
            fill = Fill(
                venue=self.venue_key,
                symbol=order.symbol,
                side=order.side,
                price=price,
                size=delta,
                order_index=order.order_index,
                client_order_index=coi,
                is_maker=order.order_type is OrderType.POST_ONLY,
            )
            fills.append(fill)
            self._emit_fill(fill)
            log.info(
                "%s: fill %s %s %s @ %s (coi=%s)",
                self.venue_key,
                order.symbol,
                order.side.value,
                delta,
                price,
                coi,
            )
        return fills

    async def _settle_taker(
        self, order: Order, *, attempts: int = 8, delay: float = 0.15
    ) -> None:
        """Resolve an IOC/market order's outcome from the inactive-order list."""
        if self._order_api is None:
            return
        spec = self.spec(order.symbol)
        token = await self._auth()
        for _ in range(attempts):
            try:
                result = await self._order_api.account_inactive_orders(
                    authorization=token,
                    account_index=self.cfg.account_index,
                    market_id=spec.market_id,
                    limit=25,
                )
            except Exception as exc:
                log.warning("%s: inactive order poll failed: %s", self.venue_key, exc)
                await asyncio.sleep(delay)
                continue

            for raw in getattr(result, "orders", None) or []:
                if _int_or_none(getattr(raw, "client_order_index", None)) != order.client_order_index:
                    continue
                if order.order_index is None:
                    order.order_index = _int_or_none(getattr(raw, "order_index", None))
                filled = _decimal_or_none(getattr(raw, "filled_base_amount", None)) or ZERO
                quote = _decimal_or_none(getattr(raw, "filled_quote_amount", None))
                delta = filled - order.filled_size
                if delta > ZERO:
                    price = (quote / filled) if quote and filled > ZERO else order.price
                    order.apply_fill(delta, price)
                    fill = Fill(
                        venue=self.venue_key,
                        symbol=order.symbol,
                        side=order.side,
                        price=price,
                        size=delta,
                        order_index=order.order_index,
                        client_order_index=order.client_order_index,
                        is_maker=False,
                    )
                    self._emit_fill(fill)
                if filled <= ZERO:
                    order.status = OrderStatus.CANCELED
                    order.error = "IOC expired unfilled"
                return
            await asyncio.sleep(delay)

        order.error = "taker order outcome unknown; left to position reconciliation"
        log.error(
            "%s: could not settle taker order coi=%s - reconciliation will pick it up",
            self.venue_key,
            order.client_order_index,
        )

    # ----------------------------------------------------------------- account

    async def positions(self) -> Dict[str, Decimal]:
        return await self.rest.positions(self.cfg.account_index)

    async def collateral(self) -> Decimal:
        return await self.rest.collateral(self.cfg.account_index)

    # ----------------------------------------------------------------- private

    def _require_signer(self) -> Any:
        if self._signer is None:
            raise LighterExecutionError(f"{self.venue_key}: execution venue not started")
        return self._signer

    async def _auth(self) -> str:
        now = time.time()
        if self._auth_token and now < self._auth_expires_at:
            return self._auth_token
        token, err = self._require_signer().create_auth_token_with_expiry(
            api_key_index=self.cfg.api_key_index
        )
        if err is not None:
            raise LighterExecutionError(f"{self.venue_key}: auth token failed: {err}")
        self._auth_token = token
        self._auth_expires_at = now + AUTH_TOKEN_TTL_SECONDS
        return token


def _extract_order_index(resp: Any) -> Optional[int]:
    for attr in ("order_index", "tx_hash", "index"):
        value = getattr(resp, attr, None)
        if attr == "order_index" and value is not None:
            return _int_or_none(value)
    return None


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _decimal_or_none(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None
