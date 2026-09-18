"""Thin async REST client for the public Lighter API.

Only the endpoints the bot actually needs. Deliberately independent of the
official SDK so market data, the scanner and paper mode all run with nothing
but ``aiohttp`` installed.
"""

from __future__ import annotations

import asyncio
import json
import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from ..models import ZERO, MarketSpec

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=10)

# orderBookOrders rejects anything above 250 with a bare "invalid param".
MAX_ORDER_LIMIT = 250


class LighterApiError(RuntimeError):
    def __init__(self, endpoint: str, code: Any, message: str) -> None:
        super().__init__(f"{endpoint}: [{code}] {message}")
        self.endpoint = endpoint
        self.code = code
        self.message = message


class LighterRateLimited(LighterApiError):
    """HTTP 429. Callers are expected to slow down, not just retry harder."""

    def __init__(self, endpoint: str, retry_after: Optional[float] = None) -> None:
        super().__init__(endpoint, 429, "Too Many Requests")
        self.retry_after = retry_after


def _dec(value: Any, default: Decimal = ZERO) -> Decimal:
    if value is None or value == "":
        return default
    return Decimal(str(value))


class LighterRest:
    """One HTTP session per venue."""

    def __init__(self, base_url: str, *, venue: str = "lighter", timeout: aiohttp.ClientTimeout = DEFAULT_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.venue = venue
        self._timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "LighterRest":
        await self.start()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    async def start(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _get(self, path: str, params: Optional[Dict[str, Any]] = None, *, retries: int = 3) -> Dict[str, Any]:
        await self.start()
        assert self._session is not None
        url = f"{self.base_url}/api/v1/{path.lstrip('/')}"
        last_exc: Optional[Exception] = None
        for attempt in range(retries):
            try:
                async with self._session.get(url, params=params) as resp:
                    status = resp.status
                    retry_after = _retry_after(resp.headers.get("Retry-After"))
                    text = await resp.text()

                if status == 429:
                    # Retrying at the same rate is how a bot blinds itself
                    # exactly when the market is busiest. Back off, and tell the
                    # caller so it can slow its whole loop down.
                    if attempt == retries - 1:
                        raise LighterRateLimited(path, retry_after)
                    await asyncio.sleep(retry_after or 0.5 * (2**attempt))
                    continue

                try:
                    body = json.loads(text)
                except ValueError:
                    # An HTML error page or an empty body: report the status,
                    # not a JSON parse error two layers from the cause.
                    raise LighterApiError(
                        path, status, f"non-JSON response: {text[:160]!r}"
                    ) from None

                if status >= 400 or not isinstance(body, dict):
                    raise LighterApiError(path, status, str(body)[:200])
                code = body.get("code")
                # 200 on the REST envelope; some endpoints omit the field entirely.
                if code not in (None, 200, 0):
                    raise LighterApiError(path, code, str(body.get("message", ""))[:200])
                return body
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_exc = exc
                if attempt == retries - 1:
                    break
                await asyncio.sleep(0.25 * (2**attempt))
        raise LighterApiError(path, "network", str(last_exc))

    # ---------------------------------------------------------------- metadata

    async def market_specs(self) -> Dict[str, MarketSpec]:
        """Every active market, keyed by symbol."""
        body = await self._get("orderBookDetails")
        specs: Dict[str, MarketSpec] = {}
        for entry in body.get("order_book_details", []):
            if entry.get("status") not in (None, "active"):
                continue
            symbol = str(entry["symbol"]).upper()
            price_decimals = int(entry.get("price_decimals", entry.get("supported_price_decimals", 2)))
            size_decimals = int(entry.get("size_decimals", entry.get("supported_size_decimals", 4)))
            specs[symbol] = MarketSpec(
                symbol=symbol,
                market_id=int(entry["market_id"]),
                price_decimals=price_decimals,
                size_decimals=size_decimals,
                min_base_amount=_dec(entry.get("min_base_amount")),
                min_quote_amount=_dec(entry.get("min_quote_amount")),
                # Fees arrive as percent strings ("0.0200" == 0.02%).
                maker_fee_bps=_dec(entry.get("maker_fee")) * Decimal(100),
                taker_fee_bps=_dec(entry.get("taker_fee")) * Decimal(100),
            )
        if not specs:
            raise LighterApiError("orderBookDetails", "empty", "no active markets returned")
        return specs

    async def mark_prices(self) -> Dict[str, Decimal]:
        body = await self._get("orderBookDetails")
        out: Dict[str, Decimal] = {}
        for entry in body.get("order_book_details", []):
            price = entry.get("mark_price")
            if price is not None:
                out[str(entry["symbol"]).upper()] = _dec(price)
        return out

    # -------------------------------------------------------------- order book

    async def order_book_levels(
        self, market_id: int, limit: int = 200
    ) -> Tuple[List[Tuple[Decimal, Decimal]], List[Tuple[Decimal, Decimal]]]:
        """Aggregate the raw order list into price levels.

        Returns ``(bids, asks)`` as ``(price, size)`` pairs, bids descending and
        asks ascending. ``limit`` counts orders per side, not price levels, and
        the endpoint rejects anything above ``MAX_ORDER_LIMIT`` outright, so it
        is clamped rather than passed through.
        """
        limit = max(1, min(MAX_ORDER_LIMIT, limit))
        body = await self._get("orderBookOrders", {"market_id": market_id, "limit": limit})
        return _aggregate(body.get("bids", [])), _aggregate(body.get("asks", []), ascending=True)

    async def funding_rates(self, exchange: str = "lighter") -> Dict[str, Decimal]:
        """Current hourly funding rate per symbol, as a fraction of notional.

        Positive means longs pay shorts. The endpoint reports several venues;
        ``exchange`` picks this deployment's own numbers.
        """
        body = await self._get("funding-rates")
        out: Dict[str, Decimal] = {}
        for entry in body.get("funding_rates", []):
            if str(entry.get("exchange", "")).lower() != exchange.lower():
                continue
            rate = entry.get("rate")
            if rate is None:
                continue
            out[str(entry["symbol"]).upper()] = _dec(rate)
        return out

    # ----------------------------------------------------------------- account

    async def account_state(self, account_index: int) -> Dict[str, Any]:
        """Public account snapshot: balances and open positions.

        This is the authoritative source the bot reconciles against — it needs
        no auth token, so it keeps working even when order-level streams do not.
        """
        body = await self._get("account", {"by": "index", "value": str(account_index)})
        accounts = body.get("accounts") or []
        if not accounts:
            raise LighterApiError("account", "empty", f"no account with index {account_index}")
        return accounts[0]

    async def positions(self, account_index: int) -> Dict[str, Decimal]:
        """Signed base position per symbol (positive long, negative short)."""
        account = await self.account_state(account_index)
        out: Dict[str, Decimal] = {}
        for entry in account.get("positions", []):
            size = _dec(entry.get("position"))
            if size == ZERO:
                continue
            sign = Decimal(int(entry.get("sign", 1)))
            out[str(entry["symbol"]).upper()] = size * sign
        return out

    async def collateral(self, account_index: int) -> Decimal:
        account = await self.account_state(account_index)
        return _dec(account.get("collateral"))


def _retry_after(header: Optional[str]) -> Optional[float]:
    """Seconds from a Retry-After header, when it carries a usable number."""
    if not header:
        return None
    try:
        return max(0.0, float(header))
    except ValueError:
        return None


def _aggregate(orders: List[Dict[str, Any]], *, ascending: bool = False) -> List[Tuple[Decimal, Decimal]]:
    levels: Dict[Decimal, Decimal] = {}
    for order in orders:
        price = _dec(order.get("price"))
        size = _dec(order.get("remaining_base_amount", order.get("size")))
        if price <= ZERO or size <= ZERO:
            continue
        levels[price] = levels.get(price, ZERO) + size
    return sorted(levels.items(), key=lambda kv: kv[0], reverse=not ascending)
