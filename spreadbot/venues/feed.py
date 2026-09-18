"""Market data feed for a Lighter deployment.

Primary transport is the ``/stream`` WebSocket (snapshot on subscribe, price
level deltas afterwards). A REST polling mode is available for environments
where the WebSocket cannot be reached, and the WebSocket mode falls back to it
automatically after repeated connection failures rather than leaving the
strategy with a stale book.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..book import OrderBook
from ..config import FeedConfig
from ..models import ZERO, MarketSpec
from .rest import LighterRateLimited, LighterRest

log = logging.getLogger(__name__)

WS_FAILURES_BEFORE_FALLBACK = 3
REST_MAX_INTERVAL_SECONDS = 30.0
REST_INTERVAL_DECAY = 0.8


def _levels(raw: Iterable[Dict[str, Any]]) -> List[Tuple[Decimal, Decimal]]:
    out: List[Tuple[Decimal, Decimal]] = []
    for entry in raw or ():
        try:
            price = Decimal(str(entry["price"]))
        except (KeyError, TypeError, ValueError):
            continue
        size_raw = entry.get("size", entry.get("remaining_base_amount", "0"))
        try:
            size = Decimal(str(size_raw))
        except (TypeError, ValueError):
            continue
        out.append((price, size))
    return out


class LighterFeed:
    """Keeps an :class:`OrderBook` per symbol up to date for one venue."""

    def __init__(
        self,
        venue_key: str,
        *,
        ws_url: str,
        rest: LighterRest,
        config: FeedConfig,
        specs: Dict[str, MarketSpec],
    ) -> None:
        self.venue_key = venue_key
        self.ws_url = ws_url
        self.rest = rest
        self.config = config
        self.specs = specs
        self.books: Dict[str, OrderBook] = {}
        self.updated = asyncio.Event()
        self.transport = config.transport
        self._by_market_id: Dict[int, str] = {}
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._ws_failures = 0
        self._connected = False

    # ------------------------------------------------------------------ public

    @property
    def connected(self) -> bool:
        return self._connected

    def book(self, symbol: str) -> OrderBook:
        return self.books[symbol.upper()]

    def is_fresh(self, symbol: str) -> bool:
        book = self.books.get(symbol.upper())
        return bool(book and book.is_fresh(self.config.staleness_ms))

    async def start(self, symbols: Sequence[str]) -> None:
        for symbol in symbols:
            symbol = symbol.upper()
            if symbol not in self.specs:
                raise KeyError(f"{self.venue_key}: market {symbol} not listed on this venue")
            self.books[symbol] = OrderBook(self.venue_key, symbol)
            self._by_market_id[self.specs[symbol].market_id] = symbol
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name=f"feed-{self.venue_key}")

    async def close(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self._connected = False

    async def wait_for_update(self, timeout: float) -> bool:
        """Block until any book changes, or ``timeout`` seconds elapse."""
        self.updated.clear()
        try:
            await asyncio.wait_for(self.updated.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def wait_ready(self, timeout: float = 15.0) -> bool:
        """Wait until every subscribed book has a usable snapshot."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.books and all(book.ready for book in self.books.values()):
                return True
            await self.wait_for_update(0.25)
        return False

    # ----------------------------------------------------------------- private

    async def _run(self) -> None:
        while not self._stop.is_set():
            if self.transport == "rest":
                await self._run_rest()
                continue
            try:
                await self._run_ws()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._connected = False
                self._ws_failures += 1
                for book in self.books.values():
                    book.clear()
                if self._ws_failures >= WS_FAILURES_BEFORE_FALLBACK:
                    log.error(
                        "%s: websocket failed %d times (%s); falling back to REST polling",
                        self.venue_key,
                        self._ws_failures,
                        exc,
                    )
                    self.transport = "rest"
                    continue
                backoff = min(
                    self.config.reconnect_backoff_max_ms,
                    self.config.reconnect_backoff_ms * (2 ** (self._ws_failures - 1)),
                ) / 1_000
                log.warning("%s: websocket error (%s); reconnecting in %.1fs", self.venue_key, exc, backoff)
                await asyncio.sleep(backoff)

    async def _run_ws(self) -> None:
        import websockets  # imported lazily so REST-only use needs no ws stack

        async with websockets.connect(self.ws_url, ping_interval=20, close_timeout=5) as ws:
            self._connected = True
            self._ws_failures = 0
            log.info("%s: websocket connected to %s", self.venue_key, self.ws_url)
            await self._subscribe(ws)
            async for raw in ws:
                if self._stop.is_set():
                    break
                try:
                    message = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                await self._handle(ws, message)
        self._connected = False

    async def _subscribe(self, ws: Any) -> None:
        for symbol in self.books:
            market_id = self.specs[symbol].market_id
            await ws.send(json.dumps({"type": "subscribe", "channel": f"order_book/{market_id}"}))

    async def _handle(self, ws: Any, message: Dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "connected":
            await self._subscribe(ws)
            return
        if kind == "ping":
            await ws.send(json.dumps({"type": "pong"}))
            return
        if kind not in ("subscribed/order_book", "update/order_book"):
            return

        symbol = self._symbol_for_channel(message.get("channel"))
        if symbol is None:
            return
        book = self.books.get(symbol)
        payload = message.get("order_book") or {}
        if book is None or not payload:
            return

        bids, asks = _levels(payload.get("bids")), _levels(payload.get("asks"))
        if kind == "subscribed/order_book":
            book.apply_snapshot(bids, asks)
        else:
            book.apply_delta(bids, asks)
        self.updated.set()

    def _symbol_for_channel(self, channel: Optional[str]) -> Optional[str]:
        if not channel:
            return None
        # Channels arrive as "order_book:<market_id>" (and "order_book/<id>" on
        # some builds); accept both.
        tail = channel.replace("/", ":").split(":")[-1]
        try:
            return self._by_market_id.get(int(tail))
        except ValueError:
            return None

    async def _run_rest(self) -> None:
        base_interval = max(0.05, self.config.rest_poll_ms / 1_000)
        interval = base_interval
        limit = max(50, self.config.depth_levels * 5)
        self._connected = True
        while not self._stop.is_set() and self.transport == "rest":
            started = time.monotonic()
            throttled = False
            failures: List[str] = []
            for symbol, book in self.books.items():
                try:
                    bids, asks = await self.rest.order_book_levels(
                        self.specs[symbol].market_id, limit=limit
                    )
                except LighterRateLimited:
                    throttled = True
                    break
                except Exception as exc:
                    failures.append(f"{symbol}: {exc}")
                    continue
                book.apply_snapshot(bids, asks)

            # A whole cycle failing means the venue is refusing us, not that one
            # market is odd: rate limiting, a bot challenge, an outage. Spinning
            # at full rate against any of those only makes it worse.
            everything_failed = bool(self.books) and len(failures) == len(self.books)
            if throttled or everything_failed:
                interval = min(REST_MAX_INTERVAL_SECONDS, max(interval, base_interval) * 2)
                log.warning(
                    "%s: %s, slowing book polling to %.1fs",
                    self.venue_key,
                    "rate limited" if throttled else f"all {len(failures)} polls failed",
                    interval,
                )
                if everything_failed and failures:
                    log.warning("%s: first failure was %s", self.venue_key, failures[0])
            else:
                if failures:
                    # One line per cycle rather than one per market.
                    log.warning(
                        "%s: %d/%d book polls failed (%s)",
                        self.venue_key,
                        len(failures),
                        len(self.books),
                        "; ".join(failures[:3]),
                    )
                if interval > base_interval:
                    interval = max(base_interval, interval * REST_INTERVAL_DECAY)
                self.updated.set()

            elapsed = time.monotonic() - started
            await asyncio.sleep(max(0.0, interval - elapsed))
        self._connected = False


class StaticFeed:
    """A feed backed by hand-made books. Used by tests and the simulator."""

    def __init__(self, venue_key: str, specs: Dict[str, MarketSpec]) -> None:
        self.venue_key = venue_key
        self.specs = specs
        self.books: Dict[str, OrderBook] = {}
        self.updated = asyncio.Event()
        self.transport = "static"
        self._connected = True

    @property
    def connected(self) -> bool:
        return self._connected

    def book(self, symbol: str) -> OrderBook:
        return self.books[symbol.upper()]

    def is_fresh(self, symbol: str) -> bool:
        book = self.books.get(symbol.upper())
        return bool(book and book.ready)

    async def start(self, symbols: Sequence[str]) -> None:
        for symbol in symbols:
            self.books[symbol.upper()] = OrderBook(self.venue_key, symbol.upper())

    async def close(self) -> None:
        self._connected = False

    async def wait_for_update(self, timeout: float) -> bool:
        self.updated.clear()
        try:
            await asyncio.wait_for(self.updated.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def wait_ready(self, timeout: float = 1.0) -> bool:
        return all(book.ready for book in self.books.values())

    def set_book(
        self,
        symbol: str,
        bids: Sequence[Tuple[Decimal, Decimal]],
        asks: Sequence[Tuple[Decimal, Decimal]],
    ) -> None:
        book = self.books.setdefault(symbol.upper(), OrderBook(self.venue_key, symbol.upper()))
        book.apply_snapshot(
            [(Decimal(str(p)), Decimal(str(s))) for p, s in bids if Decimal(str(s)) > ZERO],
            [(Decimal(str(p)), Decimal(str(s))) for p, s in asks if Decimal(str(s)) > ZERO],
        )
        self.updated.set()
