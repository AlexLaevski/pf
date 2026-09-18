import asyncio
from decimal import Decimal

import pytest

from spreadbot.hedger import Hedger
from spreadbot.models import OrderStatus, OrderType, Side
from spreadbot.venues.base import OrderRequest, SideNotAllowed
from spreadbot.venues.feed import StaticFeed
from spreadbot.venues.paper import PaperExecution

from .conftest import spec

D = Decimal


def make_venue(side: Side = Side.BUY, symbol: str = "BTC"):
    specs = {symbol: spec(symbol)}
    feed = StaticFeed("v", specs)
    feed.books = {}
    feed.set_book(symbol, [("100.0", "5")], [("100.1", "5")])
    return PaperExecution("v", side, specs, feed), feed


async def test_long_only_venue_refuses_to_open_a_short():
    venue, _ = make_venue(Side.BUY)
    with pytest.raises(SideNotAllowed):
        await venue.place(
            OrderRequest("BTC", Side.SELL, D("0.01"), D("101"), OrderType.POST_ONLY)
        )


async def test_long_only_venue_allows_a_reduce_only_sell():
    venue, _ = make_venue(Side.BUY)
    order = await venue.place(
        OrderRequest("BTC", Side.SELL, D("0.01"), D("101"), OrderType.POST_ONLY, reduce_only=True)
    )
    assert order.status is OrderStatus.OPEN


async def test_short_only_venue_refuses_to_open_a_long():
    venue, _ = make_venue(Side.SELL)
    with pytest.raises(SideNotAllowed):
        await venue.place(OrderRequest("BTC", Side.BUY, D("0.01"), D("99"), OrderType.POST_ONLY))


async def test_post_only_that_would_cross_is_rejected():
    venue, _ = make_venue(Side.BUY)
    order = await venue.place(
        OrderRequest("BTC", Side.BUY, D("0.01"), D("100.5"), OrderType.POST_ONLY)
    )
    assert order.status is OrderStatus.REJECTED
    assert "post-only" in order.error


async def test_resting_order_fills_when_the_book_sweeps_through():
    venue, feed = make_venue(Side.BUY)
    await venue.place(
        OrderRequest("BTC", Side.BUY, D("0.01"), D("99.5"), OrderType.POST_ONLY)
    )
    assert venue.poll() == []
    feed.set_book("BTC", [("99.4", "5")], [("99.5", "5")])   # swept through our bid
    fills = venue.poll()
    assert len(fills) == 1
    assert fills[0].price == D("99.5")
    assert (await venue.positions())["BTC"] == D("0.01")


async def test_ioc_fills_at_book_vwap():
    venue, _ = make_venue(Side.SELL)
    order = await venue.place(
        OrderRequest("BTC", Side.SELL, D("1"), D("99"), OrderType.IOC, max_slippage_bps=D(100))
    )
    assert order.status is OrderStatus.FILLED
    assert order.avg_fill_price == D("100.0")


async def test_ioc_rejected_when_the_book_is_too_thin():
    venue, feed = make_venue(Side.SELL)
    feed.set_book("BTC", [("100.0", "0.001")], [("100.1", "0.001")])
    order = await venue.place(
        OrderRequest("BTC", Side.SELL, D("1"), D("99"), OrderType.IOC, max_slippage_bps=D(100))
    )
    assert order.status is OrderStatus.REJECTED
    assert "too thin" in order.error


async def test_size_below_venue_minimum_is_rejected():
    venue, _ = make_venue(Side.BUY)
    with pytest.raises(ValueError, match="min_base_amount"):
        await venue.place(
            OrderRequest("BTC", Side.BUY, D("0.000001"), D("99"), OrderType.POST_ONLY)
        )


async def test_hedger_fills_in_one_attempt():
    venue, feed = make_venue(Side.SELL)
    hedger = Hedger(venue, max_slippage_bps=D(20), retry_delay=0)
    result = await hedger.execute("BTC", Side.SELL, D("0.5"), feed.book("BTC"))
    assert result.ok
    assert result.filled == D("0.5")
    assert result.attempts == 1


async def test_hedger_reports_failure_when_the_book_cannot_absorb_it():
    venue, feed = make_venue(Side.SELL)
    feed.set_book("BTC", [("100.0", "0.001")], [("100.1", "0.001")])
    hedger = Hedger(venue, max_slippage_bps=D(20), attempts=2, retry_delay=0)
    result = await hedger.execute("BTC", Side.SELL, D("1"), feed.book("BTC"))
    assert not result.ok
    assert result.error


async def test_maker_first_hedge_falls_back_to_taking():
    venue, feed = make_venue(Side.SELL)
    hedger = Hedger(venue, max_slippage_bps=D(50), retry_delay=0)
    # Posting at the touch does not fill in the paper model, so the ladder has
    # to time out and take. 50ms keeps the test quick.
    result = await hedger.execute_maker_first(
        "BTC", Side.SELL, D("0.5"), feed.book("BTC"), timeout_ms=50
    )
    assert result.ok
    assert result.filled == D("0.5")
    assert result.price == D("100.0")      # took the bid, not the posted ask


async def test_maker_first_hedge_fills_passively_when_the_book_moves():
    venue, feed = make_venue(Side.SELL)
    hedger = Hedger(venue, max_slippage_bps=D(50), retry_delay=0)

    async def lift_the_book():
        await asyncio.sleep(0.02)
        feed.set_book("BTC", [("100.2", "5")], [("100.3", "5")])

    lifter = asyncio.create_task(lift_the_book())
    result = await hedger.execute_maker_first(
        "BTC", Side.SELL, D("0.5"), feed.book("BTC"), timeout_ms=400
    )
    await lifter
    assert result.ok
    assert result.price == D("100.1")      # our posted ask, saving the spread
    assert result.attempts == 1


async def test_maker_first_never_leaves_the_hedge_pending():
    venue, feed = make_venue(Side.SELL)
    hedger = Hedger(venue, max_slippage_bps=D(50), retry_delay=0)
    result = await hedger.execute_maker_first(
        "BTC", Side.SELL, D("0.5"), feed.book("BTC"), timeout_ms=30
    )
    assert result.filled == D("0.5")
    assert venue.resting("BTC") == {}, "the unfilled maker leg must be cancelled"


async def test_cancel_all_clears_resting_orders():
    venue, _ = make_venue(Side.BUY)
    await venue.place(OrderRequest("BTC", Side.BUY, D("0.01"), D("99.5"), OrderType.POST_ONLY))
    await venue.place(OrderRequest("BTC", Side.BUY, D("0.01"), D("99.4"), OrderType.POST_ONLY))
    assert await venue.cancel_all("BTC") == 2
    assert venue.poll() == []
