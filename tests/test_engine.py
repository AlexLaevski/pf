"""End-to-end loop on synthetic books: quote -> fill -> hedge -> unwind."""

from decimal import Decimal


from spreadbot.config import parse_config
from spreadbot.engine import Engine
from spreadbot.models import Side
from spreadbot.venues.feed import StaticFeed
from spreadbot.venues.paper import PaperExecution

from .conftest import raw_config, spec

D = Decimal

HOLEY_BIDS = [("100.0", "0.01"), ("99.9", "0.01"), ("99.8", "0.01"), ("99.5", "5"), ("99.4", "5")]
MAKER_ASKS = [("100.1", "1"), ("100.2", "1")]


def build(**overrides) -> Engine:
    cfg = parse_config(raw_config(**overrides))
    maker_specs = {"BTC": spec("BTC")}
    hedge_specs = {"BTC": spec("BTC")}

    maker_feed = StaticFeed("rh", maker_specs)
    hedge_feed = StaticFeed("core", hedge_specs)
    maker_feed.set_book("BTC", HOLEY_BIDS, MAKER_ASKS)
    hedge_feed.set_book("BTC", [("100.0", "10")], [("100.05", "10")])

    maker_exec = PaperExecution("rh", Side.BUY, maker_specs, maker_feed)
    hedge_exec = PaperExecution("core", Side.SELL, hedge_specs, hedge_feed)

    return Engine(
        cfg,
        maker_feed=maker_feed,
        hedge_feed=hedge_feed,
        maker_exec=maker_exec,
        hedge_exec=hedge_exec,
        maker_specs=maker_specs,
        hedge_specs=hedge_specs,
    )


async def test_quotes_in_front_of_the_wall():
    engine = build()
    await engine._tick()
    state = engine.states["BTC"]
    assert state.quote_order is not None
    assert state.quote_order.price == D("99.6")
    assert state.quote_order.side is Side.BUY
    assert engine.stats.quotes_placed == 1


async def test_fill_is_hedged_on_the_other_venue_same_tick():
    engine = build()
    await engine._tick()

    # A sweep takes the book down through our resting bid.
    engine.maker_feed.set_book("BTC", [("99.4", "5")], [("99.5", "5")])
    await engine._tick()

    pair = engine.states["BTC"].pair
    assert pair.long_size == D("0.01")
    assert pair.short_size == D("0.01"), "the fill must be hedged immediately"
    assert pair.unhedged == D(0)
    assert pair.long_entry == D("99.6")
    assert pair.short_entry == D("100.0")
    assert pair.entry_spread_bps > D(35)
    assert engine.stats.hedges == 1


async def test_no_new_quote_while_a_position_is_open():
    engine = build()
    await engine._tick()
    engine.maker_feed.set_book("BTC", [("99.4", "5")], [("99.5", "5")])
    await engine._tick()

    engine.maker_feed.set_book("BTC", HOLEY_BIDS, MAKER_ASKS)
    await engine._tick()
    assert engine.states["BTC"].quote_order is None


async def test_unwinds_once_the_spread_converges():
    engine = build(unwind={"passive_exit": False, "max_hold_seconds": 10_000})
    await engine._tick()
    engine.maker_feed.set_book("BTC", [("99.4", "5")], [("99.5", "5")])
    await engine._tick()
    assert engine.states["BTC"].pair.matched == D("0.01")

    # The maker venue catches up: selling there now matches buying the hedge back.
    engine.maker_feed.set_book("BTC", [("100.0", "10")], [("100.05", "10")])
    engine.hedge_feed.set_book("BTC", [("99.95", "10")], [("100.0", "10")])
    await engine._tick()

    pair = engine.states["BTC"].pair
    assert pair.is_flat
    assert engine.stats.exits == 1
    assert engine.risk.state.realized_pnl > 0


async def test_passive_exit_fill_leaves_no_naked_short():
    engine = build(unwind={"passive_exit": True, "max_hold_seconds": 10_000})
    await engine._tick()
    engine.maker_feed.set_book("BTC", [("99.4", "5")], [("99.5", "5")])
    await engine._tick()

    # Spread converged: with passive_exit the bot rests a reduce-only sell.
    engine.maker_feed.set_book(
        "BTC",
        [("100.0", "10")],
        [("100.1", "0.01"), ("100.2", "0.01"), ("100.3", "0.01"), ("100.6", "5")],
    )
    engine.hedge_feed.set_book("BTC", [("99.95", "10")], [("100.0", "10")])
    await engine._tick()
    state = engine.states["BTC"]
    assert state.exit_order is not None and state.exit_order.price == D("100.5")

    # The passive sell gets lifted: the long is gone, the short must follow.
    engine.maker_feed.set_book("BTC", [("100.5", "5")], [("100.6", "5")])
    await engine._tick()
    assert state.pair.long_size == D(0)
    await engine._tick()
    assert state.pair.is_flat, "the short leg must not be left naked"


async def test_hedge_is_not_double_counted_on_later_ticks():
    engine = build()
    await engine._tick()
    engine.maker_feed.set_book("BTC", [("99.4", "5")], [("99.5", "5")])
    await engine._tick()
    short_after_hedge = engine.states["BTC"].pair.short_size
    await engine._tick()
    assert engine.states["BTC"].pair.short_size == short_after_hedge


async def test_stale_book_pulls_the_quote():
    engine = build()
    await engine._tick()
    assert engine.states["BTC"].quote_order is not None

    engine.maker_feed.books["BTC"].clear()
    await engine._tick()
    assert engine.states["BTC"].quote_order is None
    assert engine.states["BTC"].last_reason == "stale book"


async def test_halt_cancels_every_quote():
    engine = build()
    await engine._tick()
    assert engine.states["BTC"].quote_order is not None

    engine.risk.halt("test")
    await engine._tick()
    assert engine.states["BTC"].quote_order is None


async def test_resting_quotes_count_against_the_open_notional_cap():
    engine = build()
    await engine._tick()
    state = engine.states["BTC"]
    assert state.quote_order is not None

    resting = state.quote_order.remaining * state.quote_order.price
    assert engine._open_notional() == resting, "a resting quote is committed capital"

    # With the cap just under one clip, the quote must be pulled rather than
    # left out there as an unaccounted promise to buy.
    engine.risk.cfg = type(engine.risk.cfg)(
        **{**engine.risk.cfg.__dict__, "max_open_notional_usd": resting / 2}
    )
    await engine._tick()
    assert engine.states["BTC"].quote_order is None
    assert "max_open_notional_usd" in engine.states["BTC"].last_reason


async def test_position_reconciliation_adopts_the_venue():
    engine = build()
    # Someone (or a missed fill) left a long on the maker venue.
    engine.maker_exec.positions_book.setdefault("BTC", _position("rh")).size = D("0.02")
    await engine._reconcile_positions()
    pair = engine.states["BTC"].pair
    assert pair.long_size == D("0.02")
    assert pair.unhedged == D("0.02"), "drift must show up as unhedged, so the next tick hedges it"


async def test_forbidden_side_position_halts():
    engine = build()
    engine.hedge_exec.positions_book.setdefault("BTC", _position("core")).size = D("0.02")
    await engine._reconcile_positions()
    assert engine.risk.halted
    assert "forbidden side" in engine.risk.state.halt_reason


def _position(venue: str):
    from spreadbot.models import Position

    return Position(venue, "BTC")
