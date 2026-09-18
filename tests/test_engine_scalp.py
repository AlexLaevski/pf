"""Delayed-hedge (scalp) mode: fill -> wait for the bounce -> hedge if it never comes."""

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
SWEPT = ([("99.4", "5")], [("99.5", "5")])


def build(**hedge_overrides) -> Engine:
    hedge = {
        "mode": "delayed",
        "base_seconds": 30,
        "min_seconds": 10,
        "max_seconds": 30,
        "panic_bps": 50,
        "maker_first": False,
    }
    hedge.update(hedge_overrides)
    cfg = parse_config(
        raw_config(
            hedge=hedge,
            bounce={"target": "hole_top", "min_target_bps": 3},
            edge={"min_entry_bps": 3, "extra_buffer_bps": 0, "exit_bps": "0.5"},
        )
    )
    maker_specs = {"BTC": spec("BTC")}
    hedge_specs = {"BTC": spec("BTC")}
    maker_feed = StaticFeed("rh", maker_specs)
    hedge_feed = StaticFeed("core", hedge_specs)
    maker_feed.set_book("BTC", HOLEY_BIDS, MAKER_ASKS)
    hedge_feed.set_book("BTC", [("100.0", "10")], [("100.05", "10")])
    return Engine(
        cfg,
        maker_feed=maker_feed,
        hedge_feed=hedge_feed,
        maker_exec=PaperExecution("rh", Side.BUY, maker_specs, maker_feed),
        hedge_exec=PaperExecution("core", Side.SELL, hedge_specs, hedge_feed),
        maker_specs=maker_specs,
        hedge_specs=hedge_specs,
    )


async def fill_the_quote(engine: Engine) -> None:
    await engine._tick()
    engine.maker_feed.set_book("BTC", *SWEPT)
    await engine._tick()


async def test_entry_is_gated_on_the_bounce_not_the_cross_venue_spread():
    engine = build()
    await engine._tick()
    state = engine.states["BTC"]
    assert state.quote_order is not None
    assert state.quote_order.price == D("99.6")
    # Target is the near edge of the hole we were filled through: 99.8.
    assert state.planned_bounce_target == D("99.8")


async def test_fill_stays_naked_and_rests_the_take_profit():
    engine = build()
    await fill_the_quote(engine)
    state = engine.states["BTC"]
    assert state.pair.long_size == D("0.01")
    assert state.pair.short_size == D(0), "delayed mode must NOT hedge on the fill"
    assert state.naked_since is not None
    assert state.bounce_order is not None
    assert state.bounce_order.price == D("99.8")
    assert engine.stats.hedges == 0


async def test_bounce_fills_and_the_hedge_is_never_needed():
    engine = build()
    await fill_the_quote(engine)
    # Price recovers through our take-profit.
    engine.maker_feed.set_book("BTC", [("99.85", "5")], [("99.9", "5")])
    await engine._tick()
    state = engine.states["BTC"]
    assert state.pair.is_flat
    assert engine.stats.bounces_won == 1
    assert engine.stats.hedges == 0
    assert engine.risk.state.realized_pnl > 0


async def test_window_expiry_triggers_the_hedge():
    engine = build()
    await fill_the_quote(engine)
    state = engine.states["BTC"]
    assert state.pair.short_size == D(0)

    # Pretend the window elapsed.
    state.naked_since -= state.naked_window + 1
    await engine._tick()
    assert state.pair.short_size == D("0.01")
    assert state.pair.unhedged == D(0)
    assert engine.stats.hedges == 1
    assert engine.stats.panic_hedges == 0
    assert state.bounce_order is None


async def test_panic_hedge_fires_before_the_window_expires():
    engine = build()
    await fill_the_quote(engine)
    state = engine.states["BTC"]
    entry = state.pair.long_entry

    # Hedge venue collapses 1% below our entry; panic_bps is 50.
    crashed = entry * D("0.99")
    engine.hedge_feed.set_book("BTC", [(str(crashed), "10")], [(str(crashed * D("1.001")), "10")])
    await engine._tick()

    assert engine.stats.panic_hedges == 1
    assert state.pair.short_size == D("0.01")
    assert state.pair.unhedged == D(0)
    # The timer had barely started - the price, not the clock, ended the window.
    assert state.naked_since is None


async def test_small_adverse_move_does_not_panic():
    engine = build()
    await fill_the_quote(engine)
    state = engine.states["BTC"]
    entry = state.pair.long_entry

    nudge = entry * D("0.999")     # 10bps against us, panic is 50
    engine.hedge_feed.set_book("BTC", [(str(nudge), "10")], [(str(nudge * D("1.001")), "10")])
    await engine._tick()

    assert engine.stats.panic_hedges == 0
    assert state.pair.short_size == D(0)
    assert state.naked_since is not None


async def test_unhedged_notional_cap_overrides_the_window():
    engine = build()
    engine.risk.cfg = type(engine.risk.cfg)(
        **{**engine.risk.cfg.__dict__, "max_unhedged_notional_usd": D("0.01")}
    )
    await fill_the_quote(engine)
    state = engine.states["BTC"]
    assert state.pair.short_size == D("0.01"), "notional ceiling must beat the bounce window"


async def test_volatile_market_gets_a_shorter_window():
    engine = build()
    state = engine.states["BTC"]
    # Feed a jumpy mid series before the fill.
    for i in range(60):
        state.vol.update(1_000.0 + i, D("100") * (D("1.02") ** (i % 2)))
    await fill_the_quote(engine)
    assert state.naked_window == 10.0     # clamped to min_seconds


async def test_no_new_quote_while_naked():
    engine = build()
    await fill_the_quote(engine)
    # A fresh gap appears, but still below our take-profit so the bounce order
    # does not fill: the bot must sit on its hands rather than stack a second clip.
    engine.maker_feed.set_book(
        "BTC",
        [("99.45", "0.01"), ("99.44", "0.01"), ("99.43", "0.01"), ("99.2", "5"), ("99.1", "5")],
        [("99.5", "1"), ("99.6", "1")],
    )
    await engine._tick()
    state = engine.states["BTC"]
    assert state.quote_order is None
    assert state.naked_since is not None


async def test_immediate_mode_still_hedges_on_the_fill():
    engine = build(mode="immediate")
    await fill_the_quote(engine)
    state = engine.states["BTC"]
    assert state.pair.short_size == D("0.01")
    assert state.naked_since is None
    assert engine.stats.hedges == 1
