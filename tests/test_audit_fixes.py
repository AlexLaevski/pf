"""Regressions for the findings of the external code audit.

Each of these encodes a way the bot could lose money that the code allowed
before. They are grouped here so it is obvious what must never come back.
"""

from decimal import Decimal


from spreadbot.config import GapConfig, RiskConfig, parse_config
from spreadbot.engine import Engine
from spreadbot.gaps import find_wall_candidates
from spreadbot.hedger import Hedger
from spreadbot.models import Side
from spreadbot.risk import RiskManager
from spreadbot.venues.feed import StaticFeed
from spreadbot.venues.paper import PaperExecution

from .conftest import book, raw_config, spec

D = Decimal

HOLEY_BIDS = [("100.0", "0.01"), ("99.9", "0.01"), ("99.8", "0.01"), ("99.5", "5"), ("99.4", "5")]
MAKER_ASKS = [("100.1", "1"), ("100.2", "1")]
SWEPT = ([("99.4", "5")], [("99.5", "5")])


def build_engine(**overrides):
    cfg = parse_config(raw_config(**overrides))
    specs = {"BTC": spec("BTC")}
    maker_feed = StaticFeed("rh", specs, staleness_ms=3_000)
    hedge_feed = StaticFeed("core", specs, staleness_ms=3_000)
    maker_feed.set_book("BTC", HOLEY_BIDS, MAKER_ASKS)
    hedge_feed.set_book("BTC", [("100.0", "10")], [("100.05", "10")])
    return Engine(
        cfg,
        maker_feed=maker_feed,
        hedge_feed=hedge_feed,
        maker_exec=PaperExecution("rh", Side.BUY, specs, maker_feed),
        hedge_exec=PaperExecution("core", Side.SELL, specs, hedge_feed),
        maker_specs=specs,
        hedge_specs=specs,
    )


# --- audit #2: a stale book must not stop a hedge --------------------------


async def test_naked_position_is_hedged_even_when_the_maker_book_goes_stale():
    engine = build_engine()
    await engine._tick()
    engine.maker_feed.set_book("BTC", *SWEPT)
    await engine._tick()
    state = engine.states["BTC"]
    assert state.pair.short_size == D("0.01")


async def test_stale_maker_book_does_not_strand_a_naked_position():
    engine = build_engine(hedge={"mode": "delayed", "min_seconds": 60, "max_seconds": 60})
    await engine._tick()
    engine.maker_feed.set_book("BTC", *SWEPT)
    await engine._tick()
    state = engine.states["BTC"]
    assert state.pair.unhedged > 0, "delayed mode should still be holding the naked long"

    # The maker feed lags AND the price runs away on the hedge venue. The old
    # code returned at the freshness gate before ever reaching the panic check,
    # leaving the naked long with no hedge and no stop. The lag and the move
    # are usually the same event, which is what made this dangerous.
    engine.maker_feed.books["BTC"].updated_at -= 3_600
    crashed = state.pair.long_entry * D("0.98")
    engine.hedge_feed.set_book(
        "BTC", [(str(crashed), "10")], [(str(crashed * D("1.001")), "10")]
    )
    await engine._tick()
    assert state.pair.unhedged == 0, "panic hedge must still fire on a stale maker book"
    assert state.pair.short_size == D("0.01")
    assert engine.stats.panic_hedges == 1


async def test_stale_hedge_book_with_naked_exposure_escalates_instead_of_idling():
    engine = build_engine(hedge={"mode": "delayed", "min_seconds": 60, "max_seconds": 60})
    await engine._tick()
    engine.maker_feed.set_book("BTC", *SWEPT)
    await engine._tick()
    assert engine.states["BTC"].pair.unhedged > 0

    engine.hedge_feed.books["BTC"].updated_at -= 3_600
    before = engine.risk.state.consecutive_errors
    await engine._tick()
    # It cannot price the hedge, but it must not quietly do nothing either.
    assert engine.risk.state.consecutive_errors > before
    assert "stale" in engine.states["BTC"].last_reason


# --- audit #4: the maker-first cancel race ---------------------------------


async def test_maker_hedge_filled_during_cancel_is_not_taken_twice():
    specs = {"BTC": spec("BTC")}
    feed = StaticFeed("core", specs)
    feed.set_book("BTC", [("100.0", "5")], [("100.1", "5")])
    venue = PaperExecution("core", Side.SELL, specs, feed)
    hedger = Hedger(venue, max_slippage_bps=D(50), retry_delay=0)

    original_cancel = venue.cancel

    async def cancel_but_it_already_filled(order):
        # The book trades through our posted ask exactly as the cancel lands.
        feed.set_book("BTC", [("100.2", "5")], [("100.3", "5")])
        venue.poll()
        return await original_cancel(order)

    venue.cancel = cancel_but_it_already_filled
    result = await hedger.execute_maker_first(
        "BTC", Side.SELL, D("0.5"), feed.book("BTC"), timeout_ms=30
    )
    assert result.filled == D("0.5"), "must not hedge twice"
    assert (await venue.positions())["BTC"] == D("-0.5")


# --- audit #5: false walls and touch-improving quotes ----------------------


def test_levels_far_apart_do_not_add_up_to_a_wall():
    cfg = GapConfig(
        wall_notional_usd=D(900),
        wall_multiple=D(2),
        min_gap_bps=D(5),
        min_gap_ticks=2,
        join_offset_ticks=1,
        max_distance_from_mid_bps=D(500),
        min_wall_levels=2,
        max_wall_span_bps=D(10),
    )
    dust = [("100.0", "0.01"), ("99.9", "0.01"), ("99.8", "0.01"), ("99.7", "0.01")]
    # Two big levels 80bps apart: separate levels, not one block of liquidity.
    spread_out = dust + [("99.5", "8"), ("98.7", "8")]
    ob = book(spread_out, MAKER_ASKS)
    assert find_wall_candidates(ob, Side.BUY, spec(), cfg) == []

    close_together = dust + [("99.5", "8"), ("99.4", "8")]
    ob = book(close_together, MAKER_ASKS)
    assert find_wall_candidates(ob, Side.BUY, spec(), cfg)


def test_touch_improving_candidates_are_skipped_by_default():
    cfg = GapConfig(
        wall_notional_usd=D(300),
        wall_multiple=D(1),
        min_gap_bps=D(1),
        min_gap_ticks=1,
        join_offset_ticks=1,
        max_distance_from_mid_bps=D(500),
    )
    # The wall IS the best bid: quoting a tick above it is just becoming the
    # touch, with nothing underneath to stop a sweep.
    ob = book([("99.5", "5"), ("99.4", "5")], [("100.1", "1")])
    assert all(c.level_index > 0 for c in find_wall_candidates(ob, Side.BUY, spec(), cfg))

    allowed = GapConfig(**{**cfg.__dict__, "allow_touch_improving": True})
    assert any(c.level_index == 0 for c in find_wall_candidates(ob, Side.BUY, spec(), allowed))


# --- audit #7: one slow market must not block the others -------------------


async def test_a_slow_hedge_does_not_delay_other_markets():
    """Markets tick concurrently, so a hedge in flight on one does not hold
    the others' hedges behind it."""
    import asyncio

    cfg = parse_config(
        raw_config(
            markets=[
                {"symbol": s, "order_base": "0.01", "max_position_base": "0.05"}
                for s in ("AAA", "BBB")
            ]
        )
    )
    from spreadbot.models import MarketSpec

    specs = {s: MarketSpec(s, i, 1, 5, D("0.00007"), D(0)) for i, s in enumerate(("AAA", "BBB"))}
    maker_feed = StaticFeed("rh", specs)
    hedge_feed = StaticFeed("core", specs)
    for symbol in specs:
        maker_feed.set_book(symbol, HOLEY_BIDS, MAKER_ASKS)
        hedge_feed.set_book(symbol, [("100.0", "10")], [("100.05", "10")])

    engine = Engine(
        cfg,
        maker_feed=maker_feed,
        hedge_feed=hedge_feed,
        maker_exec=PaperExecution("rh", Side.BUY, specs, maker_feed),
        hedge_exec=PaperExecution("core", Side.SELL, specs, hedge_feed),
        maker_specs=specs,
        hedge_specs=specs,
    )
    await engine._tick()

    started: list[str] = []

    async def slow_tick(symbol, state, now, _real=engine._tick_market):
        started.append(symbol)
        if symbol == "AAA":
            await asyncio.sleep(0.2)     # a hedge grinding through its retries
        await _real(symbol, state, now)

    engine._tick_market = slow_tick
    for symbol in specs:
        maker_feed.set_book(symbol, *SWEPT)

    await asyncio.wait_for(engine._tick(), timeout=1.0)
    # Both markets started their tick; sequentially BBB would have waited for AAA.
    assert set(started) == {"AAA", "BBB"}
    assert engine.states["BBB"].pair.short_size == D("0.01")


async def test_a_market_is_not_ticked_twice_at_once():
    import asyncio

    engine = build_engine()
    entered = 0
    release = asyncio.Event()

    async def blocking_tick(symbol, state, now):
        nonlocal entered
        entered += 1
        await release.wait()

    engine._tick_market = blocking_tick
    first = asyncio.create_task(engine._tick())
    await asyncio.sleep(0)
    second = asyncio.create_task(engine._tick())
    await asyncio.sleep(0.05)
    release.set()
    await asyncio.gather(first, second)
    # The second tick found the market busy and skipped it rather than sending
    # a duplicate hedge on top of the one still in flight.
    assert entered == 1


# --- audit #6: the daily loss halt must expire with its day ----------------


def risk_manager(**overrides) -> RiskManager:
    base = dict(
        max_open_notional_usd=D(1_000),
        max_unhedged_notional_usd=D(100),
        unhedged_timeout_ms=1_000,
        max_daily_loss_usd=D(50),
        max_consecutive_errors=3,
    )
    base.update(overrides)
    return RiskManager(RiskConfig(**base))


def test_daily_loss_halt_clears_on_the_day_roll():
    risk = risk_manager()
    risk.book_pnl(D(-60))
    assert risk.halted
    assert risk.state.halt_kind == "daily_loss"

    # Nothing books P&L while halted, so the roll has to happen on its own.
    risk.state.day = "1999-01-01"
    assert not risk.halted
    assert risk.state.realized_pnl == 0


def test_other_halts_survive_the_day_roll():
    risk = risk_manager()
    risk.halt("position on the forbidden side", kind="forbidden_side")
    risk.state.day = "1999-01-01"
    assert risk.halted, "a halt about the world does not expire with the calendar"
    assert "forbidden" in risk.state.halt_reason


def test_error_halt_survives_the_day_roll():
    risk = risk_manager()
    for _ in range(3):
        risk.note_error("hedge failed")
    assert risk.halted
    risk.state.day = "1999-01-01"
    assert risk.halted


def test_kill_switch_halt_is_not_sticky_once_the_file_is_gone(tmp_path):
    switch = tmp_path / "HALT"
    switch.write_text("stop")
    risk = risk_manager(kill_switch_file=str(switch))
    assert risk.halted
    switch.unlink()
    risk.resume()
    assert not risk.halted
