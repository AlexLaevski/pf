from decimal import Decimal

from spreadbot.config import RiskConfig
from spreadbot.models import PairPosition
from spreadbot.risk import RiskManager

D = Decimal


def manager(**overrides) -> RiskManager:
    base = dict(
        max_open_notional_usd=D(1_000),
        max_unhedged_notional_usd=D(100),
        unhedged_timeout_ms=1_000,
        max_daily_loss_usd=D(50),
        max_consecutive_errors=3,
    )
    base.update(overrides)
    return RiskManager(RiskConfig(**base))


def pair(long=D(0), short=D(0)) -> PairPosition:
    return PairPosition("BTC", "rh", "core", long_size=long, short_size=short)


def test_allows_a_clip_inside_the_limits():
    ok, reason = manager().can_open(
        pair(), mark=D(100), clip_base=D(1), max_position_base=D(5), total_open_notional=D(0)
    )
    assert ok, reason


def test_blocks_while_unhedged():
    ok, reason = manager().can_open(
        pair(long=D(1)), mark=D(100), clip_base=D(1), max_position_base=D(5), total_open_notional=D(0)
    )
    assert not ok and "unhedged" in reason


def test_blocks_over_max_position():
    ok, reason = manager().can_open(
        pair(long=D(5), short=D(5)),
        mark=D(100),
        clip_base=D(1),
        max_position_base=D(5),
        total_open_notional=D(0),
    )
    assert not ok and "max_position_base" in reason


def test_blocks_over_open_notional():
    ok, reason = manager().can_open(
        pair(), mark=D(100), clip_base=D(1), max_position_base=D(50), total_open_notional=D(950)
    )
    assert not ok and "max_open_notional_usd" in reason


def test_daily_loss_halts():
    risk = manager()
    risk.book_pnl(D(-60))
    assert risk.halted
    ok, reason = risk.can_open(
        pair(), mark=D(100), clip_base=D(1), max_position_base=D(5), total_open_notional=D(0)
    )
    assert not ok and "halted" in reason


def test_consecutive_errors_halt():
    risk = manager()
    for _ in range(3):
        risk.note_error("boom")
    assert risk.halted


def test_note_ok_resets_the_error_counter():
    risk = manager()
    risk.note_error("boom")
    risk.note_ok()
    risk.note_error("boom")
    assert not risk.halted


def test_unhedged_breach_on_notional():
    risk = manager()
    assert risk.unhedged_breach(pair(long=D(2)), D(100), now=1_000.0) is not None


def test_unhedged_breach_on_timeout():
    risk = manager()
    small = pair(long=D("0.1"))          # 10 USD, under the notional cap
    assert risk.unhedged_breach(small, D(100), now=1_000.0) is None
    assert risk.unhedged_breach(small, D(100), now=1_000.5) is None
    assert "over timeout" in risk.unhedged_breach(small, D(100), now=1_002.0)


def test_clearing_unhedged_resets_the_clock():
    risk = manager()
    small = pair(long=D("0.1"))
    risk.unhedged_breach(small, D(100), now=1_000.0)
    risk.clear_unhedged("BTC")
    assert risk.unhedged_breach(small, D(100), now=1_002.0) is None


def test_kill_switch_file(tmp_path):
    switch = tmp_path / "HALT"
    risk = manager(kill_switch_file=str(switch))
    assert not risk.halted
    switch.write_text("stop")
    assert risk.halted
