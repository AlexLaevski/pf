from decimal import Decimal


from spreadbot.config import parse_config
from spreadbot.models import PairPosition
from spreadbot.strategy import SpreadStrategy

from .conftest import book, raw_config, spec

D = Decimal

# Maker venue (Robinhood domain): thin bids, a wall 45bps below mid.
MAKER_BIDS = [
    ("100.0", "0.01"),
    ("99.9", "0.01"),
    ("99.8", "0.01"),
    ("99.5", "5"),
    ("99.4", "5"),
]
MAKER_ASKS = [("100.1", "1"), ("100.2", "1")]


def strategy(**overrides):
    return SpreadStrategy(parse_config(raw_config(**overrides)))


def test_entry_edge_is_net_of_both_fees():
    strat = strategy()
    maker = spec(maker_fee_bps="1")
    hedge = spec(taker_fee_bps="2")
    # Buy at 100, sell at 100.5 => 50bps gross, minus 1bps maker and ~2bps taker.
    edge = strat.entry_edge_bps(D(100), D("100.5"), maker, hedge)
    assert D(46) < edge < D(48)


def test_plan_entry_quotes_in_front_of_the_wall_when_the_hedge_pays():
    strat = strategy()
    maker_book = book(MAKER_BIDS, MAKER_ASKS, venue="rh")
    hedge_book = book([("100.0", "10")], [("100.05", "10")], venue="core")
    decision = strat.plan_entry(
        "BTC", maker_book, hedge_book, spec(), spec(), capacity_base=D("0.05")
    )
    assert decision.ok, decision.reason
    plan = decision.plan
    assert plan.quote.price == D("99.6")
    assert plan.hedge_price == D("100.0")      # hits the hedge venue's bid
    assert plan.edge_bps > D(3)


def test_plan_entry_refuses_when_the_hedge_is_not_rich_enough():
    strat = strategy()
    maker_book = book(MAKER_BIDS, MAKER_ASKS, venue="rh")
    # Hedge venue bid is below our quote: selling there would lose money.
    hedge_book = book([("99.55", "10")], [("99.6", "10")], venue="core")
    decision = strat.plan_entry(
        "BTC", maker_book, hedge_book, spec(), spec(), capacity_base=D("0.05")
    )
    assert not decision.ok
    assert "below threshold" in decision.reason


def test_plan_entry_refuses_a_thin_hedge_book():
    strat = strategy()
    maker_book = book(MAKER_BIDS, MAKER_ASKS, venue="rh")
    hedge_book = book([("100.0", "0.0001")], [("100.05", "1")], venue="core")
    decision = strat.plan_entry(
        "BTC", maker_book, hedge_book, spec(), spec(), capacity_base=D("0.05")
    )
    assert not decision.ok
    assert "too thin" in decision.reason


def test_plan_entry_needs_capacity():
    strat = strategy()
    maker_book = book(MAKER_BIDS, MAKER_ASKS, venue="rh")
    hedge_book = book([("100.0", "10")], [("100.05", "10")], venue="core")
    decision = strat.plan_entry(
        "BTC", maker_book, hedge_book, spec(), spec(), capacity_base=D(0)
    )
    assert not decision.ok
    assert "capacity" in decision.reason


def test_plan_entry_clips_to_remaining_capacity():
    strat = strategy()
    maker_book = book(MAKER_BIDS, MAKER_ASKS, venue="rh")
    hedge_book = book([("100.0", "10")], [("100.05", "10")], venue="core")
    decision = strat.plan_entry(
        "BTC", maker_book, hedge_book, spec(), spec(), capacity_base=D("0.004")
    )
    assert decision.ok
    assert decision.plan.size == D("0.004")


def test_exit_fires_once_the_reverse_spread_closes():
    strat = strategy()
    pair = PairPosition("BTC", "rh", "core", long_size=D(1), short_size=D(1))
    pair.long_entry = D("99.6")
    pair.short_entry = D("100.0")
    pair.opened_at = 1_000.0
    # Maker venue caught up: we can sell at 100 and buy the short back at 100.
    maker_book = book([("100.0", "10")], [("100.05", "10")], venue="rh")
    hedge_book = book([("99.95", "10")], [("100.0", "10")], venue="core")
    plan = strat.evaluate_exit(pair, maker_book, hedge_book, spec(), spec(), now=1_001.0)
    assert plan is not None
    assert "converged" in plan.reason
    assert plan.pnl_bps > 0


def test_exit_holds_while_the_spread_is_still_open():
    strat = strategy()
    pair = PairPosition("BTC", "rh", "core", long_size=D(1), short_size=D(1))
    pair.long_entry = D("99.6")
    pair.short_entry = D("100.0")
    pair.opened_at = 1_000.0
    maker_book = book([("99.6", "10")], [("99.65", "10")], venue="rh")
    hedge_book = book([("100.0", "10")], [("100.05", "10")], venue="core")
    assert strat.evaluate_exit(pair, maker_book, hedge_book, spec(), spec(), now=1_001.0) is None


def test_stop_loss_forces_an_aggressive_exit():
    strat = strategy(unwind={"stop_loss_bps": 10, "max_hold_seconds": 10_000})
    pair = PairPosition("BTC", "rh", "core", long_size=D(1), short_size=D(1))
    pair.long_entry = D("100.0")
    pair.short_entry = D("100.05")
    pair.opened_at = 1_000.0
    # Maker venue collapsed relative to the hedge venue.
    maker_book = book([("98.0", "10")], [("98.05", "10")], venue="rh")
    hedge_book = book([("100.0", "10")], [("100.05", "10")], venue="core")
    plan = strat.evaluate_exit(pair, maker_book, hedge_book, spec(), spec(), now=1_001.0)
    assert plan is not None and plan.aggressive
    assert "stop loss" in plan.reason


def test_timeout_forces_an_aggressive_exit():
    strat = strategy(unwind={"max_hold_seconds": 1, "stop_loss_bps": 10_000})
    pair = PairPosition("BTC", "rh", "core", long_size=D(1), short_size=D(1))
    pair.long_entry = D("99.6")
    pair.short_entry = D("100.0")
    pair.opened_at = 1_000.0
    maker_book = book([("99.6", "10")], [("99.65", "10")], venue="rh")
    hedge_book = book([("100.0", "10")], [("100.05", "10")], venue="core")
    plan = strat.evaluate_exit(pair, maker_book, hedge_book, spec(), spec(), now=2_000.0)
    assert plan is not None and plan.aggressive
    assert "max hold" in plan.reason


def test_passive_exit_price_sits_in_front_of_an_ask_wall():
    strat = strategy()
    pair = PairPosition("BTC", "rh", "core", long_size=D(1), short_size=D(1))
    pair.long_entry = D("99.6")
    maker_book = book(
        [("100.0", "1")],
        [("100.1", "0.01"), ("100.2", "0.01"), ("100.3", "0.01"), ("100.6", "5")],
        venue="rh",
    )
    assert strat.passive_exit_price(pair, maker_book, spec()) == D("100.5")


def _wall():
    from spreadbot.gaps import find_wall_candidates
    from spreadbot.models import Side

    ob = book(MAKER_BIDS, MAKER_ASKS, venue="rh")
    return find_wall_candidates(ob, Side.BUY, spec(), parse_config(raw_config()).gap)[0]


def test_bounce_target_aims_at_the_top_of_the_hole():
    strat = strategy(hedge={"mode": "delayed"}, bounce={"target": "hole_top"})
    candidate = _wall()
    assert candidate.near_price == D("99.8")
    assert strat.bounce_target_price(candidate.price, candidate, spec()) == D("99.8")


def test_bounce_target_respects_the_minimum_markup():
    # A hole top only a hair above the entry gets pushed up to the floor.
    strat = strategy(hedge={"mode": "delayed"}, bounce={"target": "hole_top", "min_target_bps": 100})
    candidate = _wall()
    target = strat.bounce_target_price(candidate.price, candidate, spec())
    assert target > candidate.near_price
    assert target >= candidate.price * D("1.01")


def test_bounce_target_fixed_bps_mode():
    strat = strategy(hedge={"mode": "delayed"}, bounce={"target": "fixed_bps", "target_bps": 50})
    candidate = _wall()
    target = strat.bounce_target_price(D(100), candidate, spec())
    assert target == D("100.5")


def test_bounce_edge_charges_the_maker_fee_twice():
    strat = strategy(hedge={"mode": "delayed"})
    plain = strat.bounce_edge_bps(D(100), D("100.5"), spec())
    with_fee = strat.bounce_edge_bps(D(100), D("100.5"), spec(maker_fee_bps="1"))
    assert plain == D(50)
    assert D(47) < with_fee < D(49)


def test_delayed_mode_enters_on_the_bounce_not_the_hedge_price():
    strat = strategy(hedge={"mode": "delayed"}, bounce={"target": "hole_top"})
    maker_book = book(MAKER_BIDS, MAKER_ASKS, venue="rh")
    # Hedge venue is BELOW our quote: in immediate mode this would be refused.
    hedge_book = book([("99.55", "10")], [("99.60", "10")], venue="core")
    decision = strat.plan_entry(
        "BTC", maker_book, hedge_book, spec(), spec(), capacity_base=D("0.05")
    )
    assert decision.ok, decision.reason
    assert decision.plan.bounce_target == D("99.8")
    assert decision.plan.edge_bps > D(19)     # 99.6 -> 99.8 is ~20bps


def test_delayed_mode_still_refuses_when_the_hedge_cannot_absorb_the_clip():
    strat = strategy(hedge={"mode": "delayed"})
    maker_book = book(MAKER_BIDS, MAKER_ASKS, venue="rh")
    hedge_book = book([("100.0", "0.0001")], [("100.05", "1")], venue="core")
    decision = strat.plan_entry(
        "BTC", maker_book, hedge_book, spec(), spec(), capacity_base=D("0.05")
    )
    assert not decision.ok
    assert "too thin" in decision.reason


def test_panic_trigger_only_fires_past_the_threshold():
    strat = strategy(hedge={"mode": "delayed", "panic_bps": 50})
    assert strat.should_panic_hedge(D(100), D("99.9")) is None      # 10bps against
    assert strat.should_panic_hedge(D(100), D("99.5")) == D(50)     # exactly at
    assert strat.should_panic_hedge(D(100), D(99)) == D(100)


def test_panic_trigger_is_inert_in_immediate_mode():
    strat = strategy(hedge={"mode": "immediate"})
    assert strat.should_panic_hedge(D(100), D(90)) is None


def test_naked_window_scales_with_volatility():
    strat = strategy(
        hedge={
            "mode": "delayed",
            "base_seconds": 30,
            "min_seconds": 10,
            "max_seconds": 30,
            "reference_vol_bps_per_min": 25,
        }
    )
    assert strat.naked_window_seconds(25.0) == 30.0
    assert strat.naked_window_seconds(75.0) == 10.0
    assert strat.naked_window_seconds(None) == 10.0


def test_min_entry_bps_can_be_overridden_per_market():
    strat = strategy(
        markets=[
            {"symbol": "BTC", "order_base": "0.01", "max_position_base": "0.05", "min_entry_bps": 500}
        ]
    )
    maker_book = book(MAKER_BIDS, MAKER_ASKS, venue="rh")
    hedge_book = book([("100.0", "10")], [("100.05", "10")], venue="core")
    decision = strat.plan_entry(
        "BTC", maker_book, hedge_book, spec(), spec(), capacity_base=D("0.05")
    )
    assert not decision.ok
