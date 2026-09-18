"""Funding is charged on the position, not the trade, and on a hot token it
can exceed the entire edge. These pin down that it is actually subtracted."""

from decimal import Decimal

import pytest

from spreadbot.config import parse_config
from spreadbot.strategy import SpreadStrategy

from .conftest import book, raw_config, spec

D = Decimal

MAKER_BIDS = [("100.0", "0.01"), ("99.9", "0.01"), ("99.8", "0.01"), ("99.5", "5"), ("99.4", "5")]
MAKER_ASKS = [("100.1", "1"), ("100.2", "1")]


def strategy(**overrides):
    return SpreadStrategy(parse_config(raw_config(**overrides)))


def test_no_rates_means_no_cost():
    strat = strategy()
    assert strat.funding_cost_bps("BTC") == 0


def test_cost_is_the_difference_between_the_two_venues():
    strat = strategy(funding={"expected_hold_minutes": 60})
    # Long pays 6.42 bps/hour on the maker venue, receives 1 bps on the hedge.
    strat.set_funding({"BTC": D("0.000642")}, {"BTC": D("0.0001")})
    assert strat.funding_cost_bps("BTC") == pytest.approx(D("5.42"), abs=0.01)


def test_identical_rates_cancel_out():
    strat = strategy(funding={"expected_hold_minutes": 600})
    strat.set_funding({"BTC": D("0.0005")}, {"BTC": D("0.0005")})
    assert strat.funding_cost_bps("BTC") == 0


def test_a_richer_hedge_rate_pays_us():
    strat = strategy(funding={"expected_hold_minutes": 60})
    strat.set_funding({"BTC": D("0.0001")}, {"BTC": D("0.0005")})
    assert strat.funding_cost_bps("BTC") < 0


def test_cost_scales_with_the_hold():
    strat = strategy(funding={"expected_hold_minutes": 15})
    strat.set_funding({"BTC": D("0.000642")}, {})
    quarter_hour = strat.funding_cost_bps("BTC")
    assert quarter_hour == pytest.approx(D("1.605"), abs=0.01)
    assert strat.funding_cost_bps("BTC", hold_hours=D(4)) == pytest.approx(D("25.68"), abs=0.01)


def test_disabled_funding_is_ignored():
    strat = strategy(funding={"enabled": False, "expected_hold_minutes": 600})
    strat.set_funding({"BTC": D("0.01")}, {})
    assert strat.funding_cost_bps("BTC") == 0


def test_funding_is_subtracted_from_the_entry_edge():
    maker_book = book(MAKER_BIDS, MAKER_ASKS, venue="rh")
    hedge_book = book([("100.0", "10")], [("100.05", "10")], venue="core")

    free = strategy(funding={"enabled": False})
    costly = strategy(funding={"expected_hold_minutes": 60, "max_cost_bps": 100})
    costly.set_funding({"BTC": D("0.001")}, {})      # 10 bps/hour against us

    args = ("BTC", maker_book, hedge_book, spec(), spec())
    free_edge = free.plan_entry(*args, capacity_base=D("0.05")).plan.edge_bps
    costly_edge = costly.plan_entry(*args, capacity_base=D("0.05")).plan.edge_bps
    assert free_edge - costly_edge == pytest.approx(D(10), abs=0.01)


def test_market_is_skipped_when_funding_alone_eats_the_trade():
    strat = strategy(funding={"expected_hold_minutes": 60, "max_cost_bps": 5})
    strat.set_funding({"BTC": D("0.002")}, {})       # 20 bps/hour, limit is 5
    decision = strat.plan_entry(
        "BTC",
        book(MAKER_BIDS, MAKER_ASKS, venue="rh"),
        book([("100.0", "10")], [("100.05", "10")], venue="core"),
        spec(),
        spec(),
        capacity_base=D("0.05"),
    )
    assert not decision.ok
    assert "funding costs" in decision.reason
