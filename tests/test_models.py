from decimal import Decimal

import pytest

from spreadbot.models import Fill, PairPosition, Position, Side

from .conftest import spec

D = Decimal


def test_quantize_price_never_becomes_more_aggressive():
    s = spec(price_decimals=1)
    assert s.quantize_price(D("99.67"), side=Side.BUY) == D("99.6")
    assert s.quantize_price(D("99.61"), side=Side.SELL) == D("99.7")


def test_quantize_size_rounds_down():
    s = spec(size_decimals=3)
    assert s.quantize_size(D("0.12399")) == D("0.123")


def test_integer_encoding_matches_lighter_scaling():
    s = spec(price_decimals=2, size_decimals=4)
    assert s.price_to_int(D("1500")) == 150_000
    assert s.size_to_int(D("1")) == 10_000


def test_side_maps_to_is_ask():
    assert Side.SELL.is_ask is True
    assert Side.BUY.is_ask is False
    assert Side.BUY.opposite is Side.SELL


def test_position_averages_then_realises():
    pos = Position("v", "BTC")
    pos.apply(Fill("v", "BTC", Side.BUY, D(100), D(1)))
    pos.apply(Fill("v", "BTC", Side.BUY, D(102), D(1)))
    assert pos.size == D(2)
    assert pos.entry_price == D(101)
    pos.apply(Fill("v", "BTC", Side.SELL, D(105), D(1)))
    assert pos.size == D(1)
    assert pos.realized_pnl == D(4)


def test_pair_position_tracks_the_unhedged_leg():
    pair = PairPosition("BTC", "rh", "core")
    pair.add_long(D("99.6"), D(1))
    assert pair.unhedged == D(1)
    pair.add_short(D("100.0"), D(1))
    assert pair.unhedged == D(0)
    assert pair.matched == D(1)
    assert pair.entry_spread_bps == pytest.approx(D("40.16"), abs=0.05)


def test_pair_position_round_trip_pnl():
    pair = PairPosition("BTC", "rh", "core")
    pair.add_long(D("99.6"), D(1))
    pair.add_short(D("100.0"), D(1))
    pair.reduce_long(D("100.0"), D(1))       # sell the long 0.4 higher
    pair.reduce_short(D("100.0"), D(1))      # buy the short back flat
    assert pair.realized_pnl == D("0.4")
    assert pair.is_flat
    assert pair.opened_at is None
