from decimal import Decimal

from spreadbot.models import Side

from .conftest import book

D = Decimal


def test_snapshot_orders_levels():
    ob = book([("100", "1"), ("99", "2"), ("101", "3")], [("102", "1"), ("103", "5")])
    assert [level.price for level in ob.bids] == [D(101), D(100), D(99)]
    assert [level.price for level in ob.asks] == [D(102), D(103)]
    assert ob.best_bid == D(101)
    assert ob.best_ask == D(102)
    assert ob.mid == D("101.5")


def test_delta_updates_and_removes():
    ob = book([("100", "1"), ("99", "2")], [("102", "1")])
    ob.apply_delta([(D(100), D(0)), (D("99.5"), D(7))], [])
    assert [(l.price, l.size) for l in ob.bids] == [(D("99.5"), D(7)), (D(99), D(2))]


def test_executable_vwap_walks_the_book():
    ob = book([("100", "1")], [("102", "1"), ("103", "2")])
    vwap, filled = ob.executable_vwap(Side.BUY, D(2))
    assert filled == D(2)
    assert vwap == (D(102) * 1 + D(103) * 1) / 2


def test_executable_vwap_reports_short_fill():
    ob = book([("100", "1")], [("102", "1")])
    vwap, filled = ob.executable_vwap(Side.BUY, D(5))
    assert filled == D(1)
    assert vwap == D(102)


def test_executable_vwap_empty_side():
    ob = book([("100", "1")], [])
    assert ob.executable_vwap(Side.BUY, D(1)) is None


def test_size_at_or_better_counts_the_queue_ahead():
    ob = book([("100", "1"), ("99", "2"), ("98", "4")], [("101", "1")])
    assert ob.size_at_or_better(Side.BUY, D(99)) == D(3)
    assert ob.size_at_or_better(Side.BUY, D(98)) == D(7)


def test_depth_notional_respects_the_window():
    ob = book([("100", "1"), ("90", "10")], [("101", "1")])
    # mid is 100.5; 100 is ~50bps away, 90 is ~1045bps away.
    assert ob.depth_notional(Side.BUY, D(100)) == D(100)


def test_stale_book_is_not_fresh():
    ob = book([("100", "1")], [("101", "1")])
    assert ob.is_fresh(1_000)
    ob.updated_at -= 10
    assert not ob.is_fresh(1_000)


def test_clear_marks_not_ready():
    ob = book([("100", "1")], [("101", "1")])
    ob.clear()
    assert not ob.ready
    assert ob.best_bid is None
