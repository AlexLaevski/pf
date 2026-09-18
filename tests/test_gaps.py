from decimal import Decimal

from spreadbot.config import GapConfig
from spreadbot.gaps import book_thinness_bps, find_wall_candidates
from spreadbot.models import Side

from .conftest import book, spec

D = Decimal

# A deliberately holey bid side: dust at the touch, then a real wall 30bps down.
HOLEY_BIDS = [
    ("100.0", "0.01"),
    ("99.9", "0.01"),
    ("99.8", "0.01"),
    ("99.5", "5"),
    ("99.4", "5"),
]
ASKS = [("100.1", "1"), ("100.2", "1")]

GAP = GapConfig(
    wall_notional_usd=D(300),
    wall_multiple=D(4),
    min_gap_bps=D(5),
    min_gap_ticks=2,
    join_offset_ticks=1,
    max_distance_from_mid_bps=D(60),
)


def test_finds_the_wall_and_quotes_one_tick_in_front():
    ob = book(HOLEY_BIDS, ASKS)
    candidates = find_wall_candidates(ob, Side.BUY, spec(), GAP)
    assert candidates, "expected a wall behind the hole"
    best = candidates[0]
    assert best.wall_price == D("99.5")
    assert best.price == D("99.6")          # one tick in front of the wall
    assert best.level_index == 3
    assert best.hole_ticks == D(3)
    assert best.ahead_notional < D(5)       # almost nothing queued ahead of us


def test_dense_book_has_no_candidates():
    dense = [(str(D("100.0") - D("0.1") * i), "5") for i in range(10)]
    ob = book(dense, ASKS)
    assert find_wall_candidates(ob, Side.BUY, spec(), GAP) == []


def test_gap_too_narrow_is_rejected():
    tight = GapConfig(
        wall_notional_usd=D(300),
        wall_multiple=D(4),
        min_gap_bps=D(5),
        min_gap_ticks=5,          # the hole is only 3 ticks
        join_offset_ticks=1,
        max_distance_from_mid_bps=D(60),
    )
    ob = book(HOLEY_BIDS, ASKS)
    assert find_wall_candidates(ob, Side.BUY, spec(), tight) == []


def test_wall_too_far_from_mid_is_rejected():
    near = GapConfig(
        wall_notional_usd=D(300),
        wall_multiple=D(4),
        min_gap_bps=D(5),
        min_gap_ticks=2,
        join_offset_ticks=1,
        max_distance_from_mid_bps=D(10),   # the wall sits ~45bps away
    )
    ob = book(HOLEY_BIDS, ASKS)
    assert find_wall_candidates(ob, Side.BUY, spec(), near) == []


def test_small_wall_is_not_a_wall():
    big = GapConfig(
        wall_notional_usd=D(100_000),
        wall_multiple=D(4),
        min_gap_bps=D(5),
        min_gap_ticks=2,
        join_offset_ticks=1,
        max_distance_from_mid_bps=D(60),
    )
    ob = book(HOLEY_BIDS, ASKS)
    assert find_wall_candidates(ob, Side.BUY, spec(), big) == []


def test_quote_never_crosses_the_spread():
    ob = book(HOLEY_BIDS, ASKS)
    for candidate in find_wall_candidates(ob, Side.BUY, spec(), GAP):
        assert candidate.price < ob.best_ask


def test_ask_side_is_the_mirror_image():
    asks = [
        ("100.1", "0.01"),
        ("100.2", "0.01"),
        ("100.3", "0.01"),
        ("100.6", "5"),
    ]
    ob = book([("100.0", "1")], asks)
    candidates = find_wall_candidates(ob, Side.SELL, spec(), GAP)
    assert candidates
    assert candidates[0].wall_price == D("100.6")
    assert candidates[0].price == D("100.5")
    assert candidates[0].price > ob.best_bid


def test_ahead_notional_filter():
    ob = book(HOLEY_BIDS, ASKS)
    assert find_wall_candidates(
        ob, Side.BUY, spec(), GAP, max_ahead_notional_usd=D("0.5")
    ) == []


def test_thinness_measures_distance_to_liquidity():
    ob = book(HOLEY_BIDS, ASKS)
    # 0.03 base of dust at the touch cannot cover 100 USD; the wall does.
    assert book_thinness_bps(ob, Side.BUY, D(100)) > D(40)
    assert book_thinness_bps(ob, Side.BUY, D(10_000)) is None


def test_multi_level_wall_cluster():
    cfg = GapConfig(
        wall_notional_usd=D(900),
        wall_multiple=D(4),
        min_gap_bps=D(5),
        min_gap_ticks=2,
        join_offset_ticks=1,
        max_distance_from_mid_bps=D(60),
        min_wall_levels=2,
    )
    ob = book(HOLEY_BIDS, ASKS)
    candidates = find_wall_candidates(ob, Side.BUY, spec(), cfg)
    assert candidates and candidates[0].wall_price == D("99.5")
    assert candidates[0].wall_notional > D(900)
