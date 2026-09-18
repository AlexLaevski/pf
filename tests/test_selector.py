from decimal import Decimal

from spreadbot.config import GapConfig
from spreadbot.selector import MarketRanker, clip_size

from .conftest import book, spec

D = Decimal

GAP = GapConfig(
    wall_notional_usd=D(300),
    wall_multiple=D(4),
    min_gap_bps=D(5),
    min_gap_ticks=2,
    join_offset_ticks=1,
    max_distance_from_mid_bps=D(60),
)

HOLEY = ([("100.0", "0.01"), ("99.9", "0.01"), ("99.8", "0.01"), ("99.5", "5"), ("99.4", "5")],
         [("100.1", "1"), ("100.2", "1")])
DENSE = ([(str(D("100.0") - D("0.1") * i), "5") for i in range(10)], [("100.1", "1")])


def ranker() -> MarketRanker:
    return MarketRanker(gap=GAP)


def test_market_with_a_gap_scores_above_zero():
    r = ranker()
    r.observe(book(*HOLEY, symbol="GAPPY"), spec("GAPPY"))
    score = r.scores["GAPPY"]
    assert score.hits == 1
    assert score.hit_rate == 1
    assert score.avg_edge_bps > 0


def test_dense_market_scores_zero():
    r = ranker()
    r.observe(book(*DENSE, symbol="DENSE"), spec("DENSE"))
    assert r.scores["DENSE"].hits == 0
    assert r.scores["DENSE"].score == 0


def test_score_is_time_in_market_times_edge():
    r = ranker()
    # Two samples with a gap, two without: half the time, same edge.
    for _ in range(2):
        r.observe(book(*HOLEY, symbol="HALF"), spec("HALF"))
    for _ in range(2):
        r.observe(book(*DENSE, symbol="HALF"), spec("HALF"))
    score = r.scores["HALF"]
    assert score.samples == 4
    assert score.hits == 2
    assert score.hit_rate == D("0.5")
    assert score.score == score.hit_rate * score.avg_edge_bps


def test_always_gappy_market_outranks_a_sometimes_gappy_one():
    r = ranker()
    for _ in range(4):
        r.observe(book(*HOLEY, symbol="ALWAYS"), spec("ALWAYS"))
    for _ in range(3):
        r.observe(book(*DENSE, symbol="SOMETIMES"), spec("SOMETIMES"))
    r.observe(book(*HOLEY, symbol="SOMETIMES"), spec("SOMETIMES"))
    assert r.top_symbols(2) == ["ALWAYS", "SOMETIMES"]


def test_ranked_skips_markets_that_never_showed_a_gap():
    r = ranker()
    r.observe(book(*DENSE, symbol="DENSE"), spec("DENSE"))
    r.observe(book(*HOLEY, symbol="GAPPY"), spec("GAPPY"))
    assert [s.symbol for s in r.ranked()] == ["GAPPY"]


def test_unready_book_is_ignored():
    r = ranker()
    empty = book([], [], symbol="EMPTY")
    r.observe(empty, spec("EMPTY"))
    assert "EMPTY" not in r.scores


class TestClipSize:
    def test_targets_the_requested_notional(self):
        s = spec(price_decimals=1, size_decimals=3, min_base="0.001", min_quote="0")
        assert clip_size(s, D(100), D(50)) == D("0.5")

    def test_respects_the_venue_base_minimum(self):
        # The minimum is above the requested clip but within tolerance, so it
        # is taken as-is rather than rounded down below what the venue accepts.
        s = spec(size_decimals=3, min_base="1", min_quote="0")
        assert clip_size(s, D(100), D(50)) == D(1)

    def test_lifts_size_until_the_quote_minimum_is_met(self):
        s = spec(size_decimals=2, min_base="0.01", min_quote="10")
        size = clip_size(s, D(100), D(5))
        assert size is not None and size * D(100) >= D(10)

    def test_refuses_a_market_whose_minimum_dwarfs_the_clip(self):
        s = spec(size_decimals=0, min_base="100", min_quote="0")
        assert clip_size(s, D(100), D(50)) is None

    def test_zero_price_is_refused(self):
        assert clip_size(spec(), D(0), D(50)) is None
