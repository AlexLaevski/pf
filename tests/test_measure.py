"""The recorder decides whether the whole strategy is worth running, so its
one job — not inventing fills — is pinned down here."""

from decimal import Decimal

from spreadbot.config import parse_config
from spreadbot.measure import MeasureConfig, Recorder
from spreadbot.report import summarise
from spreadbot.strategy import SpreadStrategy

from .conftest import book, raw_config, spec

D = Decimal

HOLEY_BIDS = [("100.0", "0.01"), ("99.9", "0.01"), ("99.8", "0.01"), ("99.5", "5"), ("99.4", "5")]
MAKER_ASKS = [("100.1", "1"), ("100.2", "1")]
HEDGE = ([("100.0", "10")], [("100.05", "10")])


def make_recorder(**overrides):
    cfg = parse_config(
        raw_config(
            hedge={"mode": "delayed", "min_seconds": 10, "max_seconds": 30},
            bounce={"target": "hole_top", "min_target_bps": 3},
            **overrides,
        )
    )
    specs = {"BTC": spec("BTC")}
    return Recorder(
        cfg,
        SpreadStrategy(cfg),
        specs,
        dict(specs),
        measure=MeasureConfig(horizon_seconds=30, max_rest_seconds=300, requote_bps=2),
    )


def tick(rec, bids, asks, *, at, hedge=HEDGE):
    rec.tick(
        "BTC",
        book(bids, asks, venue="rh"),
        book(hedge[0], hedge[1], venue="core"),
        now=at,
        vol_bps_per_min=25.0,
    )


def test_quote_is_recorded_with_its_target():
    rec = make_recorder()
    tick(rec, HOLEY_BIDS, MAKER_ASKS, at=1_000)
    quote = rec.open["BTC"]
    assert quote.price == 99.6
    assert quote.target == 99.8
    assert quote.armed is False


def test_quote_below_the_touch_arms_but_does_not_fill():
    rec = make_recorder()
    tick(rec, HOLEY_BIDS, MAKER_ASKS, at=1_000)
    tick(rec, HOLEY_BIDS, MAKER_ASKS, at=1_001)
    assert rec.open["BTC"].armed is True
    assert rec.stats.fills == 0


def test_fill_needs_the_book_to_trade_through_the_quote():
    rec = make_recorder()
    tick(rec, HOLEY_BIDS, MAKER_ASKS, at=1_000)
    tick(rec, HOLEY_BIDS, MAKER_ASKS, at=1_001)      # arms
    tick(rec, [("99.5", "5")], [("99.55", "5")], at=1_002)   # swept through 99.6
    assert rec.stats.fills == 1
    assert rec.tracking[0].filled_at == 1_002


def test_touch_improving_quote_is_never_counted_as_a_fill():
    # The wall sits at the touch, so the quote IS the new best bid. Public data
    # cannot tell whether anyone traded with it - counting it would manufacture
    # a fill per tick, which is exactly the bug this guards.
    rec = make_recorder()
    bids = [("99.5", "5"), ("99.4", "5")]            # wall at the touch
    tick(rec, bids, [("100.1", "1")], at=1_000)
    for i in range(1, 10):
        tick(rec, bids, [("100.1", "1")], at=1_000 + i)
    assert rec.stats.fills == 0


def test_bounce_is_recorded_when_the_target_is_reached():
    rec = make_recorder()
    tick(rec, HOLEY_BIDS, MAKER_ASKS, at=1_000)
    tick(rec, HOLEY_BIDS, MAKER_ASKS, at=1_001)
    tick(rec, [("99.5", "5")], [("99.55", "5")], at=1_002)
    tick(rec, [("99.85", "5")], [("99.9", "5")], at=1_005)   # ask above target
    rec.finish(now=1_006)
    summary = summarise(rec.rows)
    assert summary.fills == 1
    assert summary.bounces == 1
    assert summary.mean_realised_bps > 0


def test_no_bounce_books_the_hedge_price_including_the_loss():
    rec = make_recorder()
    tick(rec, HOLEY_BIDS, MAKER_ASKS, at=1_000)
    tick(rec, HOLEY_BIDS, MAKER_ASKS, at=1_001)
    tick(rec, [("99.5", "5")], [("99.55", "5")], at=1_002)
    # Hedge venue drops 1% below entry and stays there past the horizon.
    crashed = ([("98.6", "10")], [("98.65", "10")])
    tick(rec, [("98.6", "5")], [("98.65", "5")], at=1_010, hedge=crashed)
    tick(rec, [("98.6", "5")], [("98.65", "5")], at=1_040, hedge=crashed)
    rec.finish(now=1_041)
    summary = summarise(rec.rows)
    assert summary.fills == 1
    assert summary.bounces == 0
    assert summary.mean_realised_bps < -90       # ~1% against us, recorded as such
    assert summary.median_adverse_bps > 90


def test_adverse_excursion_keeps_the_worst_point_not_the_last():
    rec = make_recorder()
    tick(rec, HOLEY_BIDS, MAKER_ASKS, at=1_000)
    tick(rec, HOLEY_BIDS, MAKER_ASKS, at=1_001)
    tick(rec, [("99.5", "5")], [("99.55", "5")], at=1_002)
    deep = ([("98.0", "10")], [("98.05", "10")])
    tick(rec, [("98.0", "5")], [("98.05", "5")], at=1_005, hedge=deep)
    tick(rec, [("99.5", "5")], [("99.55", "5")], at=1_010)   # recovers, but not to target
    rec.finish(now=1_040)
    row = rec.rows[-1]
    assert row["max_adverse_bps"] > 150


def test_vanishing_opportunity_cancels_the_quote():
    rec = make_recorder()
    tick(rec, HOLEY_BIDS, MAKER_ASKS, at=1_000)
    tick(rec, HOLEY_BIDS, MAKER_ASKS, at=1_001)
    dense = [(str(D("100.0") - D("0.1") * i), "5") for i in range(10)]
    tick(rec, dense, MAKER_ASKS, at=1_002)
    assert "BTC" not in rec.open
    assert rec.rows[-1]["outcome"] == "cancelled"


def test_finish_flushes_everything_in_flight():
    rec = make_recorder()
    tick(rec, HOLEY_BIDS, MAKER_ASKS, at=1_000)
    assert rec.rows == []
    rec.finish(now=1_010)
    assert len(rec.rows) == 1
    # Never armed, so it must not be reported as a measurable miss.
    assert rec.rows[0]["outcome"] == "unmeasurable"


def test_summary_of_an_empty_run_is_honest():
    summary = summarise([])
    assert summary.fills == 0
    assert summary.mean_realised_bps is None
    assert summary.projected_usd(50, 24) is None
