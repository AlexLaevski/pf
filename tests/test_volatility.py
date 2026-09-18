import math
from decimal import Decimal

from spreadbot.volatility import VolatilityTracker, scaled_window_seconds

D = Decimal


def feed(tracker: VolatilityTracker, prices, *, start=1_000.0, step=1.0):
    for i, price in enumerate(prices):
        tracker.update(start + i * step, D(str(price)))


def test_needs_enough_samples():
    vol = VolatilityTracker(min_samples=5)
    feed(vol, [100, 100.1, 100.2])
    assert vol.bps_per_minute() is None


def test_flat_price_has_zero_vol():
    vol = VolatilityTracker(min_samples=5)
    feed(vol, [100] * 20)
    assert vol.bps_per_minute() == 0.0


def test_jumpier_series_reads_higher():
    calm = VolatilityTracker(min_samples=5)
    wild = VolatilityTracker(min_samples=5)
    feed(calm, [100 + 0.01 * (-1) ** i for i in range(40)])
    feed(wild, [100 + 1.0 * (-1) ** i for i in range(40)])
    assert wild.bps_per_minute() > calm.bps_per_minute() * 10


def test_known_series_matches_hand_calculation():
    # 1% alternating moves, one sample per second.
    vol = VolatilityTracker(min_samples=5)
    prices = [100.0 * (1.01 ** (i % 2)) for i in range(61)]
    feed(vol, prices)
    # per-second sigma is ln(1.01); over a minute that is sqrt(60) times bigger.
    expected = math.log(1.01) * math.sqrt(60) * 10_000
    assert abs(vol.bps_per_minute() - expected) / expected < 0.02


def test_window_drops_old_samples():
    vol = VolatilityTracker(window_seconds=10, min_samples=3)
    feed(vol, [100] * 30)
    assert vol.samples <= 11


def test_rate_limits_samples():
    vol = VolatilityTracker(min_sample_seconds=1.0, min_samples=2)
    for i in range(10):
        vol.update(1_000.0 + i * 0.1, D(100))
    assert vol.samples == 1


def test_ignores_bad_prices():
    vol = VolatilityTracker(min_samples=2)
    vol.update(1_000.0, None)
    vol.update(1_001.0, D(0))
    vol.update(1_002.0, D(-5))
    assert vol.samples == 0


class TestWindowScaling:
    kwargs = dict(base_seconds=30.0, reference_vol_bps=25.0, min_seconds=8.0, max_seconds=45.0)

    def test_reference_vol_gives_base_window(self):
        assert scaled_window_seconds(25.0, **self.kwargs) == 30.0

    def test_more_volatile_shortens_the_window(self):
        # Three times the reference vol would be 10s, which is what we want.
        assert scaled_window_seconds(75.0, **self.kwargs) == 10.0

    def test_calm_lengthens_but_clamps(self):
        assert scaled_window_seconds(5.0, **self.kwargs) == 45.0

    def test_very_volatile_clamps_at_minimum(self):
        assert scaled_window_seconds(10_000.0, **self.kwargs) == 8.0

    def test_unknown_vol_is_treated_as_fast(self):
        assert scaled_window_seconds(None, **self.kwargs) == 8.0

    def test_zero_vol_gets_the_full_window(self):
        assert scaled_window_seconds(0.0, **self.kwargs) == 45.0
