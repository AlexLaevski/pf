"""Realized volatility off the mid series.

Used for one decision: how long the bot is willing to sit in a naked long
after a gap fill, waiting for the bounce, before it gives up and hedges.
A calm token earns a long window, a jumpy one a short window.

The estimator is a plain rolling variance of log returns, normalised by the
time actually elapsed rather than by the sample count — the book updates
irregularly, so counting samples would make a quiet minute look identical to
a busy second.
"""

from __future__ import annotations

import math
from collections import deque
from decimal import Decimal
from typing import Deque, Optional, Tuple

SECONDS_PER_MINUTE = 60.0
BPS = 10_000.0


class VolatilityTracker:
    """Rolling realized volatility for one market, in bps per minute."""

    __slots__ = ("window_seconds", "min_sample_seconds", "min_samples", "_samples", "_last_ts")

    def __init__(
        self,
        *,
        window_seconds: float = 180.0,
        min_sample_seconds: float = 0.25,
        min_samples: int = 12,
    ) -> None:
        self.window_seconds = window_seconds
        self.min_sample_seconds = min_sample_seconds
        self.min_samples = min_samples
        self._samples: Deque[Tuple[float, float]] = deque()   # (ts, log price)
        self._last_ts: float = 0.0

    def update(self, ts: float, mid: Optional[Decimal]) -> None:
        """Feed a new mid. Cheap enough to call on every book update."""
        if mid is None or mid <= 0:
            return
        if ts - self._last_ts < self.min_sample_seconds:
            return
        self._last_ts = ts
        self._samples.append((ts, math.log(float(mid))))
        cutoff = ts - self.window_seconds
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    @property
    def samples(self) -> int:
        return len(self._samples)

    def bps_per_minute(self) -> Optional[float]:
        """Realized vol as bps of price per minute, or None when unmeasured."""
        if len(self._samples) < self.min_samples:
            return None
        elapsed = self._samples[-1][0] - self._samples[0][0]
        if elapsed <= 0:
            return None
        total = 0.0
        previous = self._samples[0][1]
        for _, log_price in list(self._samples)[1:]:
            delta = log_price - previous
            total += delta * delta
            previous = log_price
        variance_per_second = total / elapsed
        if variance_per_second <= 0:
            return 0.0
        return math.sqrt(variance_per_second * SECONDS_PER_MINUTE) * BPS

    def reset(self) -> None:
        self._samples.clear()
        self._last_ts = 0.0


def scaled_window_seconds(
    vol_bps_per_minute: Optional[float],
    *,
    base_seconds: float,
    reference_vol_bps: float,
    min_seconds: float,
    max_seconds: float,
) -> float:
    """How long to wait for the bounce, given the current volatility.

    At ``reference_vol_bps`` the window is ``base_seconds``; twice as volatile
    halves it, half as volatile doubles it, and the result is clamped. When
    volatility is not measurable yet, the cautious end (``min_seconds``) wins —
    an unknown token is treated as a fast one.
    """
    low, high = min(min_seconds, max_seconds), max(min_seconds, max_seconds)
    if vol_bps_per_minute is None:
        return low
    if vol_bps_per_minute <= 0:
        return high
    scaled = base_seconds * (reference_vol_bps / vol_bps_per_minute)
    return max(low, min(high, scaled))
