"""Which markets are worth quoting at all.

The binding constraint on this strategy is not fill probability — it is time
in market. A quote can only be hit while it is resting, and a market only
offers a place to rest when its book has a hole in front of a wall. On a deep
book that is almost never; on a thin one it can be most of the time.

So markets are ranked on the product of the two things that matter:

    score = (доля времени с пригодной дыркой) × (средний эдж этой дырки)

which is, up to a constant, the expected bps per unit of time spent watching
the market. A market with a huge but once-a-day gap and one with a permanent
tiny gap both score low, and that is the intent.

Accumulate observations with :meth:`MarketRanker.observe` over as many passes
as you can afford, then read :meth:`MarketRanker.ranked`. One pass is enough
to weed out the hopeless; several give a usable ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional

from .book import OrderBook
from .config import GapConfig
from .gaps import find_wall_candidates
from .models import ZERO, MarketSpec, Side

BPS = Decimal(10_000)


@dataclass
class MarketScore:
    symbol: str
    samples: int = 0
    hits: int = 0
    edge_sum: Decimal = ZERO
    best_edge_bps: Decimal = ZERO
    distance_sum: Decimal = ZERO
    spread_sum: Decimal = ZERO
    depth_sum: Decimal = ZERO
    volume_usd: float = 0.0

    @property
    def hit_rate(self) -> Decimal:
        return Decimal(self.hits) / Decimal(self.samples) if self.samples else ZERO

    @property
    def avg_edge_bps(self) -> Decimal:
        return self.edge_sum / Decimal(self.hits) if self.hits else ZERO

    @property
    def avg_distance_bps(self) -> Decimal:
        return self.distance_sum / Decimal(self.hits) if self.hits else ZERO

    @property
    def avg_spread_bps(self) -> Decimal:
        return self.spread_sum / Decimal(self.samples) if self.samples else ZERO

    @property
    def avg_depth_usd(self) -> Decimal:
        return self.depth_sum / Decimal(self.samples) if self.samples else ZERO

    @property
    def score(self) -> Decimal:
        """Expected bps per unit of watching time."""
        return self.hit_rate * self.avg_edge_bps


@dataclass
class MarketRanker:
    """Accumulates gap statistics across repeated book snapshots."""

    gap: GapConfig
    depth_window_bps: Decimal = Decimal(100)
    scores: Dict[str, MarketScore] = field(default_factory=dict)

    def observe(self, book: OrderBook, spec: MarketSpec, *, volume_usd: float = 0.0) -> None:
        if not book.ready or book.mid is None:
            return
        score = self.scores.setdefault(book.symbol, MarketScore(book.symbol))
        score.samples += 1
        score.volume_usd = volume_usd or score.volume_usd
        score.spread_sum += book.spread_bps or ZERO
        score.depth_sum += book.depth_notional(Side.BUY, self.depth_window_bps)

        candidates = find_wall_candidates(
            book, Side.BUY, spec, self.gap, max_ahead_notional_usd=self.gap.max_ahead_notional_usd
        )
        if not candidates:
            return

        # The gross bounce: from the quote price back to the near edge of the
        # hole. Fees are left out here - they are per-venue and this ranking is
        # about the shape of the book, not the P&L of a specific account.
        best = max(
            candidates,
            key=lambda c: (c.near_price - c.price) / c.price if c.price > ZERO else ZERO,
        )
        edge = (best.near_price - best.price) / best.price * BPS
        score.hits += 1
        score.edge_sum += edge
        score.distance_sum += best.distance_bps
        score.best_edge_bps = max(score.best_edge_bps, edge)

    def ranked(self, *, min_samples: int = 1, min_hits: int = 1) -> List[MarketScore]:
        rows = [
            s
            for s in self.scores.values()
            if s.samples >= min_samples and s.hits >= min_hits
        ]
        return sorted(rows, key=lambda s: s.score, reverse=True)

    def top_symbols(self, count: int, **kwargs) -> List[str]:
        return [s.symbol for s in self.ranked(**kwargs)[:count]]


def clip_size(spec: MarketSpec, price: Decimal, clip_usd: Decimal) -> Optional[Decimal]:
    """Base size closest to ``clip_usd`` that the venue will actually accept."""
    if price <= ZERO:
        return None
    size = spec.quantize_size(clip_usd / price)
    if size < spec.min_base_amount:
        size = spec.min_base_amount
    # Nudge up until the notional clears the venue floor, but never chase a
    # market whose minimum clip is far past what we asked for.
    guard = 0
    while size * price < spec.min_quote_amount and guard < 10_000:
        size += spec.lot
        guard += 1
    if size * price > clip_usd * 3:
        return None
    return size
