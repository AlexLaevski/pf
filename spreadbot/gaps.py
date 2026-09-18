"""Thin-book ("holey" order book) detection.

The idea the bot trades: a book whose top is sparse — a handful of dust orders
spread over a wide price range — and which only becomes dense some distance
away, where a real wall of liquidity sits. A sweep (liquidation, market order,
a taker in a hurry) walks through the sparse region almost unopposed and stops
at the wall.

So we rest our limit order *just in front of the wall*: one tick on the near
side of it. Three things have to hold for that to be worth doing:

* there is a genuine hole immediately above the wall (nothing to share the
  price level with, and a real price step to the next level up);
* very little size is queued ahead of us between the touch and our price, so a
  modest sweep actually reaches our order;
* the wall itself is big enough to stop the sweep, which is what makes the
  fill a good price rather than the first tick of a long slide.

This module only finds the geometry. Whether the resulting price is profitable
once hedged on the other venue is :mod:`spreadbot.strategy`'s problem.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional, Sequence

from .book import OrderBook
from .config import GapConfig
from .models import ZERO, Level, MarketSpec, Side


@dataclass(frozen=True)
class WallCandidate:
    """A place to rest a maker order, in front of a dense level."""

    side: Side
    price: Decimal              # where we would quote
    wall_price: Decimal         # first price of the dense region
    near_price: Decimal         # the level on the touch side of the hole - where a sweep came from
    wall_notional: Decimal      # USD resting in the wall cluster
    hole_ticks: Decimal         # empty price steps between the wall and the level above it
    hole_bps: Decimal           # same, as bps of mid
    ahead_notional: Decimal     # USD queued at prices better than ours
    distance_bps: Decimal       # how far our quote sits from mid (positive = cheaper for a buy)
    level_index: int            # index of the wall in the scanned side

    @property
    def is_touch_improving(self) -> bool:
        """True when the wall sits at the touch and we improve the best price."""
        return self.level_index == 0


def _baseline_notional(levels: Sequence[Level]) -> Decimal:
    """Typical level size in the scanned window, as a robust median."""
    notionals = [level.notional for level in levels if level.size > ZERO]
    if not notionals:
        return ZERO
    return Decimal(statistics.median(notionals))


def _ahead_notional(levels: Sequence[Level], side: Side, price: Decimal) -> Decimal:
    """Resting USD at prices strictly better than ``price`` on ``side``."""
    total = ZERO
    for level in levels:
        better = level.price > price if side is Side.BUY else level.price < price
        if not better:
            break
        total += level.notional
    return total


def find_wall_candidates(
    book: OrderBook,
    side: Side,
    spec: MarketSpec,
    cfg: GapConfig,
    *,
    max_ahead_notional_usd: Optional[Decimal] = None,
) -> List[WallCandidate]:
    """Walk ``side`` of ``book`` outward and return every viable quote spot.

    Candidates come back ordered from the touch outward, so the first entry is
    the closest wall (highest fill probability, smallest discount) and later
    entries trade fill probability for price.

    ``max_ahead_notional_usd`` overrides ``cfg.max_ahead_notional_usd`` for one
    call; when neither is set, queue depth is not filtered on.
    """
    ahead_limit = (
        max_ahead_notional_usd
        if max_ahead_notional_usd is not None
        else cfg.max_ahead_notional_usd
    )
    levels = list(book.side(side))[: cfg.max_levels_scan]
    mid = book.mid
    if not levels or mid is None or mid <= ZERO:
        return []

    far_touch = book.best_ask if side is Side.BUY else book.best_bid
    if far_touch is None:
        return []

    baseline = _baseline_notional(levels)
    tick = spec.tick
    offset = Decimal(cfg.join_offset_ticks) * tick
    cluster_n = max(1, cfg.min_wall_levels)
    candidates: List[WallCandidate] = []

    for index in range(len(levels) - cluster_n + 1):
        # A quote at the touch has no wall under it - it simply becomes the
        # best price, alone, which is a different (and unprotected) trade.
        if index == 0 and not cfg.allow_touch_improving:
            continue

        cluster = levels[index : index + cluster_n]
        if cluster_n > 1:
            span = abs(cluster[0].price - cluster[-1].price) / mid * Decimal(10_000)
            if span > cfg.max_wall_span_bps:
                # Levels this far apart are not one block of liquidity; summing
                # them would invent a wall that nothing would actually stop at.
                continue
        cluster_notional = sum((lvl.notional for lvl in cluster), ZERO)

        is_dense = cluster_notional >= cfg.wall_notional_usd and (
            baseline <= ZERO or cluster_notional >= cfg.wall_multiple * baseline * cluster_n
        )
        if not is_dense:
            continue

        wall_price = cluster[0].price
        # The level immediately nearer the touch; at the touch itself the hole is
        # the visible spread.
        if index > 0:
            near_price = levels[index - 1].price
        else:
            near_price = far_touch

        hole_abs = (near_price - wall_price) if side is Side.BUY else (wall_price - near_price)
        if hole_abs <= ZERO:
            continue
        hole_ticks = hole_abs / tick
        hole_bps = hole_abs / mid * Decimal(10_000)

        # Our quote must land strictly inside the hole, not on top of the level
        # above the wall.
        if hole_ticks <= Decimal(cfg.join_offset_ticks):
            continue
        if hole_ticks < Decimal(cfg.min_gap_ticks) or hole_bps < cfg.min_gap_bps:
            continue

        price = wall_price + offset if side is Side.BUY else wall_price - offset
        price = spec.quantize_price(price, side=side)
        if price <= ZERO:
            continue
        # Never cross: a buy stays below the best ask, a sell above the best bid.
        if side is Side.BUY and price >= far_touch:
            continue
        if side is Side.SELL and price <= far_touch:
            continue

        distance_bps = (
            (mid - price) if side is Side.BUY else (price - mid)
        ) / mid * Decimal(10_000)
        if distance_bps > cfg.max_distance_from_mid_bps:
            continue

        ahead = _ahead_notional(levels, side, price)
        if ahead_limit is not None and ahead > ahead_limit:
            continue

        candidates.append(
            WallCandidate(
                side=side,
                price=price,
                wall_price=wall_price,
                near_price=near_price,
                wall_notional=cluster_notional,
                hole_ticks=hole_ticks,
                hole_bps=hole_bps,
                ahead_notional=ahead,
                distance_bps=distance_bps,
                level_index=index,
            )
        )

    return candidates


def book_thinness_bps(book: OrderBook, side: Side, notional_usd: Decimal) -> Optional[Decimal]:
    """How far from mid you have to go to accumulate ``notional_usd`` on ``side``.

    A high number means a thin, gappy book; it is the single number worth
    logging when deciding whether a market is worth quoting at all.
    """
    mid = book.mid
    if mid is None or mid <= ZERO:
        return None
    total = ZERO
    for level in book.side(side):
        total += level.notional
        if total >= notional_usd:
            distance = (mid - level.price) if side is Side.BUY else (level.price - mid)
            return distance / mid * Decimal(10_000)
    return None
