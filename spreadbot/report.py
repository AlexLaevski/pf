"""Turn a measurement log into the numbers that decide whether to trade.

Deliberately blunt about what it does not know. The per-trade edge here is
what the recorder observed on the public tape; it does not model queue
position, and it assumes a sweep through our price would have filled our whole
clip. Both push the result optimistic. Treat the output as an upper bound that
is at least made of real observations.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SECONDS_PER_HOUR = 3_600.0


@dataclass
class Summary:
    quotes: int = 0
    fills: int = 0
    bounces: int = 0
    cancelled: int = 0
    expired: int = 0
    unmeasurable: int = 0     # never rested below the touch, so a fill cannot be inferred
    armed: int = 0            # quotes that did rest below the touch
    span_seconds: float = 0.0
    symbols: int = 0

    median_rest_seconds: Optional[float] = None
    median_seconds_to_bounce: Optional[float] = None
    median_adverse_bps: Optional[float] = None
    p90_adverse_bps: Optional[float] = None
    mean_realised_bps: Optional[float] = None
    median_realised_bps: Optional[float] = None
    mean_quoted_edge_bps: Optional[float] = None

    @property
    def fill_rate(self) -> float:
        """Share of measurable quotes that a sweep actually reached."""
        closed = self.fills + self.cancelled + self.expired
        return self.fills / closed if closed else 0.0

    @property
    def bounce_rate(self) -> float:
        return self.bounces / self.fills if self.fills else 0.0

    @property
    def fills_per_hour(self) -> float:
        return self.fills / (self.span_seconds / SECONDS_PER_HOUR) if self.span_seconds else 0.0

    def projected_usd(self, clip_usd: float, hours: float) -> Optional[float]:
        """Extrapolate to a horizon. Linear, and only as good as the sample."""
        if self.mean_realised_bps is None or not self.span_seconds:
            return None
        return self.fills_per_hour * hours * clip_usd * self.mean_realised_bps / 10_000.0


def _median(values: List[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def _quantile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(q * len(ordered)))
    return ordered[idx]


def load_rows(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def summarise(rows: Iterable[Dict[str, Any]]) -> Summary:
    rows = list(rows)
    out = Summary(quotes=len(rows))
    if not rows:
        return out

    fills = [r for r in rows if r.get("outcome") == "filled"]
    out.fills = len(fills)
    out.bounces = sum(1 for r in fills if r.get("bounce_hit"))
    out.cancelled = sum(1 for r in rows if r.get("outcome") == "cancelled")
    out.expired = sum(1 for r in rows if r.get("outcome") in ("expired", "cut_short"))
    out.unmeasurable = sum(1 for r in rows if r.get("outcome") == "unmeasurable")
    out.armed = sum(1 for r in rows if r.get("armed"))
    out.symbols = len({r.get("symbol") for r in rows})

    placed = [r["placed_at"] for r in rows if r.get("placed_at")]
    if placed:
        out.span_seconds = max(placed) - min(placed)

    out.median_rest_seconds = _median([r["rested_seconds"] for r in fills if r.get("rested_seconds")])
    out.median_seconds_to_bounce = _median(
        [r["seconds_to_bounce"] for r in fills if r.get("seconds_to_bounce") is not None]
    )
    adverse = [r["max_adverse_bps"] for r in fills if r.get("max_adverse_bps") is not None]
    out.median_adverse_bps = _median(adverse)
    out.p90_adverse_bps = _quantile(adverse, 0.9)
    realised = [r["realised_bps"] for r in fills if r.get("realised_bps") is not None]
    out.mean_realised_bps = sum(realised) / len(realised) if realised else None
    out.median_realised_bps = _median(realised)
    quoted = [r["edge_bps"] for r in rows if r.get("edge_bps") is not None]
    out.mean_quoted_edge_bps = sum(quoted) / len(quoted) if quoted else None
    return out


def per_symbol(rows: Iterable[Dict[str, Any]]) -> Dict[str, Summary]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(row.get("symbol", "?"), []).append(row)
    return {sym: summarise(rs) for sym, rs in buckets.items()}


def format_summary(summary: Summary, *, clip_usd: float) -> str:
    lines: List[str] = []
    hours = summary.span_seconds / SECONDS_PER_HOUR
    lines.append(f"наблюдение      : {hours:.2f} ч, {summary.symbols} рынков")
    lines.append(f"котировок       : {summary.quotes}")
    lines.append(
        f"  из них снято  : {summary.cancelled}  протухло: {summary.expired}"
        f"  неизмеримо: {summary.unmeasurable}"
    )
    lines.append(f"филлов          : {summary.fills}  ({summary.fill_rate * 100:.1f}% от закрытых)")
    lines.append(f"филлов в час    : {summary.fills_per_hour:.2f}")
    if summary.median_rest_seconds is not None:
        lines.append(f"медиана ожидания: {summary.median_rest_seconds:.1f} с в стакане до филла")
    lines.append(
        f"отскоков        : {summary.bounces} ({summary.bounce_rate * 100:.1f}% от филлов)"
    )
    if summary.median_seconds_to_bounce is not None:
        lines.append(f"  медиана       : {summary.median_seconds_to_bounce:.1f} с до отскока")
    if summary.median_adverse_bps is not None:
        lines.append(
            f"против нас      : медиана {summary.median_adverse_bps:.1f} bps, "
            f"p90 {summary.p90_adverse_bps:.1f} bps"
        )
    if summary.mean_quoted_edge_bps is not None:
        lines.append(f"эдж в котировке : {summary.mean_quoted_edge_bps:.1f} bps (до факта)")
    if summary.mean_realised_bps is not None:
        lines.append(
            f"ЭДЖ ПО ФАКТУ    : {summary.mean_realised_bps:+.1f} bps средний, "
            f"{summary.median_realised_bps:+.1f} bps медиана"
        )
        gap = (summary.mean_quoted_edge_bps or 0) - summary.mean_realised_bps
        lines.append(f"  потеря на adverse selection: {gap:.1f} bps")
        for horizon, label in ((24.0, "сутки"), (24 * 7.0, "неделя")):
            projected = summary.projected_usd(clip_usd, horizon)
            if projected is not None:
                lines.append(f"экстраполяция   : {projected:+.2f} USD за {label} (клип ${clip_usd:.0f})")
    else:
        lines.append("ЭДЖ ПО ФАКТУ    : нет ни одного филла - выборки нет")
    return "\n".join(lines)
