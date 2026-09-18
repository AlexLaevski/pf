"""Risk limits and the kill switch.

The bot is only safe because of what this module refuses. The important limit
is not the P&L cap — it is ``max_unhedged_notional_usd`` together with
``unhedged_timeout_ms``: a filled long that has not been hedged is naked
directional risk, and every other rule is secondary to getting that back to
zero.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Dict, Optional, Tuple

from .config import RiskConfig
from .models import ZERO, PairPosition

log = logging.getLogger(__name__)


@dataclass
class RiskState:
    realized_pnl: Decimal = ZERO
    fees_paid: Decimal = ZERO
    consecutive_errors: int = 0
    halted: bool = False
    halt_reason: str = ""
    # A daily loss limit is by definition daily and clears at the day roll.
    # Everything else (a failed hedge, a position on the forbidden side) is a
    # statement about the world that a new calendar day does not change.
    halt_kind: str = ""
    day: str = field(default_factory=lambda: time.strftime("%Y-%m-%d"))

    @property
    def net_pnl(self) -> Decimal:
        return self.realized_pnl - self.fees_paid


class RiskManager:
    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg
        self.state = RiskState()
        self._unhedged_since: Dict[str, float] = {}

    # ------------------------------------------------------------------ state

    @property
    def halted(self) -> bool:
        # Checked here rather than only when P&L is booked: a halt stops
        # trading, trading is what books P&L, so a roll driven by P&L alone
        # would never come and the halt would be permanent.
        self._roll_day()
        if self.state.halted:
            return True
        if self.cfg.kill_switch_file and Path(self.cfg.kill_switch_file).exists():
            self.halt(f"kill switch file present: {self.cfg.kill_switch_file}", kind="kill_switch")
            return True
        return False

    def halt(self, reason: str, *, kind: str = "manual") -> None:
        if not self.state.halted:
            log.error("HALT: %s", reason)
        self.state.halted = True
        self.state.halt_reason = reason
        self.state.halt_kind = kind

    def resume(self) -> None:
        self.state.halted = False
        self.state.halt_reason = ""
        self.state.halt_kind = ""
        self.state.consecutive_errors = 0

    def note_error(self, what: str) -> None:
        self.state.consecutive_errors += 1
        log.warning("error %d/%d: %s", self.state.consecutive_errors, self.cfg.max_consecutive_errors, what)
        if self.state.consecutive_errors >= self.cfg.max_consecutive_errors:
            self.halt(
                f"{self.state.consecutive_errors} consecutive errors, last: {what}",
                kind="errors",
            )

    def note_ok(self) -> None:
        self.state.consecutive_errors = 0

    def book_pnl(self, realized: Decimal, fees: Decimal = ZERO) -> None:
        self._roll_day()
        self.state.realized_pnl += realized
        self.state.fees_paid += fees
        if self.state.net_pnl <= -self.cfg.max_daily_loss_usd:
            self.halt(
                f"daily loss limit hit ({self.state.net_pnl:.2f} USD)", kind="daily_loss"
            )

    def _roll_day(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if today == self.state.day:
            return
        log.info("new trading day %s; previous day net P&L %.2f USD", today, self.state.net_pnl)

        # A daily loss halt expires with the day it belongs to. Any other halt
        # describes something still true, so it survives the roll and stays for
        # a human to clear.
        if self.state.halted and self.state.halt_kind == "daily_loss":
            log.warning(
                "daily loss halt from %s cleared by the day roll; trading resumes",
                self.state.day,
            )
            self.state = RiskState(day=today)
        else:
            self.state = RiskState(
                day=today,
                halted=self.state.halted,
                halt_reason=self.state.halt_reason,
                halt_kind=self.state.halt_kind,
            )

    # ----------------------------------------------------------------- limits

    def can_open(
        self,
        pair: PairPosition,
        *,
        mark: Decimal,
        clip_base: Decimal,
        max_position_base: Decimal,
        total_open_notional: Decimal,
    ) -> Tuple[bool, str]:
        if self.halted:
            return False, f"halted: {self.state.halt_reason}"
        if pair.unhedged > ZERO:
            return False, "position is unhedged; no new quotes until it is flat"
        if pair.long_size + clip_base > max_position_base:
            return False, (
                f"max_position_base reached ({pair.long_size} + {clip_base} > {max_position_base})"
            )
        projected = total_open_notional + clip_base * mark
        if projected > self.cfg.max_open_notional_usd:
            return False, (
                f"max_open_notional_usd reached ({projected:.0f} > {self.cfg.max_open_notional_usd})"
            )
        return True, "ok"

    def unhedged_notional_breach(self, pair: PairPosition, mark: Decimal) -> Optional[str]:
        """The size half of the unhedged limit, without the clock.

        Delayed-hedge mode deliberately sits unhedged for seconds at a time, so
        the time limit there is the volatility window rather than
        ``unhedged_timeout_ms``. The notional ceiling still applies, and this is
        how that half is checked on its own.
        """
        unhedged = pair.unhedged
        if unhedged <= ZERO:
            return None
        notional = unhedged * mark
        if notional > self.cfg.max_unhedged_notional_usd:
            return (
                f"unhedged {unhedged} ({notional:.0f} USD) over limit "
                f"{self.cfg.max_unhedged_notional_usd}"
            )
        return None

    def unhedged_breach(
        self, pair: PairPosition, mark: Decimal, *, now: Optional[float] = None
    ) -> Optional[str]:
        """Return a reason string when the unhedged leg needs forcing shut."""
        now = now or time.time()
        key = pair.symbol
        unhedged = pair.unhedged
        if unhedged <= ZERO:
            self._unhedged_since.pop(key, None)
            return None
        since = self._unhedged_since.setdefault(key, now)
        notional = unhedged * mark
        if notional > self.cfg.max_unhedged_notional_usd:
            return (
                f"unhedged {unhedged} ({notional:.0f} USD) over limit "
                f"{self.cfg.max_unhedged_notional_usd}"
            )
        elapsed_ms = (now - since) * 1_000
        if elapsed_ms > self.cfg.unhedged_timeout_ms:
            return f"unhedged {unhedged} for {elapsed_ms:.0f}ms over timeout {self.cfg.unhedged_timeout_ms}ms"
        return None

    def clear_unhedged(self, symbol: str) -> None:
        self._unhedged_since.pop(symbol, None)
