"""Configuration: YAML in, validated dataclasses out.

Secrets never live in the YAML file. A venue names an environment variable
(``private_key_env``) and the key is read from the process environment at
start-up. Any string value may also use ``${VAR}`` expansion.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .models import Side

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(ValueError):
    """Raised for anything wrong in the config file."""


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match) -> str:
            name = match.group(1)
            if name not in os.environ:
                raise ConfigError(f"config references ${{{name}}} but it is not set in the environment")
            return os.environ[name]

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _dec(raw: Any, field_name: str) -> Decimal:
    try:
        return Decimal(str(raw))
    except (InvalidOperation, TypeError) as exc:  # pragma: no cover - defensive
        raise ConfigError(f"{field_name}: {raw!r} is not a number") from exc


def _req(data: Dict[str, Any], key: str, where: str) -> Any:
    if key not in data or data[key] is None:
        raise ConfigError(f"{where}: missing required key {key!r}")
    return data[key]


@dataclass(frozen=True)
class VenueConfig:
    """One Lighter deployment (Core, or the Robinhood domain)."""

    key: str
    name: str
    base_url: str
    ws_url: str
    allowed_side: Side
    account_index: Optional[int] = None
    api_key_index: int = 0
    private_key_env: Optional[str] = None
    maker_fee_bps: Optional[Decimal] = None
    taker_fee_bps: Optional[Decimal] = None

    @property
    def private_key(self) -> Optional[str]:
        if not self.private_key_env:
            return None
        return os.environ.get(self.private_key_env)

    def require_credentials(self) -> str:
        if self.account_index is None:
            raise ConfigError(f"venue {self.key}: account_index is required for live trading")
        key = self.private_key
        if not key:
            raise ConfigError(
                f"venue {self.key}: no API key found; set {self.private_key_env or '<private_key_env>'}"
            )
        return key

    @classmethod
    def parse(cls, key: str, data: Dict[str, Any]) -> "VenueConfig":
        where = f"venues.{key}"
        base_url = str(_req(data, "base_url", where)).rstrip("/")
        ws_url = data.get("ws_url") or _default_ws_url(base_url)
        side_raw = str(_req(data, "allowed_side", where)).lower()
        if side_raw not in ("buy", "sell"):
            raise ConfigError(f"{where}.allowed_side must be 'buy' or 'sell', got {side_raw!r}")
        if "://" not in base_url:
            raise ConfigError(f"{where}.base_url must be an absolute URL, got {base_url!r}")
        if "PLACEHOLDER" in base_url.upper() or "fill-me" in base_url:
            raise ConfigError(
                f"{where}.base_url is still a placeholder - point it at the real API host"
            )
        return cls(
            key=key,
            name=str(data.get("name", key)),
            base_url=base_url,
            ws_url=str(ws_url),
            allowed_side=Side(side_raw),
            account_index=int(data["account_index"]) if data.get("account_index") is not None else None,
            api_key_index=int(data.get("api_key_index", 0)),
            private_key_env=data.get("private_key_env"),
            maker_fee_bps=_dec(data["maker_fee_bps"], f"{where}.maker_fee_bps")
            if data.get("maker_fee_bps") is not None
            else None,
            taker_fee_bps=_dec(data["taker_fee_bps"], f"{where}.taker_fee_bps")
            if data.get("taker_fee_bps") is not None
            else None,
        )


def _default_ws_url(base_url: str) -> str:
    return base_url.replace("https://", "wss://").replace("http://", "ws://") + "/stream"


@dataclass(frozen=True)
class FeedConfig:
    transport: str = "ws"           # "ws" (streaming) or "rest" (polling fallback)
    rest_poll_ms: int = 400
    depth_levels: int = 50
    staleness_ms: int = 3_000       # a book older than this is not tradable
    reconnect_backoff_ms: int = 500
    reconnect_backoff_max_ms: int = 15_000

    @classmethod
    def parse(cls, data: Dict[str, Any]) -> "FeedConfig":
        transport = str(data.get("transport", "ws")).lower()
        if transport not in ("ws", "rest"):
            raise ConfigError("feed.transport must be 'ws' or 'rest'")
        return cls(
            transport=transport,
            rest_poll_ms=int(data.get("rest_poll_ms", 400)),
            depth_levels=int(data.get("depth_levels", 50)),
            staleness_ms=int(data.get("staleness_ms", 3_000)),
            reconnect_backoff_ms=int(data.get("reconnect_backoff_ms", 500)),
            reconnect_backoff_max_ms=int(data.get("reconnect_backoff_max_ms", 15_000)),
        )


@dataclass(frozen=True)
class GapConfig:
    """Parameters of the "thin book" detector.

    A level counts as a wall when its notional clears ``wall_notional_usd`` *and*
    it is at least ``wall_multiple`` times the typical level in the scanned
    window. The gap is the distance between the touch and that wall.
    """

    wall_notional_usd: Decimal = Decimal(20_000)
    wall_multiple: Decimal = Decimal(4)
    min_gap_bps: Decimal = Decimal("1.5")
    min_gap_ticks: int = 2
    max_levels_scan: int = 40
    join_offset_ticks: int = 1
    max_distance_from_mid_bps: Decimal = Decimal(50)
    min_wall_levels: int = 1
    # Liquidity resting at better prices than our quote. A sweep has to clear
    # all of it before it reaches us, so this is the fill-probability dial:
    # None means "don't care", which on a deep book means we rarely get hit.
    max_ahead_notional_usd: Optional[Decimal] = None

    @classmethod
    def parse(cls, data: Dict[str, Any]) -> "GapConfig":
        d = data or {}
        return cls(
            max_ahead_notional_usd=_dec(
                d["max_ahead_notional_usd"], "gap.max_ahead_notional_usd"
            )
            if d.get("max_ahead_notional_usd") is not None
            else None,
            wall_notional_usd=_dec(d.get("wall_notional_usd", 20_000), "gap.wall_notional_usd"),
            wall_multiple=_dec(d.get("wall_multiple", 4), "gap.wall_multiple"),
            min_gap_bps=_dec(d.get("min_gap_bps", "1.5"), "gap.min_gap_bps"),
            min_gap_ticks=int(d.get("min_gap_ticks", 2)),
            max_levels_scan=int(d.get("max_levels_scan", 40)),
            join_offset_ticks=int(d.get("join_offset_ticks", 1)),
            max_distance_from_mid_bps=_dec(
                d.get("max_distance_from_mid_bps", 50), "gap.max_distance_from_mid_bps"
            ),
            min_wall_levels=int(d.get("min_wall_levels", 1)),
        )


@dataclass(frozen=True)
class EdgeConfig:
    min_entry_bps: Decimal = Decimal(3)
    exit_bps: Decimal = Decimal("0.5")
    requote_ticks: int = 1
    requote_size_tol: Decimal = Decimal("0.25")
    hedge_is_taker: bool = True
    extra_buffer_bps: Decimal = Decimal("0.5")

    @classmethod
    def parse(cls, data: Dict[str, Any]) -> "EdgeConfig":
        d = data or {}
        return cls(
            min_entry_bps=_dec(d.get("min_entry_bps", 3), "edge.min_entry_bps"),
            exit_bps=_dec(d.get("exit_bps", "0.5"), "edge.exit_bps"),
            requote_ticks=int(d.get("requote_ticks", 1)),
            requote_size_tol=_dec(d.get("requote_size_tol", "0.25"), "edge.requote_size_tol"),
            hedge_is_taker=bool(d.get("hedge_is_taker", True)),
            extra_buffer_bps=_dec(d.get("extra_buffer_bps", "0.5"), "edge.extra_buffer_bps"),
        )


@dataclass(frozen=True)
class RiskConfig:
    max_open_notional_usd: Decimal = Decimal(5_000)
    max_unhedged_notional_usd: Decimal = Decimal(250)
    unhedged_timeout_ms: int = 2_000
    max_hedge_slippage_bps: Decimal = Decimal(8)
    max_daily_loss_usd: Decimal = Decimal(200)
    max_consecutive_errors: int = 5
    kill_switch_file: Optional[str] = None

    @classmethod
    def parse(cls, data: Dict[str, Any]) -> "RiskConfig":
        d = data or {}
        return cls(
            max_open_notional_usd=_dec(d.get("max_open_notional_usd", 5_000), "risk.max_open_notional_usd"),
            max_unhedged_notional_usd=_dec(
                d.get("max_unhedged_notional_usd", 250), "risk.max_unhedged_notional_usd"
            ),
            unhedged_timeout_ms=int(d.get("unhedged_timeout_ms", 2_000)),
            max_hedge_slippage_bps=_dec(
                d.get("max_hedge_slippage_bps", 8), "risk.max_hedge_slippage_bps"
            ),
            max_daily_loss_usd=_dec(d.get("max_daily_loss_usd", 200), "risk.max_daily_loss_usd"),
            max_consecutive_errors=int(d.get("max_consecutive_errors", 5)),
            kill_switch_file=d.get("kill_switch_file"),
        )


@dataclass(frozen=True)
class UnwindConfig:
    max_hold_seconds: int = 900
    stop_loss_bps: Decimal = Decimal(30)
    passive_exit: bool = True
    exit_chunk_fraction: Decimal = Decimal(1)

    @classmethod
    def parse(cls, data: Dict[str, Any]) -> "UnwindConfig":
        d = data or {}
        return cls(
            max_hold_seconds=int(d.get("max_hold_seconds", 900)),
            stop_loss_bps=_dec(d.get("stop_loss_bps", 30), "unwind.stop_loss_bps"),
            passive_exit=bool(d.get("passive_exit", True)),
            exit_chunk_fraction=_dec(d.get("exit_chunk_fraction", 1), "unwind.exit_chunk_fraction"),
        )


@dataclass(frozen=True)
class MarketConfig:
    symbol: str
    order_base: Decimal
    max_position_base: Decimal
    min_entry_bps: Optional[Decimal] = None

    @classmethod
    def parse(cls, data: Dict[str, Any]) -> "MarketConfig":
        symbol = str(_req(data, "symbol", "markets[]")).upper()
        where = f"markets.{symbol}"
        order_base = _dec(_req(data, "order_base", where), f"{where}.order_base")
        max_position = _dec(
            data.get("max_position_base", order_base * 5), f"{where}.max_position_base"
        )
        if order_base <= 0:
            raise ConfigError(f"{where}.order_base must be > 0")
        if max_position < order_base:
            raise ConfigError(f"{where}.max_position_base must be >= order_base")
        return cls(
            symbol=symbol,
            order_base=order_base,
            max_position_base=max_position,
            min_entry_bps=_dec(data["min_entry_bps"], f"{where}.min_entry_bps")
            if data.get("min_entry_bps") is not None
            else None,
        )


@dataclass(frozen=True)
class Config:
    mode: str
    maker_venue: VenueConfig
    hedge_venue: VenueConfig
    markets: List[MarketConfig]
    feed: FeedConfig = field(default_factory=FeedConfig)
    gap: GapConfig = field(default_factory=GapConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    unwind: UnwindConfig = field(default_factory=UnwindConfig)
    log_level: str = "INFO"

    @property
    def is_live(self) -> bool:
        return self.mode == "live"

    @property
    def venues(self) -> Dict[str, VenueConfig]:
        return {self.maker_venue.key: self.maker_venue, self.hedge_venue.key: self.hedge_venue}

    def market(self, symbol: str) -> MarketConfig:
        for market in self.markets:
            if market.symbol == symbol:
                return market
        raise KeyError(symbol)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    return parse_config(raw)


def parse_config(raw: Dict[str, Any]) -> Config:
    raw = _expand(raw)

    mode = str(raw.get("mode", "paper")).lower()
    if mode not in ("paper", "live"):
        raise ConfigError("mode must be 'paper' or 'live'")

    venues_raw = _req(raw, "venues", "config")
    if not isinstance(venues_raw, dict) or len(venues_raw) < 2:
        raise ConfigError("config.venues must define at least two venues")

    roles = _req(raw, "roles", "config")
    maker_key = str(_req(roles, "maker", "roles"))
    hedge_key = str(_req(roles, "hedge", "roles"))
    if maker_key == hedge_key:
        raise ConfigError("roles.maker and roles.hedge must be different venues")
    for key in (maker_key, hedge_key):
        if key not in venues_raw:
            raise ConfigError(f"roles references unknown venue {key!r}")

    maker = VenueConfig.parse(maker_key, venues_raw[maker_key])
    hedge = VenueConfig.parse(hedge_key, venues_raw[hedge_key])

    # The directional mandate: the maker venue may only go long, the hedge venue
    # may only go short. Everything downstream relies on this invariant.
    if maker.allowed_side is not Side.BUY:
        raise ConfigError(
            f"maker venue {maker.key!r} must have allowed_side: buy (long-only entry leg)"
        )
    if hedge.allowed_side is not Side.SELL:
        raise ConfigError(
            f"hedge venue {hedge.key!r} must have allowed_side: sell (short-only hedge leg)"
        )

    markets_raw = _req(raw, "markets", "config")
    if not isinstance(markets_raw, list) or not markets_raw:
        raise ConfigError("config.markets must be a non-empty list")
    markets = [MarketConfig.parse(m) for m in markets_raw]
    seen = set()
    for market in markets:
        if market.symbol in seen:
            raise ConfigError(f"duplicate market {market.symbol}")
        seen.add(market.symbol)

    return Config(
        mode=mode,
        maker_venue=maker,
        hedge_venue=hedge,
        markets=markets,
        feed=FeedConfig.parse(raw.get("feed") or {}),
        gap=GapConfig.parse(raw.get("gap") or {}),
        edge=EdgeConfig.parse(raw.get("edge") or {}),
        risk=RiskConfig.parse(raw.get("risk") or {}),
        unwind=UnwindConfig.parse(raw.get("unwind") or {}),
        log_level=str(raw.get("log_level", "INFO")).upper(),
    )
