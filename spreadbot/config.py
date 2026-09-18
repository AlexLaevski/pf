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
    # A multi-level wall has to be an actual wall: levels tens of bps apart are
    # not one block of liquidity, they are separate levels that happen to add
    # up. Ignored when min_wall_levels is 1.
    max_wall_span_bps: Decimal = Decimal(10)
    # A candidate at the touch is not the trade this bot is built on: quoting a
    # tick above the best bid makes us the top of book, alone, with no wall
    # underneath to stop the sweep that fills us.
    allow_touch_improving: bool = False
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
            max_wall_span_bps=_dec(d.get("max_wall_span_bps", 10), "gap.max_wall_span_bps"),
            allow_touch_improving=bool(d.get("allow_touch_improving", False)),
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
class HedgeConfig:
    """When the hedge leg fires, and how hard it pushes.

    ``immediate`` hedges the moment the maker leg fills — delta-neutral from
    the first millisecond, capturing only the cross-venue spread.

    ``delayed`` is the scalp: after a gap fill the bot sits in the naked long
    for a volatility-scaled window, hoping the sweep mean-reverts and the
    bounce exit fills for a much bigger gain. The hedge is the fallback when
    the bounce does not come.

    In ``delayed`` mode ``panic_bps`` is what keeps the naked window bounded:
    the timer only runs while the trade is roughly where we left it. Once the
    hedge-able price has fallen that far below our entry, the window is over
    regardless of how much time is left.
    """

    mode: str = "immediate"
    base_seconds: Decimal = Decimal(30)
    min_seconds: Decimal = Decimal(8)
    max_seconds: Decimal = Decimal(45)
    reference_vol_bps_per_min: Decimal = Decimal(25)
    vol_window_seconds: Decimal = Decimal(180)
    panic_bps: Decimal = Decimal(50)
    # How far below our entry the hedge price may already be when we enter.
    # Zero means the backstop must be at least break-even at the moment of the
    # fill: entering when the hedge venue is already lower is entering a trade
    # whose panic exit is guaranteed to be a loss.
    max_entry_adverse_bps: Decimal = Decimal(0)
    maker_first: bool = False
    maker_timeout_ms: int = 1_200
    maker_offset_ticks: int = 0

    @property
    def is_delayed(self) -> bool:
        return self.mode == "delayed"

    @classmethod
    def parse(cls, data: Dict[str, Any]) -> "HedgeConfig":
        d = data or {}
        mode = str(d.get("mode", "immediate")).lower()
        if mode not in ("immediate", "delayed"):
            raise ConfigError("hedge.mode must be 'immediate' or 'delayed'")
        cfg = cls(
            mode=mode,
            base_seconds=_dec(d.get("base_seconds", 30), "hedge.base_seconds"),
            min_seconds=_dec(d.get("min_seconds", 8), "hedge.min_seconds"),
            max_seconds=_dec(d.get("max_seconds", 45), "hedge.max_seconds"),
            reference_vol_bps_per_min=_dec(
                d.get("reference_vol_bps_per_min", 25), "hedge.reference_vol_bps_per_min"
            ),
            vol_window_seconds=_dec(d.get("vol_window_seconds", 180), "hedge.vol_window_seconds"),
            panic_bps=_dec(d.get("panic_bps", 50), "hedge.panic_bps"),
            max_entry_adverse_bps=_dec(
                d.get("max_entry_adverse_bps", 0), "hedge.max_entry_adverse_bps"
            ),
            # Maker-first is the natural pairing for the delayed hedge, where a
            # second of queue time is cheap. In immediate mode the whole point
            # is to lock the spread at once, so there it stays off by default.
            maker_first=bool(d.get("maker_first", mode == "delayed")),
            maker_timeout_ms=int(d.get("maker_timeout_ms", 1_200)),
            maker_offset_ticks=int(d.get("maker_offset_ticks", 0)),
        )
        if cfg.min_seconds > cfg.max_seconds:
            raise ConfigError("hedge.min_seconds must not exceed hedge.max_seconds")
        if cfg.min_seconds < 0:
            raise ConfigError("hedge.min_seconds must be >= 0")
        if cfg.reference_vol_bps_per_min <= 0:
            raise ConfigError("hedge.reference_vol_bps_per_min must be > 0")
        if cfg.is_delayed and cfg.panic_bps <= 0:
            raise ConfigError(
                "hedge.panic_bps must be > 0 in delayed mode: it is the only bound "
                "on how far the naked leg can run against you"
            )
        if cfg.max_entry_adverse_bps >= cfg.panic_bps and cfg.is_delayed:
            raise ConfigError(
                "hedge.max_entry_adverse_bps must be below hedge.panic_bps, otherwise "
                "an entry is allowed that panics out on its very first tick"
            )
        return cfg


@dataclass(frozen=True)
class BounceConfig:
    """Where the take-profit sits while we hold the naked long.

    ``hole_top`` aims back at the near edge of the gap we were filled through —
    that is the level the sweep came from, so it is the natural target for a
    mean reversion. ``fixed_bps`` is a flat markup over the entry instead.
    """

    target: str = "hole_top"
    target_bps: Decimal = Decimal(20)
    min_target_bps: Decimal = Decimal(3)

    @classmethod
    def parse(cls, data: Dict[str, Any]) -> "BounceConfig":
        d = data or {}
        target = str(d.get("target", "hole_top")).lower()
        if target not in ("hole_top", "fixed_bps"):
            raise ConfigError("bounce.target must be 'hole_top' or 'fixed_bps'")
        return cls(
            target=target,
            target_bps=_dec(d.get("target_bps", 20), "bounce.target_bps"),
            min_target_bps=_dec(d.get("min_target_bps", 3), "bounce.min_target_bps"),
        )


@dataclass(frozen=True)
class RotationConfig:
    """Rotate the watched markets toward wherever the gaps currently are.

    Time in market is what binds this strategy: a quote can only be hit while
    it is resting, and a given market offers somewhere to rest only while its
    book has a hole. Watching a fixed handful means sitting out most of the
    day. Rotation trades a periodic scan for being in the book far more often.
    """

    enabled: bool = False
    interval_seconds: int = 900
    watch: int = 12
    max_churn: int = 4                     # markets swapped per round, to limit thrash
    clip_usd: Decimal = Decimal(50)
    min_volume_usd: Decimal = Decimal(50_000)
    max_volume_usd: Decimal = Decimal(500_000_000)
    scan_delay_ms: int = 120               # spacing between book requests while scanning

    @classmethod
    def parse(cls, data: Dict[str, Any]) -> "RotationConfig":
        d = data or {}
        cfg = cls(
            enabled=bool(d.get("enabled", False)),
            interval_seconds=int(d.get("interval_seconds", 900)),
            watch=int(d.get("watch", 12)),
            max_churn=int(d.get("max_churn", 4)),
            clip_usd=_dec(d.get("clip_usd", 50), "rotation.clip_usd"),
            min_volume_usd=_dec(d.get("min_volume_usd", 50_000), "rotation.min_volume_usd"),
            max_volume_usd=_dec(d.get("max_volume_usd", 500_000_000), "rotation.max_volume_usd"),
            scan_delay_ms=int(d.get("scan_delay_ms", 120)),
        )
        if cfg.watch < 1:
            raise ConfigError("rotation.watch must be >= 1")
        if cfg.clip_usd <= 0:
            raise ConfigError("rotation.clip_usd must be > 0")
        return cfg


@dataclass(frozen=True)
class FundingConfig:
    """Funding is a real cost and on some markets it is the whole edge.

    We are long on the maker venue and short on the hedge venue, so funding
    nets: we pay the maker venue's rate and receive the hedge venue's. On a
    single hot token the two can be far apart, and at 6 bps/hour a fifteen
    minute hold quietly eats 1.5 bps of a 10 bps trade.
    """

    enabled: bool = True
    expected_hold_minutes: Decimal = Decimal(10)
    max_cost_bps: Decimal = Decimal(5)
    refresh_seconds: int = 300

    @property
    def expected_hold_hours(self) -> Decimal:
        return self.expected_hold_minutes / Decimal(60)

    @classmethod
    def parse(cls, data: Dict[str, Any]) -> "FundingConfig":
        d = data or {}
        cfg = cls(
            enabled=bool(d.get("enabled", True)),
            expected_hold_minutes=_dec(
                d.get("expected_hold_minutes", 10), "funding.expected_hold_minutes"
            ),
            max_cost_bps=_dec(d.get("max_cost_bps", 5), "funding.max_cost_bps"),
            refresh_seconds=int(d.get("refresh_seconds", 300)),
        )
        if cfg.expected_hold_minutes < 0:
            raise ConfigError("funding.expected_hold_minutes must be >= 0")
        return cfg


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
    hedge: HedgeConfig = field(default_factory=HedgeConfig)
    bounce: BounceConfig = field(default_factory=BounceConfig)
    funding: FundingConfig = field(default_factory=FundingConfig)
    rotation: RotationConfig = field(default_factory=RotationConfig)
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

    @property
    def symbols(self) -> List[str]:
        return [market.symbol for market in self.markets]

    def set_markets(self, markets: List[MarketConfig]) -> None:
        """Replace the watched market set at runtime (used by rotation).

        The list object is mutated in place rather than rebound, so the
        strategy and anything else holding this config sees the new set
        immediately and there is only ever one source of truth.
        """
        self.markets[:] = markets


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
        hedge=HedgeConfig.parse(raw.get("hedge") or {}),
        bounce=BounceConfig.parse(raw.get("bounce") or {}),
        funding=FundingConfig.parse(raw.get("funding") or {}),
        rotation=RotationConfig.parse(raw.get("rotation") or {}),
        risk=RiskConfig.parse(raw.get("risk") or {}),
        unwind=UnwindConfig.parse(raw.get("unwind") or {}),
        log_level=str(raw.get("log_level", "INFO")).upper(),
    )
