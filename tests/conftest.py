from __future__ import annotations

from decimal import Decimal
from typing import Dict, Sequence, Tuple

import pytest

from spreadbot.book import OrderBook
from spreadbot.config import Config, parse_config
from spreadbot.models import MarketSpec

D = Decimal


def spec(
    symbol: str = "BTC",
    *,
    market_id: int = 1,
    price_decimals: int = 1,
    size_decimals: int = 5,
    min_base: str = "0.00007",
    min_quote: str = "0",
    maker_fee_bps: str = "0",
    taker_fee_bps: str = "0",
) -> MarketSpec:
    return MarketSpec(
        symbol=symbol,
        market_id=market_id,
        price_decimals=price_decimals,
        size_decimals=size_decimals,
        min_base_amount=D(min_base),
        min_quote_amount=D(min_quote),
        maker_fee_bps=D(maker_fee_bps),
        taker_fee_bps=D(taker_fee_bps),
    )


def book(
    bids: Sequence[Tuple[str, str]],
    asks: Sequence[Tuple[str, str]],
    *,
    venue: str = "v",
    symbol: str = "BTC",
) -> OrderBook:
    ob = OrderBook(venue, symbol)
    ob.apply_snapshot([(D(p), D(s)) for p, s in bids], [(D(p), D(s)) for p, s in asks])
    return ob


def raw_config(**overrides) -> Dict:
    base = {
        "mode": "paper",
        "venues": {
            "rh": {
                "base_url": "https://rh.example",
                "allowed_side": "buy",
                "account_index": 1,
                "private_key_env": "RH_KEY",
            },
            "core": {
                "base_url": "https://core.example",
                "allowed_side": "sell",
                "account_index": 2,
                "private_key_env": "CORE_KEY",
            },
        },
        "roles": {"maker": "rh", "hedge": "core"},
        "markets": [{"symbol": "BTC", "order_base": "0.01", "max_position_base": "0.05"}],
        "gap": {
            "wall_notional_usd": 300,
            "wall_multiple": 4,
            "min_gap_bps": "0.5",
            "min_gap_ticks": 2,
            "join_offset_ticks": 1,
            "max_distance_from_mid_bps": 200,
        },
        "edge": {"min_entry_bps": 3, "extra_buffer_bps": 0, "exit_bps": "0.5"},
        "risk": {"max_open_notional_usd": 100000, "max_unhedged_notional_usd": 1000},
    }
    base.update(overrides)
    return base


@pytest.fixture
def cfg() -> Config:
    return parse_config(raw_config())
