import pytest

from spreadbot.config import ConfigError, parse_config
from spreadbot.models import Side

from .conftest import raw_config


def test_parses_roles_and_mandate():
    cfg = parse_config(raw_config())
    assert cfg.maker_venue.key == "rh"
    assert cfg.maker_venue.allowed_side is Side.BUY
    assert cfg.hedge_venue.allowed_side is Side.SELL
    assert cfg.maker_venue.ws_url == "wss://rh.example/stream"


def test_maker_must_be_long_only():
    raw = raw_config()
    raw["venues"]["rh"]["allowed_side"] = "sell"
    with pytest.raises(ConfigError, match="allowed_side: buy"):
        parse_config(raw)


def test_hedge_must_be_short_only():
    raw = raw_config()
    raw["venues"]["core"]["allowed_side"] = "buy"
    with pytest.raises(ConfigError, match="allowed_side: sell"):
        parse_config(raw)


def test_roles_must_differ():
    raw = raw_config()
    raw["roles"] = {"maker": "rh", "hedge": "rh"}
    with pytest.raises(ConfigError, match="must be different"):
        parse_config(raw)


def test_placeholder_url_is_rejected():
    raw = raw_config()
    raw["venues"]["rh"]["base_url"] = "https://REPLACE-WITH-ROBINHOOD-LIGHTER-API-HOST"
    # The example file ships a placeholder; running with it must fail loudly.
    raw["venues"]["rh"]["base_url"] = "https://PLACEHOLDER.example"
    with pytest.raises(ConfigError, match="placeholder"):
        parse_config(raw)


def test_duplicate_markets_are_rejected():
    raw = raw_config()
    raw["markets"] = [
        {"symbol": "BTC", "order_base": "0.01"},
        {"symbol": "BTC", "order_base": "0.02"},
    ]
    with pytest.raises(ConfigError, match="duplicate market"):
        parse_config(raw)


def test_max_position_must_cover_a_clip():
    raw = raw_config()
    raw["markets"] = [{"symbol": "BTC", "order_base": "0.1", "max_position_base": "0.01"}]
    with pytest.raises(ConfigError, match="max_position_base"):
        parse_config(raw)


def test_env_expansion(monkeypatch):
    monkeypatch.setenv("TEST_HOST", "core.example")
    raw = raw_config()
    raw["venues"]["core"]["base_url"] = "https://${TEST_HOST}"
    cfg = parse_config(raw)
    assert cfg.hedge_venue.base_url == "https://core.example"


def test_missing_env_var_is_an_error():
    raw = raw_config()
    raw["venues"]["core"]["base_url"] = "https://${DEFINITELY_NOT_SET_12345}"
    with pytest.raises(ConfigError, match="not set in the environment"):
        parse_config(raw)
