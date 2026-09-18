"""Rotation moves the watched set toward wherever the gaps are. The rule that
matters is the one about not walking away from a market mid-trade."""

from decimal import Decimal

from spreadbot.config import parse_config
from spreadbot.engine import Engine, MarketState
from spreadbot.models import MarketSpec, PairPosition, Side
from spreadbot.venues.feed import StaticFeed
from spreadbot.venues.paper import PaperExecution

from .conftest import raw_config

D = Decimal

HOLEY = ([("100.0", "0.01"), ("99.9", "0.01"), ("99.8", "0.01"), ("99.5", "5"), ("99.4", "5")],
         [("100.1", "1"), ("100.2", "1")])
DENSE = ([(str(D("100.0") - D("0.1") * i), "5") for i in range(10)], [("100.1", "1")])
HEDGE = ([("100.0", "10")], [("100.05", "10")])

UNIVERSE = ["AAA", "BBB", "CCC", "DDD"]


class _FakeRest:
    """Serves whatever book each symbol has been assigned."""

    def __init__(self, books):
        self.books = books

    async def order_book_levels(self, market_id, limit=250):
        symbol = UNIVERSE[market_id]
        bids, asks = self.books[symbol]
        return ([(D(p), D(s)) for p, s in bids], [(D(p), D(s)) for p, s in asks])

    async def funding_rates(self, exchange="lighter"):
        return {}


def build(watched, books, **rotation):
    rot = {"enabled": True, "watch": 2, "max_churn": 4, "clip_usd": 50, "scan_delay_ms": 0}
    rot.update(rotation)
    cfg = parse_config(
        raw_config(
            markets=[
                {"symbol": s, "order_base": "0.01", "max_position_base": "0.05"} for s in watched
            ],
            rotation=rot,
        )
    )
    specs = {
        s: MarketSpec(s, i, 1, 5, D("0.00007"), D(0)) for i, s in enumerate(UNIVERSE)
    }
    maker_feed, hedge_feed = StaticFeed("rh", specs), StaticFeed("core", specs)
    for symbol in watched:
        maker_feed.set_book(symbol, *books[symbol])
        hedge_feed.set_book(symbol, *HEDGE)
    return Engine(
        cfg,
        maker_feed=maker_feed,
        hedge_feed=hedge_feed,
        maker_exec=PaperExecution("rh", Side.BUY, specs, maker_feed),
        hedge_exec=PaperExecution("core", Side.SELL, specs, hedge_feed),
        maker_specs=specs,
        hedge_specs=specs,
        maker_rest=_FakeRest(books),
        universe=UNIVERSE,
        volumes={s: 1_000_000.0 for s in UNIVERSE},
        marks={s: D(100) for s in UNIVERSE},
    )


async def test_rotates_toward_markets_with_gaps():
    books = {"AAA": DENSE, "BBB": DENSE, "CCC": HOLEY, "DDD": HOLEY}
    engine = build(["AAA", "BBB"], books)
    await engine._rotate_markets()
    assert set(engine.cfg.symbols) == {"CCC", "DDD"}
    assert set(engine.states) == {"CCC", "DDD"}
    assert set(engine.maker_feed.books) == {"CCC", "DDD"}


async def test_a_market_holding_a_position_is_never_dropped():
    books = {"AAA": DENSE, "BBB": DENSE, "CCC": HOLEY, "DDD": HOLEY}
    engine = build(["AAA", "BBB"], books)
    engine.states["AAA"].pair.add_long(D(100), D("0.01"))
    await engine._rotate_markets()
    assert "AAA" in engine.cfg.symbols
    assert "AAA" in engine.maker_feed.books
    assert engine.states["AAA"].pair.long_size == D("0.01")


async def test_a_market_with_a_resting_quote_is_never_dropped():
    books = {"AAA": HOLEY, "BBB": DENSE, "CCC": HOLEY, "DDD": HOLEY}
    engine = build(["AAA", "BBB"], books)
    await engine._tick()
    assert engine.states["AAA"].quote_order is not None
    await engine._rotate_markets()
    assert "AAA" in engine.cfg.symbols


async def test_dropped_markets_have_their_orders_cancelled_first():
    books = {"AAA": HOLEY, "BBB": HOLEY, "CCC": HOLEY, "DDD": HOLEY}
    engine = build(["AAA", "BBB"], books)
    await engine._tick()
    resting = engine.states["AAA"].quote_order
    assert resting is not None
    # Force AAA out by pretending it is no longer worth watching.
    engine.states["AAA"].quote_order = None
    engine.states["AAA"].pair = PairPosition("AAA", "rh", "core")
    books["AAA"] = DENSE
    engine.maker_feed.set_book("AAA", *DENSE)
    await engine._rotate_markets()
    if "AAA" not in engine.cfg.symbols:
        assert engine.maker_exec.resting("AAA") == {}


async def test_watch_count_is_respected():
    books = {s: HOLEY for s in UNIVERSE}
    engine = build(["AAA"], books, watch=3)
    await engine._rotate_markets()
    assert len(engine.cfg.symbols) == 3


async def test_no_change_when_the_set_is_already_the_best():
    books = {"AAA": HOLEY, "BBB": HOLEY, "CCC": DENSE, "DDD": DENSE}
    engine = build(["AAA", "BBB"], books)
    await engine._rotate_markets()
    assert set(engine.cfg.symbols) == {"AAA", "BBB"}


async def test_new_markets_get_a_state_and_a_clip_size():
    books = {"AAA": DENSE, "BBB": DENSE, "CCC": HOLEY, "DDD": HOLEY}
    engine = build(["AAA", "BBB"], books)
    await engine._rotate_markets()
    market = engine.cfg.market("CCC")
    assert market.order_base > 0
    assert market.order_base * D(100) <= D(150)      # ~clip_usd, within tolerance
    assert isinstance(engine.states["CCC"], MarketState)


async def test_rotation_survives_a_dead_universe():
    books = {s: DENSE for s in UNIVERSE}
    engine = build(["AAA", "BBB"], books)
    await engine._rotate_markets()
    # Nothing qualifies anywhere; the bot keeps what it had rather than going blind.
    assert engine.cfg.symbols
