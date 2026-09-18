"""Command line entry points.

    spreadbot validate -c config/config.yaml    check the config, touch nothing
    spreadbot markets  --url <venue>            list markets on a venue
    spreadbot gaps     --url <venue> -s BTC     watch one venue's book for holes
    spreadbot scan     -c config/config.yaml    read-only: gaps + hedge + edge
    spreadbot run      -c config/config.yaml    trade (paper or live)

``gaps`` and ``markets`` need nothing but a public API URL, so they are the way
to sanity-check a venue before any keys exist.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import logging
import signal
import sys
import time
from decimal import Decimal
from typing import List, Optional, Sequence

from .book import OrderBook
from .config import ConfigError, FeedConfig, GapConfig, load_config
from .gaps import book_thinness_bps, find_wall_candidates
from .logging_setup import setup_logging
from .models import Side
from .strategy import SpreadStrategy
from .venues.feed import LighterFeed
from .venues.rest import LighterRest

log = logging.getLogger("spreadbot.cli")


# --------------------------------------------------------------------- helpers


def _fmt(value: Optional[Decimal], places: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{places}f}"


async def _feed_for(
    url: str, symbols: Sequence[str], *, transport: str = "ws", venue_key: str = "venue"
):
    rest = LighterRest(url, venue=venue_key)
    specs = await rest.market_specs()
    unknown = [s for s in symbols if s.upper() not in specs]
    if unknown:
        await rest.close()
        raise SystemExit(f"{venue_key}: unknown markets {', '.join(unknown)}")
    feed = LighterFeed(
        venue_key,
        ws_url=url.replace("https://", "wss://").replace("http://", "ws://") + "/stream",
        rest=rest,
        config=FeedConfig(transport=transport),
        specs=specs,
    )
    await feed.start([s.upper() for s in symbols])
    return rest, specs, feed


# -------------------------------------------------------------------- commands


async def cmd_markets(args: argparse.Namespace) -> int:
    async with LighterRest(args.url) as rest:
        specs = await rest.market_specs()
        marks = await rest.mark_prices()
    print(f"{'symbol':<12}{'id':>5}{'tick':>14}{'lot':>14}{'min base':>14}{'mark':>14}")
    for symbol in sorted(specs):
        spec = specs[symbol]
        print(
            f"{symbol:<12}{spec.market_id:>5}{str(spec.tick):>14}{str(spec.lot):>14}"
            f"{str(spec.min_base_amount):>14}{str(marks.get(symbol, '-')):>14}"
        )
    print(f"\n{len(specs)} active markets on {args.url}")
    return 0


async def cmd_gaps(args: argparse.Namespace) -> int:
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    gap_cfg = GapConfig(
        wall_notional_usd=Decimal(str(args.wall_usd)),
        wall_multiple=Decimal(str(args.wall_multiple)),
        min_gap_bps=Decimal(str(args.min_gap_bps)),
        min_gap_ticks=args.min_gap_ticks,
        max_distance_from_mid_bps=Decimal(str(args.max_distance_bps)),
        max_ahead_notional_usd=(
            Decimal(str(args.max_ahead_usd)) if args.max_ahead_usd is not None else None
        ),
    )
    rest, specs, feed = await _feed_for(
        args.url, symbols, transport=args.transport, venue_key="venue"
    )
    try:
        if not await feed.wait_ready(20):
            print("no order book snapshot received", file=sys.stderr)
            return 1
        deadline = time.time() + args.duration
        while time.time() < deadline:
            for symbol in symbols:
                book = feed.book(symbol)
                if not book.ready:
                    continue
                _print_gaps(book, specs[symbol], gap_cfg)
            if args.once:
                break
            await feed.wait_for_update(args.interval)
            await asyncio.sleep(max(0.0, args.interval))
    finally:
        await feed.close()
        await rest.close()
    return 0


def _print_gaps(book: OrderBook, spec, gap_cfg: GapConfig) -> None:
    mid = book.mid
    thin_bid = book_thinness_bps(book, Side.BUY, Decimal(50_000))
    thin_ask = book_thinness_bps(book, Side.SELL, Decimal(50_000))
    print(
        f"\n[{time.strftime('%H:%M:%S')}] {book.symbol} mid={_fmt(mid, spec.price_decimals)} "
        f"spread={_fmt(book.spread_bps)}bps "
        f"depth50k={_fmt(thin_bid)}/{_fmt(thin_ask)}bps"
    )
    for side in (Side.BUY, Side.SELL):
        candidates = find_wall_candidates(book, side, spec, gap_cfg)
        label = "bid" if side is Side.BUY else "ask"
        if not candidates:
            print(f"   {label}: no wall with a gap in front of it")
            continue
        for candidate in candidates[:3]:
            print(
                f"   {label}: quote {candidate.price} in front of wall {candidate.wall_price} "
                f"({candidate.wall_notional:,.0f} USD, lvl {candidate.level_index}) "
                f"gap {_fmt(candidate.hole_bps)}bps / {_fmt(candidate.hole_ticks, 0)} ticks, "
                f"{_fmt(candidate.distance_bps)}bps from mid, "
                f"{candidate.ahead_notional:,.0f} USD queued ahead"
            )


async def cmd_scan(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    setup_logging(cfg.log_level, args.logfile)
    symbols = [m.symbol for m in cfg.markets]
    strategy = SpreadStrategy(cfg)

    maker_rest, maker_specs, maker_feed = await _feed_for(
        cfg.maker_venue.base_url, symbols, transport=cfg.feed.transport, venue_key=cfg.maker_venue.key
    )
    hedge_rest, hedge_specs, hedge_feed = await _feed_for(
        cfg.hedge_venue.base_url, symbols, transport=cfg.feed.transport, venue_key=cfg.hedge_venue.key
    )
    try:
        if not (await maker_feed.wait_ready(20) and await hedge_feed.wait_ready(20)):
            print("order books did not arrive", file=sys.stderr)
            return 1
        deadline = time.time() + args.duration
        while time.time() < deadline:
            print(f"\n=== {time.strftime('%H:%M:%S')} ===")
            for symbol in symbols:
                maker_book = maker_feed.book(symbol)
                hedge_book = hedge_feed.book(symbol)
                market = cfg.market(symbol)
                decision = strategy.plan_entry(
                    symbol,
                    maker_book,
                    hedge_book,
                    maker_specs[symbol],
                    hedge_specs[symbol],
                    capacity_base=market.max_position_base,
                )
                head = (
                    f"{symbol:<8} "
                    f"{cfg.maker_venue.key} {maker_book.best_bid}/{maker_book.best_ask}  "
                    f"{cfg.hedge_venue.key} {hedge_book.best_bid}/{hedge_book.best_ask}"
                )
                if decision.ok and decision.plan is not None:
                    plan = decision.plan
                    print(
                        f"{head}\n   BUY {plan.size} @ {plan.quote.price} "
                        f"(wall {plan.candidate.wall_price}, gap {_fmt(plan.candidate.hole_bps)}bps) "
                        f"-> hedge SELL @ {plan.hedge_price} | EDGE {_fmt(plan.edge_bps)}bps"
                    )
                else:
                    print(
                        f"{head}\n   no trade: {decision.reason}"
                        + (
                            f" (best {_fmt(decision.best_edge_bps)}bps, "
                            f"{decision.candidates} candidates)"
                            if decision.best_edge_bps is not None
                            else ""
                        )
                    )
            if args.once:
                break
            await asyncio.sleep(args.interval)
    finally:
        for closer in (maker_feed, hedge_feed, maker_rest, hedge_rest):
            with contextlib.suppress(Exception):
                await closer.close()
    return 0


async def cmd_run(args: argparse.Namespace) -> int:
    from .engine import build_engine

    cfg = load_config(args.config)
    if args.paper:
        cfg = dataclasses.replace(cfg, mode="paper")
    setup_logging(cfg.log_level, args.logfile)

    if cfg.is_live:
        cfg.maker_venue.require_credentials()
        cfg.hedge_venue.require_credentials()
        if not args.yes:
            print(
                "About to trade with REAL money:\n"
                f"  maker (long only) : {cfg.maker_venue.key} {cfg.maker_venue.base_url} "
                f"account {cfg.maker_venue.account_index}\n"
                f"  hedge (short only): {cfg.hedge_venue.key} {cfg.hedge_venue.base_url} "
                f"account {cfg.hedge_venue.account_index}\n"
                f"  markets           : {', '.join(m.symbol for m in cfg.markets)}\n"
                f"  max open notional : {cfg.risk.max_open_notional_usd} USD\n"
            )
            if input("Type 'trade' to continue: ").strip() != "trade":
                print("aborted")
                return 1

    engine = await build_engine(cfg)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, engine.stop)
    try:
        await engine.run()
    except KeyboardInterrupt:  # pragma: no cover - interactive
        engine.stop()
    return 0


async def cmd_validate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    print(f"mode           : {cfg.mode}")
    print(f"maker (long)   : {cfg.maker_venue.key} {cfg.maker_venue.base_url}")
    print(f"hedge (short)  : {cfg.hedge_venue.key} {cfg.hedge_venue.base_url}")
    print(f"markets        : {', '.join(m.symbol for m in cfg.markets)}")
    print(f"min entry edge : {cfg.edge.min_entry_bps} + {cfg.edge.extra_buffer_bps} bps buffer")
    print(f"max open       : {cfg.risk.max_open_notional_usd} USD")
    print(f"max unhedged   : {cfg.risk.max_unhedged_notional_usd} USD "
          f"for {cfg.risk.unhedged_timeout_ms} ms")

    problems: List[str] = []
    for venue_cfg in (cfg.maker_venue, cfg.hedge_venue):
        try:
            async with LighterRest(venue_cfg.base_url, venue=venue_cfg.key) as rest:
                specs = await rest.market_specs()
        except Exception as exc:
            problems.append(f"{venue_cfg.key}: cannot reach {venue_cfg.base_url}: {exc}")
            continue
        missing = [m.symbol for m in cfg.markets if m.symbol not in specs]
        if missing:
            problems.append(f"{venue_cfg.key}: markets not listed: {', '.join(missing)}")
        else:
            print(f"{venue_cfg.key:<15}: {len(specs)} markets, all configured symbols present")
        if cfg.is_live and venue_cfg.private_key is None:
            problems.append(
                f"{venue_cfg.key}: live mode but {venue_cfg.private_key_env} is not set"
            )

    if problems:
        print("\nproblems:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nconfig OK")
    return 0


# ----------------------------------------------------------------------- entry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spreadbot", description=__doc__)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--logfile", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    p_markets = sub.add_parser("markets", help="list markets on a venue")
    p_markets.add_argument("--url", required=True, help="venue API base URL")
    p_markets.set_defaults(func=cmd_markets)

    p_gaps = sub.add_parser("gaps", help="watch one venue's book for thin spots")
    p_gaps.add_argument("--url", required=True)
    p_gaps.add_argument("-s", "--symbols", default="BTC,ETH")
    p_gaps.add_argument("--wall-usd", type=float, default=20_000)
    p_gaps.add_argument("--wall-multiple", type=float, default=4.0)
    p_gaps.add_argument("--min-gap-bps", type=float, default=1.5)
    p_gaps.add_argument("--min-gap-ticks", type=int, default=2)
    p_gaps.add_argument("--max-distance-bps", type=float, default=50.0)
    p_gaps.add_argument(
        "--max-ahead-usd",
        type=float,
        default=None,
        help="skip spots with more than this much USD queued ahead of the quote",
    )
    p_gaps.add_argument("--transport", choices=["ws", "rest"], default="ws")
    p_gaps.add_argument("--interval", type=float, default=2.0)
    p_gaps.add_argument("--duration", type=float, default=60.0)
    p_gaps.add_argument("--once", action="store_true")
    p_gaps.set_defaults(func=cmd_gaps)

    p_scan = sub.add_parser("scan", help="read-only cross-venue opportunity scan")
    p_scan.add_argument("-c", "--config", required=True)
    p_scan.add_argument("--interval", type=float, default=2.0)
    p_scan.add_argument("--duration", type=float, default=60.0)
    p_scan.add_argument("--once", action="store_true")
    p_scan.set_defaults(func=cmd_scan)

    p_run = sub.add_parser("run", help="run the bot")
    p_run.add_argument("-c", "--config", required=True)
    p_run.add_argument("--paper", action="store_true", help="force paper mode")
    p_run.add_argument("--yes", action="store_true", help="skip the live-trading confirmation")
    p_run.set_defaults(func=cmd_run)

    p_validate = sub.add_parser("validate", help="check config and venue reachability")
    p_validate.add_argument("-c", "--config", required=True)
    p_validate.set_defaults(func=cmd_validate)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, args.logfile)
    try:
        return asyncio.run(args.func(args))
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
