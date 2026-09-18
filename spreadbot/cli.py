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
from pathlib import Path
from typing import List, Optional, Sequence

from .book import OrderBook
from .config import ConfigError, FeedConfig, GapConfig, load_config
from .gaps import book_thinness_bps, find_wall_candidates
from .logging_setup import setup_logging
from .measure import MeasureConfig, Recorder
from .models import Side
from .report import format_summary, load_rows, per_symbol, summarise
from .selector import MarketRanker, clip_size
from .strategy import SpreadStrategy
from .venues.feed import LighterFeed
from .venues.rest import LighterRest
from .volatility import VolatilityTracker

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


async def cmd_rank(args: argparse.Namespace) -> int:
    """Score every active market by how often it offers a place to rest a quote."""
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
    ranker = MarketRanker(gap=gap_cfg)
    async with LighterRest(args.url) as rest:
        specs = await rest.market_specs()
        details = (await rest._get("orderBookDetails"))["order_book_details"]
        volume = {d["symbol"]: float(d.get("daily_quote_token_volume") or 0) for d in details}
        marks = await rest.mark_prices()
        universe = [
            s for s in specs if args.min_volume <= volume.get(s, 0.0) <= args.max_volume
        ]
        print(f"скан {len(universe)} рынков x {args.passes} проход(ов)...", file=sys.stderr)

        for pass_no in range(args.passes):
            for symbol in universe:
                try:
                    bids, asks = await rest.order_book_levels(
                        specs[symbol].market_id, limit=args.depth
                    )
                except Exception as exc:
                    log.debug("%s: %s", symbol, exc)
                    continue
                book = OrderBook("scan", symbol)
                book.apply_snapshot(bids, asks)
                ranker.observe(book, specs[symbol], volume_usd=volume.get(symbol, 0.0))
            if pass_no + 1 < args.passes:
                await asyncio.sleep(args.pass_interval)

        rows = ranker.ranked()
        print(
            f"\n{'symbol':<11}{'score':>8}{'доля':>8}{'эдж bps':>10}{'от мида':>9}"
            f"{'spread':>8}{'глуб $':>11}{'vol $/сут':>13}"
        )
        for row in rows[: args.top]:
            print(
                f"{row.symbol:<11}{row.score:>8.1f}{row.hit_rate * 100:>7.0f}%"
                f"{row.avg_edge_bps:>10.1f}{row.avg_distance_bps:>9.1f}"
                f"{row.avg_spread_bps:>8.1f}{row.avg_depth_usd:>11,.0f}{row.volume_usd:>13,.0f}"
            )
        print(f"\n{len(rows)} рынков с хотя бы одной дыркой из {len(universe)} просканированных")

        if args.emit_config:
            print("\n# --- вставить в config.yaml ---\nmarkets:")
            for row in rows[: args.top]:
                spec, price = specs[row.symbol], marks.get(row.symbol)
                if price is None:
                    continue
                size = clip_size(spec, price, Decimal(str(args.clip_usd)))
                if size is None:
                    continue
                print(
                    f"  - symbol: {row.symbol}\n"
                    f"    order_base: {size}\n"
                    f"    max_position_base: {spec.quantize_size(size * 2)}"
                )
    return 0


async def cmd_measure(args: argparse.Namespace) -> int:
    """Record what the bot would have done, without placing a single order."""
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
    trackers = {s: VolatilityTracker(window_seconds=float(cfg.hedge.vol_window_seconds)) for s in symbols}
    sink = open(args.out, "a") if args.out else None
    recorder = Recorder(
        cfg,
        strategy,
        maker_specs,
        hedge_specs,
        measure=MeasureConfig(
            horizon_seconds=args.horizon,
            max_rest_seconds=args.max_rest,
        ),
        sink=sink,
    )
    try:
        if not (await maker_feed.wait_ready(30) and await hedge_feed.wait_ready(30)):
            print("стаканы не пришли", file=sys.stderr)
            return 1
        log.info(
            "измерение %d рынков на %.0f мин, горизонт отслеживания %.0fs",
            len(symbols),
            args.duration / 60,
            args.horizon,
        )
        deadline = time.time() + args.duration
        while time.time() < deadline:
            now = time.time()
            for symbol in symbols:
                maker_book = maker_feed.books.get(symbol)
                hedge_book = hedge_feed.books.get(symbol)
                if maker_book is None or hedge_book is None:
                    continue
                trackers[symbol].update(now, maker_book.mid)
                if not (maker_feed.is_fresh(symbol) and hedge_feed.is_fresh(symbol)):
                    continue
                recorder.tick(
                    symbol,
                    maker_book,
                    hedge_book,
                    now=now,
                    vol_bps_per_min=trackers[symbol].bps_per_minute(),
                )
            await maker_feed.wait_for_update(0.2)
        recorder.finish()
    finally:
        if sink is not None:
            sink.close()
        for closer in (maker_feed, hedge_feed, maker_rest, hedge_rest):
            with contextlib.suppress(Exception):
                await closer.close()

    print("\n" + format_summary(summarise(recorder.rows), clip_usd=args.clip_usd))
    if args.out:
        print(f"\nсырые строки: {args.out}")
    return 0


async def cmd_report(args: argparse.Namespace) -> int:
    rows = load_rows(args.path)
    if not rows:
        print("в логе нет строк", file=sys.stderr)
        return 1
    print(format_summary(summarise(rows), clip_usd=args.clip_usd))
    if args.by_symbol:
        print(f"\n{'symbol':<12}{'котир':>7}{'филлов':>8}{'отскок':>8}{'эдж факт':>10}")
        for symbol, summary in sorted(
            per_symbol(rows).items(),
            key=lambda kv: kv[1].mean_realised_bps or -1e9,
            reverse=True,
        ):
            realised = (
                f"{summary.mean_realised_bps:+.1f}" if summary.mean_realised_bps is not None else "-"
            )
            print(
                f"{symbol:<12}{summary.quotes:>7}{summary.fills:>8}"
                f"{summary.bounces:>8}{realised:>10}"
            )
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


async def cmd_preflight(args: argparse.Namespace) -> int:
    """Everything that must be true before real money moves. Places no orders."""
    cfg = load_config(args.config)
    setup_logging(cfg.log_level, args.logfile)
    problems: List[str] = []
    warnings: List[str] = []

    print(f"режим          : {cfg.mode}")
    print(f"хедж           : {cfg.hedge.mode}"
          + (f", окно {cfg.hedge.min_seconds}-{cfg.hedge.max_seconds}с, "
             f"аварийный выход {cfg.hedge.panic_bps}bps" if cfg.hedge.is_delayed else ""))
    print(f"ротация        : {'вкл, ' + str(cfg.rotation.watch) + ' рынков' if cfg.rotation.enabled else 'выкл'}")
    print(f"рынков сейчас  : {len(cfg.markets)}")

    if cfg.hedge.is_delayed:
        warnings.append(
            "delayed: в окне ожидания позиция ГОЛАЯ. panic_bps ограничивает глубину, "
            "но не отменяет уже случившийся убыток"
        )

    for venue_cfg in (cfg.maker_venue, cfg.hedge_venue):
        label = f"{venue_cfg.key} ({venue_cfg.allowed_side.value}-only)"
        try:
            async with LighterRest(venue_cfg.base_url, venue=venue_cfg.key) as rest:
                specs = await rest.market_specs()
                missing = [m.symbol for m in cfg.markets if m.symbol not in specs]
                if missing:
                    problems.append(f"{venue_cfg.key}: нет рынков {', '.join(missing)}")

                for market in cfg.markets:
                    spec = specs.get(market.symbol)
                    if spec is None:
                        continue
                    if market.order_base < spec.min_base_amount:
                        problems.append(
                            f"{venue_cfg.key}:{market.symbol} order_base {market.order_base} "
                            f"ниже минимума {spec.min_base_amount}"
                        )

                if venue_cfg.account_index is None:
                    problems.append(f"{venue_cfg.key}: не задан account_index")
                else:
                    account = await rest.account_state(venue_cfg.account_index)
                    collateral = Decimal(str(account.get("collateral", 0)))
                    available = Decimal(str(account.get("available_balance", 0)))
                    print(
                        f"{label:<28}: коллатерал {collateral:.2f}, "
                        f"свободно {available:.2f}, {len(specs)} рынков"
                    )
                    if collateral <= 0:
                        problems.append(f"{venue_cfg.key}: на аккаунте нет средств")
                    elif available < cfg.risk.max_open_notional_usd / 10:
                        warnings.append(
                            f"{venue_cfg.key}: свободно {available:.0f} при лимите "
                            f"max_open_notional_usd {cfg.risk.max_open_notional_usd}"
                        )
                    # Positions on the forbidden side would halt the bot at once.
                    for entry in account.get("positions", []):
                        size = Decimal(str(entry.get("position", 0)))
                        if size == 0:
                            continue
                        sign = int(entry.get("sign", 1))
                        wrong = (
                            venue_cfg.allowed_side is Side.BUY and sign < 0
                        ) or (venue_cfg.allowed_side is Side.SELL and sign > 0)
                        if wrong:
                            problems.append(
                                f"{venue_cfg.key}: уже есть позиция на запрещённой стороне "
                                f"({entry.get('symbol')} {'short' if sign < 0 else 'long'} {size})"
                            )
        except Exception as exc:
            problems.append(f"{venue_cfg.key}: не отвечает {venue_cfg.base_url}: {exc}")
            continue

        if cfg.is_live:
            if venue_cfg.private_key is None:
                problems.append(
                    f"{venue_cfg.key}: live-режим, но {venue_cfg.private_key_env} не задан"
                )
            else:
                print(f"{venue_cfg.key:<28}: ключ из {venue_cfg.private_key_env} найден")

    if cfg.risk.kill_switch_file:
        switch = Path(cfg.risk.kill_switch_file)
        print(f"kill switch    : {switch} ({'АКТИВЕН' if switch.exists() else 'свободен'})")
        if switch.exists():
            warnings.append("kill switch активен - бот не откроет ни одной новой позиции")
        elif not switch.parent.exists():
            problems.append(f"каталог для kill switch не существует: {switch.parent}")

    for warning in warnings:
        print(f"\n[!] {warning}")
    if problems:
        print("\nблокирует запуск:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nпроверки пройдены" + (" (но прочитай предупреждения выше)" if warnings else ""))
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

    p_rank = sub.add_parser("rank", help="score every market by how gappy its book is")
    p_rank.add_argument("--url", required=True)
    p_rank.add_argument("--top", type=int, default=30)
    p_rank.add_argument("--passes", type=int, default=3, help="book snapshots per market")
    p_rank.add_argument("--pass-interval", type=float, default=20.0)
    p_rank.add_argument("--depth", type=int, default=250)
    p_rank.add_argument("--wall-usd", type=float, default=10_000)
    p_rank.add_argument("--wall-multiple", type=float, default=3.0)
    p_rank.add_argument("--min-gap-bps", type=float, default=15.0)
    p_rank.add_argument("--min-gap-ticks", type=int, default=2)
    p_rank.add_argument("--max-distance-bps", type=float, default=150.0)
    p_rank.add_argument("--max-ahead-usd", type=float, default=60_000)
    p_rank.add_argument("--min-volume", type=float, default=50_000)
    p_rank.add_argument("--max-volume", type=float, default=5e8)
    p_rank.add_argument("--clip-usd", type=float, default=50.0)
    p_rank.add_argument(
        "--emit-config", action="store_true", help="print a ready markets: block"
    )
    p_rank.set_defaults(func=cmd_rank)

    p_measure = sub.add_parser(
        "measure", help="record would-be quotes and their outcomes, placing no orders"
    )
    p_measure.add_argument("-c", "--config", required=True)
    p_measure.add_argument("--duration", type=float, default=3_600.0, help="seconds to watch")
    p_measure.add_argument("--horizon", type=float, default=60.0, help="seconds tracked after a fill")
    p_measure.add_argument("--max-rest", type=float, default=600.0)
    p_measure.add_argument("--clip-usd", type=float, default=50.0)
    p_measure.add_argument("--out", default=None, help="append JSONL rows here")
    p_measure.set_defaults(func=cmd_measure)

    p_report = sub.add_parser("report", help="summarise a measurement log")
    p_report.add_argument("path")
    p_report.add_argument("--clip-usd", type=float, default=50.0)
    p_report.add_argument("--by-symbol", action="store_true")
    p_report.set_defaults(func=cmd_report)

    p_run = sub.add_parser("run", help="run the bot")
    p_run.add_argument("-c", "--config", required=True)
    p_run.add_argument("--paper", action="store_true", help="force paper mode")
    p_run.add_argument("--yes", action="store_true", help="skip the live-trading confirmation")
    p_run.set_defaults(func=cmd_run)

    p_preflight = sub.add_parser(
        "preflight", help="pre-launch checks: keys, balances, minimums, positions"
    )
    p_preflight.add_argument("-c", "--config", required=True)
    p_preflight.set_defaults(func=cmd_preflight)

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
