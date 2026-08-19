"""Command-line interface for the FPL team builder."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path
from typing import Optional

from . import report
from .analysis import ModelConfig, ProjectionModel, best_value, differentials, top_by_position
from .api import FPLClient, FPLError, GameData
from .models import POSITIONS
from .optimizer import OptimizerError, build_squad_object, optimize_squad, suggest_transfers
from .reddit import RedditClient, RedditError, cross_reference, extract_player_buzz, gather_threads

log = logging.getLogger("fpl_builder")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--horizon", type=int, default=5,
                        help="number of gameweeks to plan over (default: 5)")
    parser.add_argument("--gw", type=int, default=None,
                        help="gameweek to plan from (default: the next one)")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="read bootstrap.json/fixtures.json from here instead of the API")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="where to cache API responses")
    parser.add_argument("--refresh", action="store_true",
                        help="bypass the cache and refetch from the API")
    parser.add_argument("--weight-model", type=float, default=0.60,
                        help="weight on the underlying-stats model (default: 0.60)")
    parser.add_argument("--weight-form", type=float, default=0.20,
                        help="weight on recent form (default: 0.20)")
    parser.add_argument("--weight-ep", type=float, default=0.20,
                        help="weight on FPL's own ep_next (default: 0.20)")
    parser.add_argument("--no-defcon", action="store_true",
                        help="ignore defensive-contribution points")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="write a Markdown report to this path")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logging")


def _add_reddit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-reddit", action="store_true",
                        help="skip Reddit analysis entirely")
    parser.add_argument("--subreddits", nargs="+", default=["FantasyPL"],
                        help="subreddits to mine (default: FantasyPL)")
    parser.add_argument("--reddit-limit", type=int, default=25,
                        help="threads to pull per listing (default: 25)")
    parser.add_argument("--reddit-threads", type=int, default=5,
                        help="top threads to pull comments for (default: 5)")


def load_game_data(args) -> GameData:
    if args.data_dir:
        bootstrap = Path(args.data_dir) / "bootstrap.json"
        fixtures = Path(args.data_dir) / "fixtures.json"
        missing = [p for p in (bootstrap, fixtures) if not p.exists()]
        if missing:
            raise SystemExit(
                f"error: missing {', '.join(str(m) for m in missing)}. "
                f"Run `fpl-builder fetch --data-dir {args.data_dir}` first."
            )
        log.info("loading offline data from %s", args.data_dir)
        return GameData.from_files(bootstrap, fixtures)

    client = FPLClient(cache_dir=args.cache_dir)
    return GameData.load(client, use_cache=not args.refresh)


def build_model_config(args) -> ModelConfig:
    return ModelConfig(
        horizon=args.horizon,
        weight_model=args.weight_model,
        weight_form=args.weight_form,
        weight_ep=args.weight_ep,
        include_defcon=not args.no_defcon,
    )


def gather_reddit(args, data, gameweek: int) -> tuple:
    """Return (threads, buzz, cross_reference) -- empty if unavailable."""
    if getattr(args, "no_reddit", False):
        return [], {}, {}
    try:
        client = RedditClient(cache_dir=(Path(args.cache_dir) / "reddit") if args.cache_dir else None)
        threads = gather_threads(
            client,
            subreddits=args.subreddits,
            limit=args.reddit_limit,
            gameweek=gameweek,
            fetch_comments=args.reddit_threads,
        )
    except RedditError as exc:
        print(f"warning: Reddit analysis skipped -- {exc}", file=sys.stderr)
        return [], {}, {}

    if not threads:
        print("warning: no Reddit threads retrieved.", file=sys.stderr)
        return [], {}, {}

    buzz = extract_player_buzz(threads, data.players)
    return threads, buzz, {}


def cmd_build(args) -> int:
    data = load_game_data(args)
    start_gw = args.gw or data.next_gameweek
    config = build_model_config(args)
    model = ProjectionModel(data, config)

    log.info("projecting %d players from GW%d over %d GWs",
             len(data.players), start_gw, config.horizon)
    projections = model.project_all(start_gw=start_gw)

    lock_ids, exclude_ids = _resolve_name_filters(args, data)

    try:
        squad = optimize_squad(
            projections,
            budget=args.budget,
            max_per_team=args.max_per_team,
            bench_weight=args.bench_weight,
            lock_ids=lock_ids,
            exclude_ids=exclude_ids,
        )
    except OptimizerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    threads, buzz, _ = gather_reddit(args, data, start_gw)
    cross = cross_reference(projections, buzz) if buzz else {}

    print(report.format_squad(squad, data, config.horizon))
    if threads:
        print(report.format_threads(threads))
    if buzz:
        print(report.format_buzz(buzz, data))
    if cross:
        print(report.format_cross_reference(cross, data))

    _maybe_write_report(
        args, squad, projections, data, config.horizon, start_gw, threads, buzz, cross
    )
    return 0


def cmd_players(args) -> int:
    data = load_game_data(args)
    start_gw = args.gw or data.next_gameweek
    config = build_model_config(args)
    projections = ProjectionModel(data, config).project_all(start_gw=start_gw)

    if args.max_price is not None:
        projections = [p for p in projections if p.player.cost <= args.max_price]
    if args.team:
        wanted = {t.lower() for t in args.team}
        projections = [
            p for p in projections
            if data.teams.get(p.player.team)
            and data.teams[p.player.team].short_name.lower() in wanted
        ]

    if args.sort == "value":
        projections = best_value(projections, limit=len(projections))
    elif args.sort == "differential":
        projections = differentials(projections, max_ownership=args.max_ownership,
                                    limit=len(projections))

    if args.position:
        for position in args.position:
            print(report.format_rankings(
                top_by_position(projections, position.upper(), args.limit),
                data, f"TOP {position.upper()} (GW{start_gw}, {config.horizon} GW horizon)",
                args.limit,
            ))
    else:
        print(report.format_rankings(
            projections, data,
            f"TOP PLAYERS (GW{start_gw}, {config.horizon} GW horizon, sorted by {args.sort})",
            args.limit,
        ))

    _maybe_write_report(args, None, projections, data, config.horizon, start_gw, [], {}, {})
    return 0


def cmd_reddit(args) -> int:
    data = load_game_data(args)
    start_gw = args.gw or data.next_gameweek
    config = build_model_config(args)
    projections = ProjectionModel(data, config).project_all(start_gw=start_gw)

    threads, buzz, _ = gather_reddit(args, data, start_gw)
    if not threads:
        return 1

    cross = cross_reference(projections, buzz) if buzz else {}
    print(report.format_threads(threads, limit=args.limit))
    if buzz:
        print(report.format_buzz(buzz, data, limit=args.limit))
    if cross:
        print(report.format_cross_reference(cross, data))

    _maybe_write_report(
        args, None, projections, data, config.horizon, start_gw, threads, buzz, cross
    )
    return 0


def cmd_team(args) -> int:
    data = load_game_data(args)
    config = build_model_config(args)
    start_gw = args.gw or data.next_gameweek
    projections = ProjectionModel(data, config).project_all(start_gw=start_gw)
    by_id = {p.player.id: p for p in projections}

    client = FPLClient(cache_dir=args.cache_dir)
    picks_gw = args.picks_gw or max(1, data.current_gameweek)
    try:
        entry = client.entry(args.entry)
        picks_payload = client.entry_picks(args.entry, picks_gw, use_cache=not args.refresh)
    except FPLError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    bank = entry.get("last_deadline_bank")
    bank = (bank / 10.0) if isinstance(bank, (int, float)) else 0.0

    current = []
    for pick in picks_payload.get("picks", []):
        projection = by_id.get(pick.get("element"))
        if projection:
            current.append(projection)

    if len(current) != 15:
        print(
            f"warning: resolved {len(current)}/15 picks for entry {args.entry} in GW{picks_gw}.",
            file=sys.stderr,
        )
    if not current:
        print("error: no picks resolved -- check the entry id and gameweek.", file=sys.stderr)
        return 1

    manager = f"{entry.get('player_first_name', '')} {entry.get('player_last_name', '')}".strip()
    print(f"\nTEAM: {entry.get('name', '?')} ({manager or 'unknown manager'})")
    print(f"Bank: {bank:.1f}m  |  Picks from GW{picks_gw}  |  Planning GW{start_gw}"
          f"-GW{start_gw + config.horizon - 1}")

    squad = build_squad_object(current)
    print(report.format_squad(squad, data, config.horizon))

    moves = suggest_transfers(
        current, projections, bank=bank,
        free_transfers=args.free_transfers, max_transfers=args.max_transfers,
    )
    print(report.format_transfers(moves, data))

    threads, buzz, _ = gather_reddit(args, data, start_gw)
    cross = cross_reference(projections, buzz) if buzz else {}
    if threads:
        print(report.format_threads(threads))
    if buzz:
        print(report.format_buzz(buzz, data))
    if cross:
        print(report.format_cross_reference(cross, data))

    _maybe_write_report(
        args, squad, projections, data, config.horizon, start_gw, threads, buzz, cross, moves
    )
    return 0


def cmd_fetch(args) -> int:
    """Download and store the raw payloads for later offline runs."""
    client = FPLClient(cache_dir=args.cache_dir)
    try:
        data = GameData.load(client, use_cache=not args.refresh)
    except FPLError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    target = Path(args.data_dir or "fpl-data")
    data.save(target)
    print(f"Saved {len(data.players)} players and {len(data.fixtures)} fixtures to {target}/")
    print(f"Current GW: {data.current_gameweek}  |  Next GW: {data.next_gameweek}")
    print(f"Replay offline with: fpl-builder build --data-dir {target}")
    return 0


def _resolve_name_filters(args, data) -> tuple:
    """Turn --lock/--exclude names or ids into player ids."""
    def resolve(values) -> set:
        resolved = set()
        for value in values or []:
            if str(value).isdigit():
                resolved.add(int(value))
                continue
            needle = str(value).lower()
            matches = [
                p for p in data.players
                if needle == p.web_name.lower() or needle in p.full_name.lower()
            ]
            if not matches:
                raise SystemExit(f"error: no player matches '{value}'")
            if len(matches) > 1:
                exact = [p for p in matches if p.web_name.lower() == needle]
                if len(exact) == 1:
                    matches = exact
                else:
                    names = ", ".join(f"{p.web_name} (id {p.id})" for p in matches[:8])
                    raise SystemExit(f"error: '{value}' is ambiguous -- {names}")
            resolved.add(matches[0].id)
        return resolved

    return resolve(getattr(args, "lock", None)), resolve(getattr(args, "exclude", None))


def _maybe_write_report(args, squad, projections, data, horizon, start_gw,
                        threads=None, buzz=None, cross=None, transfers=None) -> None:
    if not args.output:
        return
    markdown = report.markdown_report(
        squad, projections, data, horizon, start_gw,
        threads=threads, buzz=buzz, cross=cross, transfers=transfers,
        generated_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown)
    print(f"\nMarkdown report written to {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fpl-builder",
        description="Build and analyse a Fantasy Premier League squad using the "
                    "public FPL API and r/FantasyPL discussion.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build an optimal squad from scratch")
    _add_common_args(build)
    _add_reddit_args(build)
    build.add_argument("--budget", type=float, default=100.0, help="squad budget in millions")
    build.add_argument("--max-per-team", type=int, default=3, help="club limit (default: 3)")
    build.add_argument("--bench-weight", type=float, default=0.12,
                       help="how much bench points count toward the objective")
    build.add_argument("--lock", nargs="+", default=[],
                       help="players who must be in the squad (name or id)")
    build.add_argument("--exclude", nargs="+", default=[],
                       help="players to keep out of the squad (name or id)")
    build.set_defaults(func=cmd_build)

    players = subparsers.add_parser("players", help="rank players by projected points")
    _add_common_args(players)
    players.add_argument("--position", nargs="+", choices=[p.lower() for p in POSITIONS.values()]
                         + list(POSITIONS.values()), help="filter by position")
    players.add_argument("--limit", type=int, default=20, help="rows to show (default: 20)")
    players.add_argument("--sort", choices=["points", "value", "differential"], default="points")
    players.add_argument("--max-price", type=float, default=None, help="price ceiling")
    players.add_argument("--max-ownership", type=float, default=5.0,
                         help="ownership ceiling for --sort differential")
    players.add_argument("--team", nargs="+", default=None,
                         help="filter by team short name, e.g. ARS LIV")
    players.set_defaults(func=cmd_players)

    reddit_cmd = subparsers.add_parser("reddit", help="mine r/FantasyPL for player signal")
    _add_common_args(reddit_cmd)
    _add_reddit_args(reddit_cmd)
    reddit_cmd.add_argument("--limit", type=int, default=15, help="rows to show (default: 15)")
    reddit_cmd.set_defaults(func=cmd_reddit)

    team = subparsers.add_parser("team", help="analyse an existing team and suggest transfers")
    _add_common_args(team)
    _add_reddit_args(team)
    team.add_argument("entry", type=int, help="your FPL entry (team) id")
    team.add_argument("--picks-gw", type=int, default=None,
                      help="gameweek to read picks from (default: current)")
    team.add_argument("--free-transfers", type=int, default=1)
    team.add_argument("--max-transfers", type=int, default=3)
    team.set_defaults(func=cmd_team)

    fetch = subparsers.add_parser("fetch", help="save API data for offline use")
    _add_common_args(fetch)
    fetch.set_defaults(func=cmd_fetch)

    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return args.func(args)
    except FPLError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
