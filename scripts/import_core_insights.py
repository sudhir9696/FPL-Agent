#!/usr/bin/env python3
"""Build FPL-API-shaped data from the FPL Core Insights dataset.

Why this exists: some networks block `fantasy.premierleague.com` outright.
FPL Core Insights (https://github.com/olbauday/FPL-Core-Insights) mirrors the
same data to GitHub as CSVs, refreshed twice daily, and adds CBIT metrics and
Elo ratings. This script converts it into the `bootstrap.json` / `fixtures.json`
pair that `fpl-agent --data-dir` already reads, so the whole tool runs on real,
current data without touching the FPL API.

Pre-season the current season's counting stats are all zero, so a player's
statistical base is carried forward from the prior season, matched on the
`player_code` that is stable across seasons. Prices, ownership, availability
and set-piece order always come from the current season.

Usage:
    git clone --depth 1 https://github.com/olbauday/FPL-Core-Insights
    python scripts/import_core_insights.py FPL-Core-Insights -o fpl-data
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

POSITION_IDS = {"Goalkeeper": 1, "Defender": 2, "Midfielder": 3, "Forward": 4}

# Counting stats carried forward from the prior season as the model's base.
CARRIED_STATS = [
    "minutes", "starts", "goals_scored", "assists", "clean_sheets",
    "goals_conceded", "own_goals", "penalties_saved", "penalties_missed",
    "yellow_cards", "red_cards", "saves", "bonus", "bps",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "defensive_contribution",
    "tackles", "clearances_blocks_interceptions", "recoveries",
    "influence", "creativity", "threat", "ict_index",
]

# Live fields that must always come from the current season.
CURRENT_STATS = [
    "status", "chance_of_playing_next_round", "chance_of_playing_this_round",
    "selected_by_percent", "form", "ep_next", "ep_this", "news",
    "total_points", "event_points", "points_per_game",
    "transfers_in_event", "transfers_out_event",
    "penalties_order", "corners_and_indirect_freekicks_order",
    "direct_freekicks_order",
]


def _num(value, default=0):
    if value is None or value == "":
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return int(f) if f == int(f) else f


def read_csv(path: Path) -> list:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def latest_rows_by_player(rows: list) -> dict:
    """Cumulative snapshots are per-gameweek; keep each player's last one."""
    latest: dict = {}
    for row in rows:
        pid = row.get("id")
        if not pid:
            continue
        gw = _num(row.get("gw"), 0)
        if pid not in latest or gw >= latest[pid][0]:
            latest[pid] = (gw, row)
    return {pid: row for pid, (_gw, row) in latest.items()}


def build_teams(rows: list) -> tuple:
    """Return (elements-ready team list, code->id map)."""
    teams, code_to_id = [], {}
    for row in rows:
        team_id = _num(row["id"])
        code_to_id[str(_num(row["code"]))] = team_id
        # Pre-season the granular strengths are zeroed and only a 1-5 overall
        # tier is published; models.py already projects tiers onto the in-season
        # scale, so pass both through untouched.
        teams.append({
            "id": team_id,
            "name": row.get("name", ""),
            "short_name": row.get("short_name", ""),
            "strength": _num(row.get("strength")) or _num(row.get("strength_overall_home"), 3),
            "strength_overall_home": _num(row.get("strength_overall_home"), 3),
            "strength_overall_away": _num(row.get("strength_overall_away"), 3),
            "strength_attack_home": _num(row.get("strength_attack_home")),
            "strength_attack_away": _num(row.get("strength_attack_away")),
            "strength_defence_home": _num(row.get("strength_defence_home")),
            "strength_defence_away": _num(row.get("strength_defence_away")),
        })
    return teams, code_to_id


def build_elements(season_dir: Path, prior_dir: Path | None, code_to_id: dict) -> list:
    players = read_csv(season_dir / "players.csv")
    stats = latest_rows_by_player(read_csv(season_dir / "playerstats.csv"))

    prior_by_code: dict = {}
    if prior_dir and (prior_dir / "players.csv").exists():
        prior_players = read_csv(prior_dir / "players.csv")
        prior_stats = latest_rows_by_player(read_csv(prior_dir / "playerstats.csv"))
        by_id_to_code = {p["player_id"]: p["player_code"] for p in prior_players}
        for pid, row in prior_stats.items():
            code = by_id_to_code.get(pid)
            if code:
                # Several snapshots share a code; keep the fullest season.
                existing = prior_by_code.get(code)
                if not existing or _num(row.get("minutes")) > _num(existing.get("minutes")):
                    prior_by_code[code] = row

    elements, carried = [], 0
    for player in players:
        pid = player["player_id"]
        stat = stats.get(pid, {})
        element = {
            "id": _num(pid),
            "web_name": player.get("web_name", ""),
            "first_name": player.get("first_name", ""),
            "second_name": player.get("second_name", ""),
            "team": code_to_id.get(str(_num(player.get("team_code"))), 1),
            "element_type": POSITION_IDS.get(player.get("position", ""), 3),
            # Core Insights publishes price in millions; the API uses tenths.
            "now_cost": int(round(_num(stat.get("now_cost"), 4.0) * 10)),
        }

        for field in CURRENT_STATS:
            value = stat.get(field)
            element[field] = value if value not in ("", None) else None
        element["news"] = element.get("news") or ""
        # chance_of_playing is null when the player is fully fit.
        for field in ("chance_of_playing_next_round", "chance_of_playing_this_round"):
            element[field] = _num(element[field], None) if element[field] else None
        element["status"] = element.get("status") or "a"

        prior = prior_by_code.get(player.get("player_code"))
        source = stat if _num(stat.get("minutes")) > 0 else (prior or {})
        if source is prior and prior:
            carried += 1
        for field in CARRIED_STATS:
            element[field] = _num(source.get(field), 0)

        elements.append(element)

    print(f"  carried prior-season stats for {carried}/{len(elements)} players",
          file=sys.stderr)
    return elements


def build_fixtures(season_dir: Path, code_to_id: dict, teams: list) -> list:
    """Read every By Gameweek/GW*/fixtures.csv into API-shaped fixtures."""
    tier = {t["id"]: t["strength_overall_home"] for t in teams}
    gw_root = season_dir / "By Gameweek"
    fixtures, fixture_id, seen = [], 1, set()

    directories = sorted(
        (d for d in gw_root.iterdir() if d.is_dir() and d.name.startswith("GW")),
        key=lambda d: int(d.name[2:] or 0),
    )
    for directory in directories:
        path = directory / "fixtures.csv"
        if not path.exists():
            continue
        for row in read_csv(path):
            if row.get("tournament") and row["tournament"] != "prem":
                continue    # cup ties are not FPL fixtures
            home = code_to_id.get(str(_num(row.get("home_team"))))
            away = code_to_id.get(str(_num(row.get("away_team"))))
            if not home or not away:
                continue
            key = (row.get("match_id") or f"{row.get('gameweek')}-{home}-{away}")
            if key in seen:
                continue
            seen.add(key)
            fixtures.append({
                "id": fixture_id,
                "event": _num(row.get("gameweek"), None),
                "team_h": home,
                "team_a": away,
                # No published FDR here: approximate it from the opponent's tier.
                "team_h_difficulty": max(1, min(5, tier.get(away, 3))),
                "team_a_difficulty": max(1, min(5, tier.get(home, 3) + 1)),
                "finished": str(row.get("finished", "")).lower() == "true",
                "kickoff_time": row.get("kickoff_time") or None,
            })
            fixture_id += 1
    return fixtures


def build_events(fixtures: list) -> list:
    """Derive gameweek state from which fixtures have finished."""
    by_gw: dict = {}
    for fixture in fixtures:
        if fixture["event"]:
            by_gw.setdefault(fixture["event"], []).append(fixture)

    events = []
    unfinished = sorted(gw for gw, fx in by_gw.items() if not all(f["finished"] for f in fx))
    current = unfinished[0] if unfinished else max(by_gw or [1])
    for gw in sorted(by_gw):
        matches = by_gw[gw]
        finished = all(f["finished"] for f in matches)
        kickoffs = sorted(f["kickoff_time"] for f in matches if f["kickoff_time"])
        events.append({
            "id": gw,
            "name": f"Gameweek {gw}",
            "finished": finished,
            "is_current": gw == current,
            "is_next": gw == current + 1,
            "deadline_time": kickoffs[0] if kickoffs else None,
        })
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo", type=Path, help="path to a FPL-Core-Insights checkout")
    parser.add_argument("-o", "--output", type=Path, default=Path("fpl-data"))
    parser.add_argument("--season", default=None, help="e.g. 2026-2027 (default: latest)")
    parser.add_argument("--prior-season", default=None,
                        help="season to carry stats from (default: the one before)")
    args = parser.parse_args()

    data_root = args.repo / "data"
    if not data_root.is_dir():
        print(f"error: {data_root} not found -- is that a FPL-Core-Insights checkout?",
              file=sys.stderr)
        return 1

    seasons = sorted(d.name for d in data_root.iterdir() if d.is_dir())
    season = args.season or seasons[-1]
    if season not in seasons:
        print(f"error: season {season} not in {seasons}", file=sys.stderr)
        return 1
    prior = args.prior_season
    if prior is None:
        index = seasons.index(season)
        prior = seasons[index - 1] if index > 0 else None

    season_dir = data_root / season
    prior_dir = data_root / prior if prior else None
    print(f"Importing {season}" + (f", carrying stats from {prior}" if prior else ""),
          file=sys.stderr)

    teams, code_to_id = build_teams(read_csv(season_dir / "teams.csv"))
    elements = build_elements(season_dir, prior_dir, code_to_id)
    fixtures = build_fixtures(season_dir, code_to_id, teams)
    events = build_events(fixtures)

    bootstrap = {
        "events": events,
        "teams": teams,
        "elements": elements,
        "element_types": [
            {"id": 1, "singular_name_short": "GKP", "squad_select": 2},
            {"id": 2, "singular_name_short": "DEF", "squad_select": 5},
            {"id": 3, "singular_name_short": "MID", "squad_select": 5},
            {"id": 4, "singular_name_short": "FWD", "squad_select": 3},
        ],
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "bootstrap.json").write_text(json.dumps(bootstrap))
    (args.output / "fixtures.json").write_text(json.dumps(fixtures))

    current = next((e["id"] for e in events if e["is_current"]), None)
    print(f"Wrote {len(elements)} players, {len(teams)} teams, {len(fixtures)} fixtures "
          f"to {args.output}/", file=sys.stderr)
    print(f"Current GW: {current}   Run: fpl-agent build --data-dir {args.output}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
