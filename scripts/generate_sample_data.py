#!/usr/bin/env python3
"""Generate a realistic offline dataset mirroring the FPL API schema.

Produces `bootstrap.json` and `fixtures.json` with the same field names and
types the live endpoints return, so tests and demo runs exercise exactly the
code paths a live run does. Deterministic: a fixed seed means stable output.

Player names are randomly assembled from real-sounding name pools and do NOT
correspond to real footballers or their real positions -- only the schema,
value ranges and statistical relationships are realistic.

Usage:  python scripts/generate_sample_data.py [output_dir]
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

SEED = 20250819


def _poisson_at_least(k: int, lam: float) -> float:
    """P(X >= k) for X ~ Poisson(lam)."""
    if k <= 0:
        return 1.0
    if lam <= 0:
        return 0.0
    term = math.exp(-lam)
    cumulative = term
    for i in range(1, k):
        term *= lam / i
        cumulative += term
    return max(0.0, min(1.0, 1.0 - cumulative))

TEAMS = [
    ("Arsenal", "ARS", 5), ("Aston Villa", "AVL", 4), ("Bournemouth", "BOU", 3),
    ("Brentford", "BRE", 3), ("Brighton", "BHA", 4), ("Chelsea", "CHE", 4),
    ("Crystal Palace", "CRY", 3), ("Everton", "EVE", 3), ("Fulham", "FUL", 3),
    ("Ipswich", "IPS", 2), ("Leicester", "LEI", 2), ("Liverpool", "LIV", 5),
    ("Man City", "MCI", 5), ("Man Utd", "MUN", 4), ("Newcastle", "NEW", 4),
    ("Nott'm Forest", "NFO", 3), ("Southampton", "SOU", 2), ("Spurs", "TOT", 4),
    ("West Ham", "WHU", 3), ("Wolves", "WOL", 3),
]

FIRST_NAMES = [
    "James", "Mohamed", "Bruno", "Cole", "Kai", "Alexander", "Diogo", "Bukayo",
    "Erling", "Phil", "Declan", "Ollie", "Morgan", "Anthony", "Jarrod", "Dominic",
    "Luis", "Pedro", "Joao", "Matheus", "Nicolas", "Emiliano", "Antoine", "Yoane",
    "Marc", "Trent", "Virgil", "Andrew", "Kieran", "Ben", "Levi", "Noni",
]
LAST_NAMES = [
    "Saka", "Salah", "Palmer", "Haaland", "Foden", "Odegaard", "Watkins", "Isak",
    "Gordon", "Mbeumo", "Wissa", "Solanke", "Bowen", "Rogers", "Semenyo", "Kluivert",
    "Gabriel", "Saliba", "Virgil", "Robertson", "Trippier", "Mykolenko", "Munoz", "Hall",
    "Raya", "Sanchez", "Pickford", "Onana", "Verbruggen", "Areola", "Flekken", "Henderson",
    "Rice", "Caicedo", "Gravenberch", "Mainoo", "Baleba", "Wharton", "Anderson", "Ward",
]

POSITION_PLAN = [(1, 3), (2, 8), (3, 9), (4, 4)]  # (element_type, count) per club


def _round1(value: float) -> float:
    return round(value, 1)


def build_dataset(seed: int = SEED) -> tuple:
    rng = random.Random(seed)

    teams = []
    for index, (name, short, tier) in enumerate(TEAMS, start=1):
        base_attack = 950 + tier * 90 + rng.randint(-40, 40)
        base_defence = 950 + tier * 85 + rng.randint(-40, 40)
        teams.append({
            "id": index, "name": name, "short_name": short, "strength": tier,
            "strength_overall_home": base_attack + 60,
            "strength_overall_away": base_attack,
            "strength_attack_home": base_attack + 55,
            "strength_attack_away": base_attack,
            "strength_defence_home": base_defence + 55,
            "strength_defence_away": base_defence,
        })

    # 38 gameweeks; 12 already played so players carry real season totals.
    played_gws = 12
    events = []
    for gw in range(1, 39):
        events.append({
            "id": gw, "name": f"Gameweek {gw}",
            "finished": gw <= played_gws,
            "is_current": gw == played_gws,
            "is_next": gw == played_gws + 1,
            "deadline_time": f"2025-{8 + (gw // 5):02d}-{(gw % 28) + 1:02d}T11:00:00Z",
        })

    # Double round robin, rotated so every gameweek has 10 fixtures.
    fixtures, fixture_id = [], 1
    team_ids = [t["id"] for t in teams]
    for gw in range(1, 39):
        rotation = team_ids[1:]
        shift = (gw - 1) % len(rotation)
        rotated = [team_ids[0]] + rotation[shift:] + rotation[:shift]
        half = len(rotated) // 2
        for slot in range(half):
            home, away = rotated[slot], rotated[-(slot + 1)]
            if gw % 2 == 0:  # reverse fixture in the second meeting
                home, away = away, home
            home_tier = next(t["strength"] for t in teams if t["id"] == home)
            away_tier = next(t["strength"] for t in teams if t["id"] == away)
            fixtures.append({
                "id": fixture_id,
                "event": gw,
                "team_h": home,
                "team_a": away,
                # FDR: how hard this match is for the named team.
                "team_h_difficulty": min(5, max(1, away_tier)),
                "team_a_difficulty": min(5, max(1, home_tier + 1)),
                "finished": gw <= played_gws,
                # The real API sets both flags on a settled match; only in the
                # day or so after the whistle does provisional lead `finished`.
                "finished_provisional": gw <= played_gws,
                "kickoff_time": f"2025-{8 + (gw // 5):02d}-{(gw % 28) + 1:02d}T14:00:00Z",
            })
            fixture_id += 1

    elements, element_id = [], 1
    used_names = set()
    for team in teams:
        tier = team["strength"]
        for element_type, count in POSITION_PLAN:
            for slot in range(count):
                # Slot 0 is the nailed first-choice player, later slots are backups.
                nailed = slot == 0 or (element_type in (2, 3) and slot <= 2)
                quality = tier / 5.0 * (1.0 if nailed else 0.55) * rng.uniform(0.75, 1.25)

                minutes_played = int(
                    played_gws * 90 * (0.92 if nailed else rng.uniform(0.05, 0.5))
                    * rng.uniform(0.85, 1.0)
                )
                minutes_played = max(0, min(played_gws * 90, minutes_played))
                starts = int(round(minutes_played / 90.0 * rng.uniform(0.9, 1.0)))
                nineties = max(minutes_played / 90.0, 0.01)

                if element_type == 1:
                    xg90, xa90 = 0.0, 0.0
                elif element_type == 2:
                    xg90, xa90 = 0.05 * quality, 0.07 * quality
                elif element_type == 3:
                    xg90, xa90 = 0.22 * quality, 0.20 * quality
                else:
                    xg90, xa90 = 0.48 * quality, 0.16 * quality

                expected_goals = xg90 * nineties * rng.uniform(0.8, 1.2)
                expected_assists = xa90 * nineties * rng.uniform(0.8, 1.2)
                goals = int(round(expected_goals * rng.uniform(0.6, 1.5)))
                assists = int(round(expected_assists * rng.uniform(0.6, 1.5)))

                # Better defences concede less.
                xgc90 = max(0.35, 2.3 - tier * 0.28) * rng.uniform(0.85, 1.15)
                expected_goals_conceded = xgc90 * nineties
                goals_conceded = int(round(expected_goals_conceded * rng.uniform(0.8, 1.2)))
                clean_sheets = int(round(nineties * max(0.0, 0.42 - xgc90 * 0.12)))

                saves = int(nineties * rng.uniform(2.2, 4.0)) if element_type == 1 else 0
                # As in the real API, this is a count of defensive ACTIONS
                # (clearances, blocks, interceptions, tackles), not of awards
                # won. Rates are the league medians per 90 by position.
                defcon_rate = {1: 0.0, 2: 7.7, 3: 8.0, 4: 4.5}[element_type]
                defcon_per90 = defcon_rate * rng.uniform(0.6, 1.4)
                defensive_contribution = round(nineties * defcon_per90, 1)
                # An award needs 10 actions in a match for a defender, 12 for
                # everyone else; approximate how often that threshold was met.
                defcon_threshold = {1: 999, 2: 10, 3: 12, 4: 12}[element_type]
                defcon_awards = int(round(
                    starts * _poisson_at_least(defcon_threshold, defcon_per90)
                ))

                points = (
                    starts * 2
                    + goals * {1: 6, 2: 6, 3: 5, 4: 4}[element_type]
                    + assists * 3
                    + clean_sheets * {1: 4, 2: 4, 3: 1, 4: 0}[element_type]
                    + saves // 3
                    + defcon_awards * 2
                )
                points = max(0, points + rng.randint(-3, 6))
                bonus = max(0, int(points * rng.uniform(0.05, 0.16)))
                bps = int(points * rng.uniform(2.0, 3.4))

                base_price = {1: 4.3, 2: 4.3, 3: 4.8, 4: 5.2}[element_type]
                price = base_price + quality * {1: 1.6, 2: 2.4, 3: 6.5, 4: 6.8}[element_type]
                price = _round1(min(15.0, max(3.8, price + rng.uniform(-0.3, 0.3))))

                ppg = points / max(1, played_gws if nailed else max(1, starts))
                form = max(0.0, ppg * rng.uniform(0.6, 1.5))
                ownership = max(0.1, min(75.0, (points ** 1.35) / 30 * rng.uniform(0.4, 1.8)))

                # A handful of players carry injury flags.
                roll = rng.random()
                if roll < 0.03:
                    status, chance, news = "i", 0, "Hamstring injury - expected back in 3 weeks"
                elif roll < 0.07:
                    status, chance, news = "d", 75, "Knock - 75% chance of playing"
                else:
                    status, chance, news = "a", None, ""

                first = rng.choice(FIRST_NAMES)
                last = rng.choice(LAST_NAMES)
                web_name = last
                suffix = 2
                while (web_name, team["id"]) in used_names or web_name in {
                    n for n, _ in used_names
                }:
                    web_name = f"{last} {suffix}"
                    suffix += 1
                used_names.add((web_name, team["id"]))

                elements.append({
                    "id": element_id,
                    "web_name": web_name,
                    "first_name": first,
                    "second_name": last,
                    "team": team["id"],
                    "element_type": element_type,
                    "now_cost": int(round(price * 10)),
                    "total_points": points,
                    "form": f"{form:.1f}",
                    "points_per_game": f"{ppg:.1f}",
                    "selected_by_percent": f"{ownership:.1f}",
                    "minutes": minutes_played,
                    "starts": starts,
                    "goals_scored": goals,
                    "assists": assists,
                    "clean_sheets": clean_sheets,
                    "goals_conceded": goals_conceded,
                    "saves": saves,
                    "bonus": bonus,
                    "bps": bps,
                    "influence": f"{points * 2.1:.1f}",
                    "creativity": f"{assists * 30.0:.1f}",
                    "threat": f"{goals * 40.0:.1f}",
                    "ict_index": f"{points * 0.9:.1f}",
                    "expected_goals": f"{expected_goals:.2f}",
                    "expected_assists": f"{expected_assists:.2f}",
                    "expected_goal_involvements": f"{expected_goals + expected_assists:.2f}",
                    "expected_goals_conceded": f"{expected_goals_conceded:.2f}",
                    "defensive_contribution": defensive_contribution,
                    "status": status,
                    "chance_of_playing_next_round": chance,
                    "chance_of_playing_this_round": chance,
                    "ep_next": f"{max(0.0, ppg * rng.uniform(0.7, 1.3)):.1f}",
                    "ep_this": f"{max(0.0, ppg * rng.uniform(0.7, 1.3)):.1f}",
                    "news": news,
                    "transfers_in_event": rng.randint(0, 90000),
                    "transfers_out_event": rng.randint(0, 90000),
                })
                element_id += 1

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
    return bootstrap, fixtures


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/fixtures")
    target.mkdir(parents=True, exist_ok=True)
    bootstrap, fixtures = build_dataset()
    (target / "bootstrap.json").write_text(json.dumps(bootstrap, indent=1))
    (target / "fixtures.json").write_text(json.dumps(fixtures, indent=1))
    print(
        f"Wrote {len(bootstrap['elements'])} players, {len(bootstrap['teams'])} teams "
        f"and {len(fixtures)} fixtures to {target}/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
