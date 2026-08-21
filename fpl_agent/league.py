"""Mini-league analysis: who owns what, and where your edge is.

Global ownership (`selected_by_percent`) is the wrong yardstick when you are
chasing 30 people in a work league. What matters is *league* ownership: a
player owned by 4% globally but 60% of your rivals is template to you, and a
70%-owned global template you do not have is a standing risk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional

log = logging.getLogger(__name__)


@dataclass
class LeagueEntry:
    """One manager's row in the standings, plus their picks once loaded."""

    entry: int
    entry_name: str
    player_name: str
    rank: int
    total: int
    event_total: int = 0
    last_rank: int = 0
    picks: list = field(default_factory=list)      # player ids
    captain: Optional[int] = None

    @classmethod
    def from_api(cls, raw: dict) -> "LeagueEntry":
        return cls(
            entry=raw.get("entry") or 0,
            entry_name=raw.get("entry_name", "") or "",
            player_name=raw.get("player_name", "") or "",
            rank=int(raw.get("rank") or 0),
            total=int(raw.get("total") or 0),
            event_total=int(raw.get("event_total") or 0),
            last_rank=int(raw.get("last_rank") or 0),
        )


@dataclass
class OwnershipRow:
    """How widely one player is owned inside the league."""

    player_id: int
    name: str
    position: str
    cost: float
    league_owners: int
    league_size: int
    captained_by: int = 0
    global_ownership: float = 0.0
    expected_points: float = 0.0

    @property
    def league_ownership(self) -> float:
        if not self.league_size:
            return 0.0
        return round(100.0 * self.league_owners / self.league_size, 1)

    @property
    def edge(self) -> float:
        """League ownership minus global ownership, in percentage points.

        Positive means your league is more loaded on this player than the game
        at large -- owning them protects you locally but wins you nothing.
        """
        return round(self.league_ownership - self.global_ownership, 1)

    @property
    def effective_ownership(self) -> float:
        """League ownership plus captaincy, the number that actually moves rank."""
        if not self.league_size:
            return 0.0
        return round(
            100.0 * (self.league_owners + self.captained_by) / self.league_size, 1
        )


def find_entry(entries: Iterable[LeagueEntry], needle: str) -> Optional[LeagueEntry]:
    """Look up a manager by team name or player name, case-insensitively."""
    needle = needle.strip().lower()
    candidates = list(entries)
    for entry in candidates:
        if entry.entry_name.lower() == needle or entry.player_name.lower() == needle:
            return entry
    for entry in candidates:
        if needle in entry.entry_name.lower() or needle in entry.player_name.lower():
            return entry
    return None


def load_picks(
    client, entries: Iterable[LeagueEntry], gameweek: int, use_cache: bool = True
) -> list:
    """Populate each entry's picks. Entries that fail to load are skipped."""
    from .api import FPLError

    loaded = []
    for entry in entries:
        try:
            payload = client.entry_picks(entry.entry, gameweek, use_cache=use_cache)
        except FPLError as exc:
            log.warning("could not load picks for entry %s: %s", entry.entry, exc)
            continue
        picks = payload.get("picks", [])
        entry.picks = [p["element"] for p in picks if p.get("element")]
        captains = [p["element"] for p in picks if p.get("is_captain")]
        entry.captain = captains[0] if captains else None
        loaded.append(entry)
    return loaded


def ownership_table(
    entries: Iterable[LeagueEntry], data, projections: Optional[Iterable] = None
) -> list:
    """Ownership of every player across the league, most-owned first."""
    entries = [e for e in entries if e.picks]
    league_size = len(entries)
    if not league_size:
        return []

    owners: dict = {}
    captains: dict = {}
    for entry in entries:
        for player_id in set(entry.picks):
            owners[player_id] = owners.get(player_id, 0) + 1
        if entry.captain:
            captains[entry.captain] = captains.get(entry.captain, 0) + 1

    points = {}
    if projections:
        points = {p.player.id: p.expected_points for p in projections}

    rows = []
    for player_id, count in owners.items():
        player = data.players_by_id.get(player_id)
        if player is None:
            continue
        rows.append(OwnershipRow(
            player_id=player_id,
            name=player.web_name,
            position=player.position,
            cost=player.cost,
            league_owners=count,
            league_size=league_size,
            captained_by=captains.get(player_id, 0),
            global_ownership=player.selected_by,
            expected_points=points.get(player_id, 0.0),
        ))

    rows.sort(key=lambda r: (-r.league_owners, -r.expected_points))
    return rows


def compare_to_league(
    my_entry: LeagueEntry, rows: Iterable[OwnershipRow], min_rival_ownership: float = 40.0
) -> dict:
    """Position one squad against the rest of the league.

    * `my_differentials` - you own them, most rivals do not
    * `my_risks`         - most rivals own them, you do not
    * `shared_template`  - you and most of the league both own them
    """
    mine = set(my_entry.picks)
    rows = list(rows)

    differentials = [
        r for r in rows if r.player_id in mine and r.league_ownership <= 100 - min_rival_ownership
    ]
    risks = [
        r for r in rows
        if r.player_id not in mine and r.league_ownership >= min_rival_ownership
    ]
    template = [
        r for r in rows if r.player_id in mine and r.league_ownership >= min_rival_ownership
    ]

    differentials.sort(key=lambda r: -r.expected_points)
    risks.sort(key=lambda r: (-r.effective_ownership, -r.expected_points))
    template.sort(key=lambda r: -r.league_ownership)

    return {
        "my_differentials": differentials,
        "my_risks": risks,
        "shared_template": template,
    }
