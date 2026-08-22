"""European competition congestion.

The FPL API knows only about the Premier League -- it carries no field for
UEFA competitions anywhere. So the midweek calendar is kept alongside the code
in `data/european_fixtures.json`, sourced from uefa.com.

What this is for is rotation, not fatigue. A side that played on Thursday
night and kicks off again on Saturday has had two days; managers rest people.
That turnaround is why Europa League clubs are far more rotation-prone for a
weekend fixture than Champions League clubs, who play Tuesday or Wednesday and
get three or four days. The multiplier below scales a player's start
probability by how much rest their club actually had.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent / "data" / "european_fixtures.json"

# Start-probability multiplier by days between the European tie and the
# Premier League kick-off. Two days is the Thursday-to-Saturday case and the
# one that actually moves team sheets; by five days the effect is gone.
REST_DAY_MULTIPLIER = {0: 0.70, 1: 0.72, 2: 0.75, 3: 0.87, 4: 0.95}
MAX_REST_DAYS_CONSIDERED = 4


class EuropeanCalendar:
    """Which clubs play midweek in Europe, and how much rest that leaves."""

    def __init__(self, payload: Optional[dict] = None) -> None:
        self.payload = payload if payload is not None else self._load()
        self.competitions = self.payload.get("competitions", {})
        self.teams_verified = bool(self.payload.get("_teams_verified"))

    @staticmethod
    def _load() -> dict:
        try:
            return json.loads(DATA_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("could not read %s (%s); European congestion disabled",
                        DATA_PATH, exc)
            return {}

    @property
    def active(self) -> bool:
        """False until at least one competition has teams filled in."""
        return any(c.get("teams") for c in self.competitions.values())

    def competition_for(self, team_short_name: str) -> Optional[str]:
        for code, comp in self.competitions.items():
            if team_short_name in (comp.get("teams") or []):
                return code
        return None

    def _matchday_dates(self, code: str) -> list:
        dates = []
        for md in self.competitions.get(code, {}).get("matchdays", []):
            for raw in md.get("dates", []):
                try:
                    dates.append(dt.date.fromisoformat(raw))
                except ValueError:
                    log.warning("bad date %r in %s matchday %s", raw, code, md.get("md"))
        return sorted(dates)

    def rest_days_before(self, team_short_name: str, kickoff: dt.datetime) -> Optional[int]:
        """Days between this club's last European tie and `kickoff`.

        Returns None when the club is not in Europe, or played nothing recent.
        Which night of a matchday window a club plays is only known once the
        draw is made, so the latest date in the window is assumed -- the
        conservative reading, since it implies the least rest.
        """
        code = self.competition_for(team_short_name)
        if code is None:
            return None
        match_date = kickoff.date()
        candidates = [d for d in self._matchday_dates(code)
                      if 0 <= (match_date - d).days <= MAX_REST_DAYS_CONSIDERED]
        if not candidates:
            return None
        return (match_date - max(candidates)).days

    def rotation_multiplier(self, team_short_name: str, kickoff: Optional[dt.datetime]) -> tuple:
        """(multiplier, note). Multiplier is 1.0 when there is no congestion."""
        if kickoff is None or not self.active:
            return 1.0, None
        rest = self.rest_days_before(team_short_name, kickoff)
        if rest is None:
            return 1.0, None
        multiplier = REST_DAY_MULTIPLIER.get(rest, 1.0)
        if multiplier >= 1.0:
            return 1.0, None
        code = self.competition_for(team_short_name)
        return multiplier, f"{code} {rest}d turnaround"


def parse_kickoff(raw: Optional[str]) -> Optional[dt.datetime]:
    """Parse an FPL kickoff timestamp ('2026-09-12T14:00:00Z')."""
    if not raw:
        return None
    try:
        return dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
