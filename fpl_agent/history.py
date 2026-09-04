"""Per-gameweek history, and the rolling form derived from it.

`bootstrap-static` carries only season totals, so by GW20 a player who was
excellent in August and anonymous since reads exactly like one who was the
reverse. FPL's own `form` field is a blunt instrument for this: it averages
total points over 30 days, which folds fixture luck and bonus into a single
number and says nothing about the underlying rates.

`element-summary/{id}/history` returns a row per gameweek carrying the same
Opta fields as the season totals. Aggregating a recent window of those gives a
form measure on the same footing as the season rates, so the two can be
blended rather than one overriding the other.

The vaastav GitHub mirror carries the same data and is the fallback when the
API is unreachable, but it lags the live API by gameweeks -- so it is not the
source here.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

log = logging.getLogger(__name__)

# Aggregated over the window; these names match the API's.
_FIELDS = (
    "minutes",
    "starts",
    "total_points",
    "expected_goals",
    "expected_assists",
    "expected_goals_conceded",
    "defensive_contribution",
    "bonus",
    "saves",
)


def _f(value, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class GameweekRow:
    """One player's return in one gameweek."""

    round: int
    minutes: float
    starts: float
    total_points: float
    expected_goals: float
    expected_assists: float
    expected_goals_conceded: float
    defensive_contribution: float
    bonus: float
    saves: float

    @classmethod
    def from_api(cls, raw: dict) -> "GameweekRow":
        return cls(round=int(_f(raw.get("round"))),
                   **{f: _f(raw.get(f)) for f in _FIELDS})


@dataclass
class Window:
    """Totals over a run of gameweeks, and the per-90 rates they imply."""

    gameweeks: int
    minutes: float
    starts: float
    total_points: float
    expected_goals: float
    expected_assists: float
    expected_goals_conceded: float
    defensive_contribution: float
    bonus: float
    saves: float

    def per90(self, field: str) -> float:
        if self.minutes <= 0:
            return 0.0
        return getattr(self, field) * 90.0 / self.minutes

    @property
    def start_rate(self) -> float:
        return self.starts / self.gameweeks if self.gameweeks else 0.0


class PlayerHistory:
    """Every gameweek row the API holds for one player, oldest first."""

    def __init__(self, rows: Iterable[dict]) -> None:
        self.rows: List[GameweekRow] = sorted(
            (GameweekRow.from_api(r) for r in rows), key=lambda r: r.round
        )

    def __len__(self) -> int:
        return len(self.rows)

    def window(self, gameweeks: int) -> Optional[Window]:
        """Aggregate the most recent `gameweeks` rows, or None if there are none.

        Gameweeks with no minutes stay in the window. A player who did not
        feature had no chance to score, and dropping those rows would flatter
        someone who has lost their place into looking merely rested.
        """
        recent = self.rows[-gameweeks:] if gameweeks > 0 else self.rows
        if not recent:
            return None
        return Window(gameweeks=len(recent),
                      **{f: sum(getattr(r, f) for r in recent) for f in _FIELDS})


class HistoryStore:
    """Per-gameweek history for a set of players, fetched once and reused."""

    def __init__(self, histories: Optional[Dict[int, PlayerHistory]] = None) -> None:
        self.histories: Dict[int, PlayerHistory] = histories or {}

    def __contains__(self, player_id: int) -> bool:
        return player_id in self.histories

    def __len__(self) -> int:
        return len(self.histories)

    def get(self, player_id: int) -> Optional[PlayerHistory]:
        return self.histories.get(player_id)

    def window(self, player_id: int, gameweeks: int) -> Optional[Window]:
        history = self.histories.get(player_id)
        return history.window(gameweeks) if history else None

    @classmethod
    def load(cls, client, player_ids: Iterable[int], workers: int = 8,
             use_cache: bool = True) -> "HistoryStore":
        """Fetch `element-summary` per player.

        One request each, so the caller decides who is worth the calls;
        responses are disk-cached by the client. A player whose fetch fails is
        simply absent, and the model falls back to season totals for them.
        """
        ids = list(dict.fromkeys(player_ids))
        if not ids:
            return cls()

        def fetch(player_id: int):
            try:
                payload = client.element_summary(player_id, use_cache=use_cache)
                return player_id, PlayerHistory(payload.get("history", []))
            except Exception as exc:  # one missing player must not sink the run
                log.debug("no history for %s: %s", player_id, exc)
                return player_id, None

        histories: Dict[int, PlayerHistory] = {}
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for player_id, history in pool.map(fetch, ids):
                if history is not None and len(history):
                    histories[player_id] = history
        log.info("loaded gameweek history for %d/%d players", len(histories), len(ids))
        return cls(histories)
