"""Client for the public Fantasy Premier League REST API.

No API key or authentication is required for any endpoint used here -- they are
the same public JSON endpoints the official web app calls. A short-lived disk
cache keeps repeat runs fast and avoids hammering the servers.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import requests

from .models import Fixture, Player, Team

log = logging.getLogger(__name__)

BASE_URL = "https://fantasy.premierleague.com/api"

# The API rejects requests without a browser-like User-Agent.
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


class FPLError(RuntimeError):
    """Raised when the FPL API cannot be reached or returns an error."""


class FPLClient:
    """Thin, cached wrapper over the public FPL endpoints."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        cache_ttl: int = 3600,
        timeout: int = 30,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "fpl_builder"
        self.cache_ttl = cache_ttl
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # --- plumbing -----------------------------------------------------
    def _cache_path(self, key: str) -> Path:
        safe = key.strip("/").replace("/", "_").replace("?", "_").replace("=", "-")
        return self.cache_dir / f"{safe}.json"

    def _read_cache(self, key: str) -> Optional[Any]:
        path = self._cache_path(key)
        if not path.exists():
            return None
        if self.cache_ttl >= 0 and (time.time() - path.stat().st_mtime) > self.cache_ttl:
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def _write_cache(self, key: str, payload: Any) -> None:
        try:
            self._cache_path(key).write_text(json.dumps(payload))
        except OSError as exc:  # a broken cache must never break a run
            log.debug("could not cache %s: %s", key, exc)

    def get(self, endpoint: str, use_cache: bool = True) -> Any:
        """GET an endpoint (e.g. 'bootstrap-static/') and return parsed JSON."""
        if use_cache:
            cached = self._read_cache(endpoint)
            if cached is not None:
                log.debug("cache hit for %s", endpoint)
                return cached

        url = f"{BASE_URL}/{endpoint.lstrip('/')}"
        log.info("GET %s", url)
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.ProxyError as exc:
            raise FPLError(
                f"Could not reach {url} -- the connection was refused by a network "
                f"proxy, not by FPL. The FPL API itself is public and needs no key. "
                f"If you are on a restricted/corporate network or a sandboxed CI "
                f"runner, try again from an unrestricted network, or run with "
                f"--offline against cached/sample data. ({exc})"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise FPLError(f"Request to {url} failed: {exc}") from exc
        except ValueError as exc:
            raise FPLError(f"{url} did not return valid JSON: {exc}") from exc

        self._write_cache(endpoint, payload)
        return payload

    # --- endpoints ----------------------------------------------------
    def bootstrap_static(self, use_cache: bool = True) -> dict:
        """Everything: players (`elements`), teams, gameweeks, scoring rules."""
        return self.get("bootstrap-static/", use_cache=use_cache)

    def fixtures(self, event: Optional[int] = None, use_cache: bool = True) -> list:
        """All 380 fixtures with kickoff times and FDR, or one gameweek's."""
        endpoint = f"fixtures/?event={event}" if event else "fixtures/"
        return self.get(endpoint, use_cache=use_cache)

    def element_summary(self, player_id: int, use_cache: bool = True) -> dict:
        """Match-by-match history and upcoming fixtures for one player."""
        return self.get(f"element-summary/{player_id}/", use_cache=use_cache)

    def event_live(self, gameweek: int, use_cache: bool = True) -> dict:
        """Live per-player stats and points during an active gameweek."""
        # Live data goes stale in minutes, so cache it far more briefly.
        original_ttl, self.cache_ttl = self.cache_ttl, min(self.cache_ttl, 60)
        try:
            return self.get(f"event/{gameweek}/live/", use_cache=use_cache)
        finally:
            self.cache_ttl = original_ttl

    def entry(self, entry_id: int, use_cache: bool = True) -> dict:
        """Public summary of a manager's team."""
        return self.get(f"entry/{entry_id}/", use_cache=use_cache)

    def entry_picks(self, entry_id: int, gameweek: int, use_cache: bool = True) -> dict:
        """A manager's 15 picks for a given gameweek."""
        return self.get(f"entry/{entry_id}/event/{gameweek}/picks/", use_cache=use_cache)


class GameData:
    """Parsed, indexed view of bootstrap-static + fixtures."""

    def __init__(self, bootstrap: dict, fixtures: list) -> None:
        self.raw_bootstrap = bootstrap
        self.teams = {t["id"]: Team.from_api(t) for t in bootstrap.get("teams", [])}
        self.players = [Player.from_api(e) for e in bootstrap.get("elements", [])]
        self.players_by_id = {p.id: p for p in self.players}
        self.fixtures = [Fixture.from_api(f) for f in fixtures]
        self.events = bootstrap.get("events", [])

    @classmethod
    def load(cls, client: FPLClient, use_cache: bool = True) -> "GameData":
        return cls(client.bootstrap_static(use_cache), client.fixtures(use_cache=use_cache))

    @classmethod
    def from_files(cls, bootstrap_path: Path, fixtures_path: Path) -> "GameData":
        """Build from JSON dumped to disk -- used for offline and test runs."""
        return cls(
            json.loads(Path(bootstrap_path).read_text()),
            json.loads(Path(fixtures_path).read_text()),
        )

    @property
    def current_gameweek(self) -> int:
        """The gameweek in progress, else the next one, else 1."""
        for event in self.events:
            if event.get("is_current"):
                return event["id"]
        for event in self.events:
            if event.get("is_next"):
                return event["id"]
        unfinished = [e["id"] for e in self.events if not e.get("finished")]
        return min(unfinished) if unfinished else 1

    @property
    def next_gameweek(self) -> int:
        for event in self.events:
            if event.get("is_next"):
                return event["id"]
        return self.current_gameweek

    def team_games_played(self, team_id: int) -> int:
        return sum(1 for f in self.fixtures if f.finished and f.involves(team_id))

    def upcoming_fixtures(self, team_id: int, start_gw: int, horizon: int) -> list:
        """Fixtures for a team across [start_gw, start_gw+horizon), blanks and
        doubles included -- a double gameweek naturally yields two entries."""
        end_gw = start_gw + horizon
        return [
            f
            for f in self.fixtures
            if f.event is not None
            and start_gw <= f.event < end_gw
            and f.involves(team_id)
            and not f.finished
        ]

    def save(self, directory: Path) -> None:
        """Persist the raw payloads so a later run can work fully offline."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "bootstrap.json").write_text(json.dumps(self.raw_bootstrap))
        (directory / "fixtures.json").write_text(
            json.dumps([f.__dict__ for f in self.fixtures])
        )
