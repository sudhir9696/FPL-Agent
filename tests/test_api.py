import json

import pytest
import requests

from fpl_builder.api import FPLClient, FPLError, GameData


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


class FakeSession:
    def __init__(self, payload=None, exc=None):
        self.payload = payload if payload is not None else {"ok": True}
        self.exc = exc
        self.headers = {}
        self.calls = 0

    def get(self, url, timeout=None):
        self.calls += 1
        if self.exc:
            raise self.exc
        return FakeResponse(self.payload)


def test_get_caches_responses(tmp_path):
    session = FakeSession({"elements": []})
    client = FPLClient(cache_dir=tmp_path, session=session)
    client.get("bootstrap-static/")
    client.get("bootstrap-static/")
    assert session.calls == 1


def test_refresh_bypasses_the_cache(tmp_path):
    session = FakeSession({"elements": []})
    client = FPLClient(cache_dir=tmp_path, session=session)
    client.get("bootstrap-static/")
    client.get("bootstrap-static/", use_cache=False)
    assert session.calls == 2


def test_expired_cache_refetches(tmp_path):
    session = FakeSession({"elements": []})
    client = FPLClient(cache_dir=tmp_path, cache_ttl=0, session=session)
    client.get("bootstrap-static/")
    client.get("bootstrap-static/")
    assert session.calls == 2


def test_proxy_error_explains_that_fpl_needs_no_key(tmp_path):
    session = FakeSession(exc=requests.exceptions.ProxyError("blocked"))
    client = FPLClient(cache_dir=tmp_path, session=session)
    with pytest.raises(FPLError, match="public and needs no key"):
        client.get("bootstrap-static/")


def test_network_error_is_wrapped(tmp_path):
    session = FakeSession(exc=requests.exceptions.ConnectTimeout("timeout"))
    client = FPLClient(cache_dir=tmp_path, session=session)
    with pytest.raises(FPLError, match="failed"):
        client.get("bootstrap-static/")


def test_fixtures_endpoint_takes_an_event(tmp_path):
    captured = {}

    class Recorder(FakeSession):
        def get(self, url, timeout=None):
            captured["url"] = url
            return FakeResponse([])

    client = FPLClient(cache_dir=tmp_path, session=Recorder([]))
    client.fixtures(event=7)
    assert "fixtures/?event=7" in captured["url"]


def test_gamedata_indexes_players_and_teams(data):
    assert len(data.teams) == 20
    assert len(data.players) == len(data.players_by_id)
    assert data.players_by_id[data.players[0].id] is data.players[0]


def test_current_and_next_gameweek(data):
    assert data.current_gameweek >= 1
    assert data.next_gameweek == data.current_gameweek + 1


def test_upcoming_fixtures_respect_the_horizon(data):
    team_id = data.players[0].team
    fixtures = data.upcoming_fixtures(team_id, data.next_gameweek, 5)
    assert fixtures
    for fixture in fixtures:
        assert data.next_gameweek <= fixture.event < data.next_gameweek + 5
        assert fixture.involves(team_id)
        assert not fixture.finished


def test_upcoming_fixtures_empty_beyond_the_season(data):
    assert data.upcoming_fixtures(data.players[0].team, 500, 5) == []


def test_save_round_trips(tmp_path, data):
    data.save(tmp_path)
    reloaded = GameData.from_files(tmp_path / "bootstrap.json", tmp_path / "fixtures.json")
    assert len(reloaded.players) == len(data.players)
    assert len(reloaded.fixtures) == len(data.fixtures)
    assert reloaded.next_gameweek == data.next_gameweek
