"""Mini-league analysis, with the standings and picks endpoints mocked."""

from pathlib import Path

import pytest

from fpl_agent import cli
from fpl_agent.api import FPLError
from fpl_agent.league import (
    LeagueEntry, compare_to_league, find_entry, load_picks, ownership_table,
)

FIXTURES = Path(__file__).parent / "fixtures"


def make_entry(entry, name, manager, rank, total=2000, picks=None, captain=None):
    e = LeagueEntry(entry=entry, entry_name=name, player_name=manager,
                    rank=rank, total=total)
    e.picks = picks or []
    e.captain = captain
    return e


# --- lookup ----------------------------------------------------------
def test_find_entry_by_exact_team_name():
    entries = [make_entry(1, "DontBottleThisYear", "sudhir didugu", 1),
               make_entry(2, "Other Team", "Someone Else", 2)]
    assert find_entry(entries, "DontBottleThisYear").entry == 1


def test_find_entry_is_case_insensitive_and_partial():
    entries = [make_entry(1, "DontBottleThisYear", "sudhir didugu", 1)]
    assert find_entry(entries, "dontbottlethisyear").entry == 1
    assert find_entry(entries, "bottle").entry == 1
    assert find_entry(entries, "sudhir").entry == 1


def test_find_entry_prefers_an_exact_match():
    entries = [make_entry(1, "Team A Reserve", "X", 1), make_entry(2, "Team A", "Y", 2)]
    assert find_entry(entries, "Team A").entry == 2


def test_find_entry_returns_none_when_absent():
    assert find_entry([make_entry(1, "A", "B", 1)], "nobody") is None


def test_entry_parses_from_api():
    entry = LeagueEntry.from_api({
        "entry": 1234567, "entry_name": "DontBottleThisYear",
        "player_name": "sudhir didugu", "rank": 3, "last_rank": 5,
        "total": 2177, "event_total": 61,
    })
    assert entry.entry == 1234567
    assert entry.rank == 3
    assert entry.total == 2177


def test_entry_tolerates_missing_fields():
    entry = LeagueEntry.from_api({})
    assert entry.entry == 0 and entry.rank == 0


# --- ownership -------------------------------------------------------
@pytest.fixture
def league_squads(data):
    """Three managers: two share a core, one is contrarian."""
    ids = [p.id for p in data.players[:30]]
    core, fringe = ids[:15], ids[15:30]
    return [
        make_entry(1, "Alpha", "A", 1, picks=core, captain=core[0]),
        make_entry(2, "Beta", "B", 2, picks=core, captain=core[0]),
        make_entry(3, "Gamma", "C", 3, picks=fringe, captain=fringe[0]),
    ]


def test_ownership_counts_owners(data, league_squads):
    rows = ownership_table(league_squads, data)
    by_id = {r.player_id: r for r in rows}
    shared = league_squads[0].picks[0]
    assert by_id[shared].league_owners == 2
    assert by_id[shared].league_size == 3
    assert by_id[shared].league_ownership == pytest.approx(66.7, abs=0.1)


def test_ownership_is_sorted_by_owners(data, league_squads):
    rows = ownership_table(league_squads, data)
    assert [r.league_owners for r in rows] == sorted(
        (r.league_owners for r in rows), reverse=True
    )


def test_effective_ownership_includes_captaincy(data, league_squads):
    rows = ownership_table(league_squads, data)
    captained = league_squads[0].captain
    row = next(r for r in rows if r.player_id == captained)
    # 2 owners + 2 captains over 3 managers.
    assert row.effective_ownership > row.league_ownership


def test_edge_compares_league_to_global(data, league_squads):
    rows = ownership_table(league_squads, data)
    for row in rows:
        assert row.edge == pytest.approx(row.league_ownership - row.global_ownership, abs=0.1)


def test_ownership_attaches_projections(data, league_squads, projections):
    rows = ownership_table(league_squads, data, projections)
    assert any(r.expected_points > 0 for r in rows)


def test_ownership_empty_without_picks(data):
    assert ownership_table([make_entry(1, "A", "B", 1)], data) == []


def test_ownership_ignores_unknown_player_ids(data, league_squads):
    league_squads[0].picks = league_squads[0].picks + [999999]
    rows = ownership_table(league_squads, data)
    assert all(r.player_id != 999999 for r in rows)


# --- comparison ------------------------------------------------------
def test_comparison_splits_differentials_risks_and_template(data, league_squads):
    rows = ownership_table(league_squads, data)
    gamma = league_squads[2]                       # the contrarian
    result = compare_to_league(gamma, rows, min_rival_ownership=50.0)

    mine = set(gamma.picks)
    assert all(r.player_id in mine for r in result["my_differentials"])
    assert all(r.player_id not in mine for r in result["my_risks"])
    assert all(r.player_id in mine for r in result["shared_template"])
    # The core that Alpha and Beta share is a risk Gamma does not hold.
    assert result["my_risks"]


def test_comparison_for_a_template_manager(data, league_squads):
    rows = ownership_table(league_squads, data)
    result = compare_to_league(league_squads[0], rows, min_rival_ownership=50.0)
    assert result["shared_template"]


def test_risks_are_sorted_by_effective_ownership(data, league_squads):
    rows = ownership_table(league_squads, data)
    risks = compare_to_league(league_squads[2], rows, min_rival_ownership=50.0)["my_risks"]
    assert [r.effective_ownership for r in risks] == sorted(
        (r.effective_ownership for r in risks), reverse=True
    )


# --- picks loading ---------------------------------------------------
class FakeClient:
    def __init__(self, picks_by_entry, fail_for=()):
        self.picks_by_entry = picks_by_entry
        self.fail_for = set(fail_for)

    def entry_picks(self, entry_id, gameweek, use_cache=True):
        if entry_id in self.fail_for:
            raise FPLError("boom")
        return {"picks": [
            {"element": pid, "is_captain": i == 0}
            for i, pid in enumerate(self.picks_by_entry[entry_id])
        ]}


def test_load_picks_populates_entries():
    entries = [make_entry(1, "A", "x", 1), make_entry(2, "B", "y", 2)]
    client = FakeClient({1: [10, 11, 12], 2: [20, 21]})
    loaded = load_picks(client, entries, gameweek=5)
    assert len(loaded) == 2
    assert loaded[0].picks == [10, 11, 12]
    assert loaded[0].captain == 10


def test_load_picks_skips_failures():
    entries = [make_entry(1, "A", "x", 1), make_entry(2, "B", "y", 2)]
    client = FakeClient({1: [10]}, fail_for=[2])
    loaded = load_picks(client, entries, gameweek=5)
    assert [e.entry for e in loaded] == [1]


# --- CLI -------------------------------------------------------------
@pytest.fixture
def fake_league(monkeypatch, data):
    meta = {"id": 804829, "name": "Test League"}
    standings = [
        {"entry": 100 + i, "entry_name": f"Team {i}", "player_name": f"Manager {i}",
         "rank": i + 1, "last_rank": i + 2, "total": 2200 - i * 10, "event_total": 60 - i}
        for i in range(5)
    ]
    standings[2]["entry_name"] = "DontBottleThisYear"
    standings[2]["player_name"] = "sudhir didugu"

    ids = [p.id for p in data.players[:30]]
    picks = {100 + i: ids[:15] if i < 3 else ids[15:30] for i in range(5)}

    monkeypatch.setattr(cli.FPLClient, "league_entries",
                        lambda self, lid, max_entries=50, use_cache=True: (meta, standings))
    monkeypatch.setattr(
        cli.FPLClient, "entry_picks",
        lambda self, eid, gw, use_cache=True: {
            "picks": [{"element": pid, "is_captain": i == 0}
                      for i, pid in enumerate(picks[eid])]
        },
    )
    return meta, standings


def test_league_command_prints_standings(fake_league, capsys, tmp_path):
    code = cli.main(["league", "804829", "--data-dir", str(FIXTURES),
                     "--cache-dir", str(tmp_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "Test League" in out
    assert "DontBottleThisYear" in out
    assert "ENTRY ID" in out


def test_league_command_finds_you_and_reports_entry_id(fake_league, capsys, tmp_path):
    cli.main(["league", "804829", "--data-dir", str(FIXTURES),
              "--cache-dir", str(tmp_path), "--me", "DontBottleThisYear"])
    out = capsys.readouterr().out
    assert "Found you: 'DontBottleThisYear'" in out
    assert "entry id 102" in out


def test_league_command_warns_when_you_are_not_found(fake_league, capsys, tmp_path):
    cli.main(["league", "804829", "--data-dir", str(FIXTURES),
              "--cache-dir", str(tmp_path), "--me", "NotInThisLeague"])
    assert "not found" in capsys.readouterr().err


def test_league_analyze_shows_ownership_and_comparison(fake_league, capsys, tmp_path):
    code = cli.main(["league", "804829", "--data-dir", str(FIXTURES),
                     "--cache-dir", str(tmp_path), "--me", "DontBottleThisYear",
                     "--analyze", "5"])
    assert code == 0
    out = capsys.readouterr().out
    assert "LEAGUE OWNERSHIP" in out
    assert "YOUR POSITION IN THE LEAGUE" in out
    assert "Your differentials" in out
    assert "Your risks" in out


def test_league_without_analyze_hints_at_it(fake_league, capsys, tmp_path):
    cli.main(["league", "804829", "--data-dir", str(FIXTURES), "--cache-dir", str(tmp_path)])
    assert "--analyze" in capsys.readouterr().out


def test_league_command_surfaces_api_errors(monkeypatch, capsys, tmp_path):
    def boom(self, lid, max_entries=50, use_cache=True):
        raise FPLError("blocked by proxy")

    monkeypatch.setattr(cli.FPLClient, "league_entries", boom)
    code = cli.main(["league", "804829", "--data-dir", str(FIXTURES),
                     "--cache-dir", str(tmp_path)])
    assert code == 1
    assert "blocked by proxy" in capsys.readouterr().err


def test_league_command_handles_empty_league(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli.FPLClient, "league_entries",
                        lambda self, lid, max_entries=50, use_cache=True: ({"id": 1}, []))
    code = cli.main(["league", "804829", "--data-dir", str(FIXTURES),
                     "--cache-dir", str(tmp_path)])
    assert code == 1
    assert "no entries" in capsys.readouterr().err
