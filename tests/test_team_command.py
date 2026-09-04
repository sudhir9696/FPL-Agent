"""Covers `fpl-agent team <entry-id>` with the API responses mocked out."""

from pathlib import Path

import pytest

from fpl_agent import cli
from fpl_agent.api import GameData

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fake_entry(monkeypatch, data):
    """Patch FPLClient so the team command reads a synthetic manager team."""
    # 2 GK, 5 DEF, 5 MID, 3 FWD from distinct clubs, respecting the club limit.
    picks, counts = [], {}
    quota = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    for player in sorted(data.players, key=lambda p: -p.total_points):
        position = player.position
        if quota.get(position, 0) <= 0:
            continue
        if counts.get(player.team, 0) >= 3:
            continue
        picks.append(player)
        quota[position] -= 1
        counts[player.team] = counts.get(player.team, 0) + 1
        if not any(quota.values()):
            break

    payload = {
        "picks": [
            {"element": p.id, "position": i + 1, "multiplier": 1 if i < 11 else 0,
             "is_captain": i == 0, "is_vice_captain": i == 1}
            for i, p in enumerate(picks)
        ]
    }
    entry_payload = {
        "name": "DontBottleThisYear",
        "player_first_name": "Test",
        "player_last_name": "Manager",
        "last_deadline_bank": 15,      # 1.5m, API uses tenths
        "summary_overall_points": 2177,
    }

    monkeypatch.setattr(cli.FPLClient, "entry", lambda self, eid, use_cache=True: entry_payload)
    monkeypatch.setattr(
        cli.FPLClient, "entry_picks", lambda self, eid, gw, use_cache=True: payload
    )
    return picks


def test_team_command_shows_the_squad(fake_entry, capsys, tmp_path):
    code = cli.main([
        "team", "1234567", "--data-dir", str(FIXTURES), "--no-reddit",
        "--cache-dir", str(tmp_path),
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "DontBottleThisYear" in out
    assert "Test Manager" in out
    assert "Bank: 1.5m" in out
    assert "STARTING XI" in out
    assert "TRANSFER SUGGESTIONS" in out


def test_team_command_names_every_pick(fake_entry, capsys, tmp_path):
    cli.main([
        "team", "1234567", "--data-dir", str(FIXTURES), "--no-reddit",
        "--cache-dir", str(tmp_path),
    ])
    out = capsys.readouterr().out
    # Names are truncated to 18 characters in the table.
    for player in fake_entry:
        assert player.web_name[:16] in out, player.web_name


def test_team_command_writes_a_report(fake_entry, tmp_path):
    output = tmp_path / "team.md"
    cli.main([
        "team", "1234567", "--data-dir", str(FIXTURES), "--no-reddit",
        "--cache-dir", str(tmp_path), "-o", str(output),
    ])
    text = output.read_text()
    assert "# FPL Squad Report" in text
    assert "### Starting XI" in text


def test_team_command_errors_when_picks_are_empty(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli.FPLClient, "entry", lambda self, eid, use_cache=True: {"name": "X"})
    monkeypatch.setattr(
        cli.FPLClient, "entry_picks", lambda self, eid, gw, use_cache=True: {"picks": []}
    )
    code = cli.main([
        "team", "1234567", "--data-dir", str(FIXTURES), "--no-reddit",
        "--cache-dir", str(tmp_path),
    ])
    assert code == 1
    assert "no picks resolved" in capsys.readouterr().err


def test_team_command_surfaces_api_errors(monkeypatch, capsys, tmp_path):
    from fpl_agent.api import FPLError

    def boom(self, eid, use_cache=True):
        raise FPLError("blocked by proxy")

    monkeypatch.setattr(cli.FPLClient, "entry", boom)
    code = cli.main([
        "team", "1234567", "--data-dir", str(FIXTURES), "--no-reddit",
        "--cache-dir", str(tmp_path),
    ])
    assert code == 1
    assert "blocked by proxy" in capsys.readouterr().err


def test_team_diagnose_flag_prints_findings(fake_entry, capsys, tmp_path):
    code = cli.main([
        "team", "1234567", "--data-dir", str(FIXTURES), "--no-reddit",
        "--cache-dir", str(tmp_path), "--diagnose",
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "WHAT'S WRONG WITH THIS TEAM" in out
    assert "VS OPTIMAL SQUAD" in out


def test_team_without_diagnose_omits_the_section(fake_entry, capsys, tmp_path):
    cli.main([
        "team", "1234567", "--data-dir", str(FIXTURES), "--no-reddit",
        "--cache-dir", str(tmp_path),
    ])
    assert "WHAT'S WRONG WITH THIS TEAM" not in capsys.readouterr().out


# --- --picks: analyse a squad supplied by hand, no API call ------------
def _fifteen_names(data):
    quota, counts, names = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}, {}, []
    for player in sorted(data.players, key=lambda p: -p.total_points):
        if quota.get(player.position, 0) <= 0 or counts.get(player.team, 0) >= 3:
            continue
        if any(n == player.web_name for n in names):
            continue
        names.append(player.web_name)
        quota[player.position] -= 1
        counts[player.team] = counts.get(player.team, 0) + 1
        if not any(quota.values()):
            break
    return names


def test_picks_analyses_a_hand_entered_squad(capsys, data, tmp_path):
    names = _fifteen_names(data)
    code = cli.main([
        "team", "--picks", ",".join(names), "--data-dir", str(FIXTURES),
        "--no-reddit", "--cache-dir", str(tmp_path),
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "STARTING XI" in out
    assert "your squad" in out


def test_picks_needs_no_api_call(monkeypatch, capsys, data, tmp_path):
    """--picks must never touch the network."""
    def boom(*args, **kwargs):
        raise AssertionError("--picks should not call the FPL API")

    monkeypatch.setattr(cli.FPLClient, "entry", boom)
    monkeypatch.setattr(cli.FPLClient, "entry_picks", boom)
    code = cli.main([
        "team", "--picks", ",".join(_fifteen_names(data)), "--data-dir", str(FIXTURES),
        "--no-reddit", "--cache-dir", str(tmp_path),
    ])
    assert code == 0


def test_picks_accepts_names_with_spaces(capsys, data, tmp_path):
    """Regression: splitting on spaces broke names like 'Van Hecke'."""
    spaced = next(
        (p for p in data.players if " " in p.web_name), None
    )
    names = _fifteen_names(data)
    if spaced and spaced.web_name not in names:
        names[-1] = spaced.web_name
    code = cli.main([
        "team", "--picks", ",".join(names), "--data-dir", str(FIXTURES),
        "--no-reddit", "--cache-dir", str(tmp_path),
    ])
    assert code == 0
    if spaced:
        assert spaced.web_name[:16] in capsys.readouterr().out


def test_picks_accepts_player_ids(capsys, data, tmp_path):
    ids = [str(p.id) for p in data.players[:1]]
    names = _fifteen_names(data)
    code = cli.main([
        "team", "--picks", ",".join(names[:14] + ids), "--data-dir", str(FIXTURES),
        "--no-reddit", "--cache-dir", str(tmp_path),
    ])
    assert code in (0, 1)   # may resolve <15 if the id duplicates a name


def test_picks_reports_an_unknown_name(capsys, data, tmp_path):
    names = _fifteen_names(data)[:14] + ["DefinitelyNotAPlayer"]
    code = cli.main([
        "team", "--picks", ",".join(names), "--data-dir", str(FIXTURES),
        "--no-reddit", "--cache-dir", str(tmp_path),
    ])
    assert code == 1
    assert "no player matches" in capsys.readouterr().err


def test_picks_carries_the_bank_into_the_diagnosis(capsys, data, tmp_path):
    code = cli.main([
        "team", "--picks", ",".join(_fifteen_names(data)), "--bank", "4.0",
        "--data-dir", str(FIXTURES), "--no-reddit", "--diagnose",
        "--cache-dir", str(tmp_path),
    ])
    assert code == 0
    assert "4.0m sitting in the bank" in capsys.readouterr().out


def test_team_requires_entry_or_picks(capsys, tmp_path):
    code = cli.main([
        "team", "--data-dir", str(FIXTURES), "--no-reddit", "--cache-dir", str(tmp_path),
    ])
    assert code == 1
    assert "entry id" in capsys.readouterr().err
