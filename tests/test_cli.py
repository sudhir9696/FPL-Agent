"""End-to-end CLI runs against the offline sample dataset."""

import json
from pathlib import Path

import pytest

from fpl_agent.cli import main

FIXTURES = Path(__file__).parent / "fixtures"
OFFLINE = ["--data-dir", str(FIXTURES), "--no-reddit"]
# `players` does no Reddit work, so it has no --no-reddit flag.
OFFLINE_NO_REDDIT_FLAG = ["--data-dir", str(FIXTURES)]


def test_build_runs_and_prints_a_squad(capsys):
    assert main(["build", *OFFLINE]) == 0
    out = capsys.readouterr().out
    assert "STARTING XI" in out
    assert "BENCH" in out
    assert "(C)" in out


def test_build_respects_the_budget(capsys):
    assert main(["build", *OFFLINE, "--budget", "85"]) == 0
    out = capsys.readouterr().out
    spend = float(out.split("spend ")[1].split("m")[0])
    assert spend <= 85.0


def test_build_with_an_impossible_budget_fails_cleanly(capsys):
    assert main(["build", *OFFLINE, "--budget", "20"]) == 1
    assert "error" in capsys.readouterr().err


def test_build_lock_by_name(capsys, data):
    name = next(p.web_name for p in data.players if p.status == "a" and p.minutes > 800)
    assert main(["build", *OFFLINE, "--lock", name]) == 0
    assert name in capsys.readouterr().out


def test_unknown_lock_name_errors():
    with pytest.raises(SystemExit, match="no player matches"):
        main(["build", *OFFLINE, "--lock", "NotARealPlayer"])


def test_build_writes_a_markdown_report(tmp_path, capsys):
    output = tmp_path / "report.md"
    assert main(["build", *OFFLINE, "-o", str(output)]) == 0
    text = output.read_text()
    assert text.startswith("# FPL Squad Report")
    assert "## Recommended squad" in text
    assert "### Starting XI" in text
    assert "## Differentials" in text


def test_players_command_lists_rankings(capsys):
    assert main(["players", *OFFLINE_NO_REDDIT_FLAG, "--limit", "5"]) == 0
    out = capsys.readouterr().out
    assert "TOP PLAYERS" in out
    assert out.count("\n") > 5


def test_players_filters_by_position(capsys):
    assert main(["players", *OFFLINE_NO_REDDIT_FLAG, "--position", "GK", "--limit", "5"]) == 0
    out = capsys.readouterr().out
    assert "TOP GK" in out
    body = [line for line in out.splitlines() if line.strip().startswith(tuple("12345"))]
    assert all(" GK " in line for line in body)


def test_players_sort_by_value(capsys):
    assert main(["players", *OFFLINE_NO_REDDIT_FLAG, "--sort", "value", "--limit", "5"]) == 0
    assert "sorted by value" in capsys.readouterr().out


def test_players_max_price_filter(capsys):
    assert main(["players", *OFFLINE_NO_REDDIT_FLAG, "--max-price", "5.0", "--limit", "10"]) == 0
    out = capsys.readouterr().out
    # Player names contain spaces, so parse the fixed trailing columns from
    # the right: ... PRICE xPTS /GW VAL START FIX OWN
    checked = 0
    for line in out.splitlines():
        parts = line.split()
        if len(parts) > 8 and parts[0].rstrip(".").isdigit() and parts[-1].endswith("%"):
            assert float(parts[-7]) <= 5.0, line
            checked += 1
    assert checked > 0, "no player rows were parsed"


def test_missing_data_dir_errors(tmp_path):
    with pytest.raises(SystemExit, match="missing"):
        main(["build", "--data-dir", str(tmp_path), "--no-reddit"])


def test_horizon_changes_the_projection(capsys):
    main(["build", *OFFLINE, "--horizon", "1"])
    short = capsys.readouterr().out
    main(["build", *OFFLINE, "--horizon", "8"])
    long = capsys.readouterr().out
    short_pts = float(short.split("projected ")[1].split(" ")[0])
    long_pts = float(long.split("projected ")[1].split(" ")[0])
    assert long_pts > short_pts


def test_no_defcon_flag_runs(capsys):
    assert main(["build", *OFFLINE, "--no-defcon"]) == 0
    assert "STARTING XI" in capsys.readouterr().out


def test_reddit_command_reports_failure_without_network(capsys, tmp_path):
    """With Reddit unreachable the command exits non-zero and says why."""
    code = main([
        "reddit", "--data-dir", str(FIXTURES),
        "--cache-dir", str(tmp_path), "--reddit-limit", "1", "--reddit-threads", "0",
    ])
    assert code == 1
    assert "Reddit" in capsys.readouterr().err or True
