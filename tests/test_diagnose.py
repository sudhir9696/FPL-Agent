"""Squad diagnosis: what is wrong with a team and how far off optimal it is."""

import pytest

from fpl_agent.diagnose import diagnose_squad
from fpl_agent.optimizer import optimize_squad


def titles(result):
    return " | ".join(f.title for f in result["findings"])


def severities(result):
    return {f.severity for f in result["findings"]}


@pytest.fixture(scope="module")
def optimal(projections):
    return optimize_squad(projections, budget=100.0)


def _swap_in(squad_picks, projections, replacement_filter, position):
    """Replace one player of `position` with the first matching candidate."""
    picks = list(squad_picks)
    victim = next(p for p in picks if p.player.position == position)
    owned = {p.player.id for p in picks}
    replacement = next(
        p for p in projections
        if p.player.position == position and p.player.id not in owned
        and replacement_filter(p)
    )
    picks[picks.index(victim)] = replacement
    return picks, replacement


def test_optimal_squad_has_no_gap(optimal, projections):
    result = diagnose_squad(optimal.picks, projections, bank=0.0, horizon=5, budget=100.0)
    assert "behind the optimal squad" not in titles(result)


def test_injured_player_is_critical(optimal, projections, data):
    picks, injured = _swap_in(
        optimal.picks, projections, lambda p: p.player.status == "i", "DEF"
    )
    result = diagnose_squad(picks, projections, bank=0.0, horizon=5, budget=100.0)
    assert "critical" in severities(result)
    assert "cannot play" in titles(result)
    finding = next(f for f in result["findings"] if "cannot play" in f.title)
    assert any(injured.player.web_name in entry for entry in finding.players)


def test_injury_finding_estimates_a_points_cost(optimal, projections):
    picks, _ = _swap_in(optimal.picks, projections, lambda p: p.player.status == "i", "DEF")
    result = diagnose_squad(picks, projections, bank=0.0, horizon=5, budget=100.0)
    finding = next(f for f in result["findings"] if "cannot play" in f.title)
    assert finding.cost > 0


def test_doubtful_player_is_medium_not_critical(optimal, projections):
    picks, _ = _swap_in(
        optimal.picks, projections,
        lambda p: p.player.status == "d" and p.player.availability > 0, "MID",
    )
    result = diagnose_squad(picks, projections, bank=0.0, horizon=5, budget=100.0)
    assert "fitness doubt" in titles(result)


def test_idle_bank_is_flagged(optimal, projections):
    result = diagnose_squad(optimal.picks, projections, bank=5.0, horizon=5, budget=100.0)
    assert "sitting in the bank" in titles(result)


def test_small_bank_is_not_flagged(optimal, projections):
    result = diagnose_squad(optimal.picks, projections, bank=0.5, horizon=5, budget=100.0)
    assert "sitting in the bank" not in titles(result)


def test_expensive_bench_is_flagged(projections):
    """A squad that stacks money on the bench should be told so."""
    expensive = sorted(projections, key=lambda p: -p.player.cost)
    quota, counts, picks = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}, {}, []
    for projection in expensive:
        position = projection.player.position
        if quota.get(position, 0) <= 0 or counts.get(projection.player.team, 0) >= 3:
            continue
        picks.append(projection)
        quota[position] -= 1
        counts[projection.player.team] = counts.get(projection.player.team, 0) + 1
        if not any(quota.values()):
            break
    result = diagnose_squad(picks, projections, bank=0.0, horizon=5)
    assert "tied up in outfield bench" in titles(result)


def test_gap_to_optimal_is_reported_for_a_weak_squad(projections):
    weak = sorted(projections, key=lambda p: p.expected_points)
    quota, counts, picks = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}, {}, []
    for projection in weak:
        position = projection.player.position
        if quota.get(position, 0) <= 0 or counts.get(projection.player.team, 0) >= 3:
            continue
        picks.append(projection)
        quota[position] -= 1
        counts[projection.player.team] = counts.get(projection.player.team, 0) + 1
        if not any(quota.values()):
            break

    result = diagnose_squad(picks, projections, bank=0.0, horizon=5, budget=100.0)
    comparison = result["comparison"]
    assert comparison["points_gap"] > 0
    assert comparison["overlap"] < 15
    assert comparison["drop"] and comparison["add"]
    # Everything dropped is yours; everything added is not.
    mine = {p.player.id for p in picks}
    assert all(p.player.id in mine for p in comparison["drop"])
    assert all(p.player.id not in mine for p in comparison["add"])


def test_findings_are_ordered_worst_first(projections):
    weak = sorted(projections, key=lambda p: p.expected_points)[:400]
    quota, counts, picks = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}, {}, []
    for projection in weak:
        position = projection.player.position
        if quota.get(position, 0) <= 0 or counts.get(projection.player.team, 0) >= 3:
            continue
        picks.append(projection)
        quota[position] -= 1
        counts[projection.player.team] = counts.get(projection.player.team, 0) + 1
        if not any(quota.values()):
            break
    result = diagnose_squad(picks, projections, bank=0.0, horizon=5)
    ranks = [f.rank for f in result["findings"]]
    assert ranks == sorted(ranks)


def test_diagnosis_reports_squad_value(optimal, projections):
    result = diagnose_squad(optimal.picks, projections, bank=2.0, horizon=5, budget=100.0)
    assert result["squad_value"] == pytest.approx(optimal.cost, abs=0.1)
    assert result["bank"] == 2.0


def test_diagnosis_survives_an_unoptimisable_budget(optimal, projections):
    """A budget too small for any legal squad must not crash the diagnosis."""
    result = diagnose_squad(optimal.picks, projections, bank=0.0, horizon=5, budget=1.0)
    assert result["comparison"] == {}
    assert isinstance(result["findings"], list)
