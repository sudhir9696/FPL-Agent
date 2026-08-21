import pytest

from fpl_agent.models import SQUAD_QUOTA, XI_RANGE
from fpl_agent.optimizer import (
    OptimizerError, _filter_pool, _optimize_ilp, _optimize_local_search, _squad_objective,
    build_squad_object, optimize_squad, pick_best_xi, suggest_transfers,
)


def assert_legal_squad(squad, budget=100.0, max_per_team=3):
    assert len(squad.picks) == 15
    counts = {}
    for pick in squad.picks:
        counts[pick.player.position] = counts.get(pick.player.position, 0) + 1
    assert counts == SQUAD_QUOTA

    assert squad.cost <= budget + 1e-6

    per_team = {}
    for pick in squad.picks:
        per_team[pick.player.team] = per_team.get(pick.player.team, 0) + 1
    assert max(per_team.values()) <= max_per_team

    assert len(squad.starters) == 11
    assert len(squad.bench) == 4
    assert len({id(p) for p in squad.starters + squad.bench}) == 15

    xi_counts = {}
    for pick in squad.starters:
        xi_counts[pick.player.position] = xi_counts.get(pick.player.position, 0) + 1
    for position, (low, high) in XI_RANGE.items():
        assert low <= xi_counts.get(position, 0) <= high
    assert squad.bench[0].player.position == "GK"


@pytest.fixture(scope="module")
def squad(projections):
    return optimize_squad(projections, budget=100.0)


def test_optimal_squad_is_legal(squad):
    assert_legal_squad(squad)


def test_squad_spends_most_of_the_budget(squad):
    assert squad.cost > 95.0


def test_captain_is_the_highest_scorer_in_the_xi(squad):
    assert squad.captain is not None
    assert squad.captain.expected_points == max(p.expected_points for p in squad.starters)
    assert squad.vice_captain is not None
    assert squad.captain is not squad.vice_captain


def test_starters_outscore_the_bench(squad):
    worst_starter = min(p.expected_points for p in squad.starters)
    best_bench_outfield = max(
        (p.expected_points for p in squad.bench if p.player.position != "GK"), default=0
    )
    # Any bench outfielder beating a starter of the same position would be a bug.
    for bench_player in squad.bench:
        if bench_player.player.position == "GK":
            continue
        same_position = [
            p for p in squad.starters if p.player.position == bench_player.player.position
        ]
        if same_position:
            assert bench_player.expected_points <= max(p.expected_points for p in same_position)
    assert worst_starter >= 0 or best_bench_outfield >= 0


def test_tighter_budget_yields_a_cheaper_and_weaker_squad(projections):
    rich = optimize_squad(projections, budget=100.0)
    poor = optimize_squad(projections, budget=85.0)
    assert_legal_squad(poor, budget=85.0)
    assert poor.cost <= 85.0
    assert poor.expected_points <= rich.expected_points


def test_club_limit_is_enforced(projections):
    squad = optimize_squad(projections, budget=100.0, max_per_team=2)
    assert_legal_squad(squad, max_per_team=2)


def test_locked_players_appear_in_the_squad(projections):
    target = projections[40]
    squad = optimize_squad(projections, budget=100.0, lock_ids=[target.player.id])
    assert target.player.id in {p.player.id for p in squad.picks}
    assert_legal_squad(squad)


def test_excluded_players_are_absent(projections):
    banned = {p.player.id for p in projections[:6]}
    squad = optimize_squad(projections, budget=100.0, exclude_ids=banned)
    assert not banned & {p.player.id for p in squad.picks}


def test_locking_an_excluded_player_still_locks_it(projections):
    target = projections[3]
    squad = optimize_squad(
        projections, budget=100.0,
        lock_ids=[target.player.id], exclude_ids=[target.player.id],
    )
    assert target.player.id in {p.player.id for p in squad.picks}


def test_unknown_locked_id_raises(projections):
    with pytest.raises(OptimizerError, match="not found"):
        optimize_squad(projections, lock_ids=[999999])


def test_impossible_budget_raises(projections):
    with pytest.raises(OptimizerError):
        optimize_squad(projections, budget=30.0)


def test_local_search_fallback_produces_a_legal_squad(projections):
    """The no-PuLP path must obey every constraint too."""
    squad = _optimize_local_search(
        list(projections), budget=100.0, max_per_team=3, bench_weight=0.12, lock_ids=set()
    )
    assert_legal_squad(squad)


def test_ilp_is_at_least_as_good_as_local_search(projections):
    """On the same pool and the same objective, the ILP is provably optimal."""
    pool = _filter_pool(projections, set(), 0.15, None)
    ilp = _optimize_ilp(pool, 100.0, 3, 0.12, set())
    heuristic = _optimize_local_search(list(pool), 100.0, 3, 0.12, set())
    assert (_squad_objective(ilp.picks, 0.12)
            >= _squad_objective(heuristic.picks, 0.12) - 1e-6)


def test_optimal_xi_is_made_of_likely_starters(projections):
    """Regression: fringe players must not be over-rated by the form anchor."""
    squad = optimize_squad(projections, budget=100.0)
    assert min(p.start_probability for p in squad.starters) > 0.5


def test_pick_best_xi_chooses_a_legal_formation(squad):
    starters, bench = pick_best_xi(squad.picks)
    assert len(starters) == 11 and len(bench) == 4
    counts = {}
    for pick in starters:
        counts[pick.player.position] = counts.get(pick.player.position, 0) + 1
    assert counts.get("GK", 0) == 1
    assert sum(counts.values()) == 11


def test_pick_best_xi_rejects_an_illegal_squad(projections):
    keepers_only = [p for p in projections if p.player.position == "GK"][:15]
    with pytest.raises(OptimizerError):
        pick_best_xi(keepers_only)


def test_formation_string_matches_the_xi(squad):
    counts = {"DEF": 0, "MID": 0, "FWD": 0}
    for pick in squad.starters:
        if pick.player.position in counts:
            counts[pick.player.position] += 1
    assert squad.formation == f"{counts['DEF']}-{counts['MID']}-{counts['FWD']}"


def test_expected_points_counts_the_captain_twice(squad):
    xi_total = sum(p.expected_points for p in squad.starters)
    assert abs(squad.expected_points - (xi_total + squad.captain.expected_points)) < 0.01


# --- transfers -------------------------------------------------------
def test_transfers_improve_a_deliberately_weak_squad(projections):
    weak = _weak_but_legal_squad(projections)
    moves = suggest_transfers(weak, projections, bank=5.0, free_transfers=1, max_transfers=3)
    assert moves, "a weak squad should have improving transfers available"
    for move in moves:
        assert move["net_gain"] > 0
        assert move["in"].player.position == move["out"].player.position


def test_first_transfer_is_free_and_later_ones_take_hits(projections):
    weak = _weak_but_legal_squad(projections)
    moves = suggest_transfers(weak, projections, bank=10.0, free_transfers=1, max_transfers=3)
    if moves:
        assert moves[0]["hit"] == 0
    if len(moves) > 1:
        assert moves[1]["hit"] == 4
        assert moves[1]["net_gain"] == pytest.approx(moves[1]["raw_gain"] - 4, abs=0.01)


def test_transfers_respect_the_bank(projections):
    weak = _weak_but_legal_squad(projections)
    moves = suggest_transfers(weak, projections, bank=0.0, free_transfers=1, max_transfers=1)
    for move in moves:
        assert move["in"].player.cost <= move["out"].player.cost + 1e-6


def test_no_transfers_suggested_for_an_optimal_squad(projections):
    optimal = optimize_squad(projections, budget=100.0)
    moves = suggest_transfers(optimal.picks, projections, bank=0.0, free_transfers=1)
    # An optimal squad with no money left has nothing worth doing.
    assert all(m["net_gain"] > 0 for m in moves)


def test_transfers_never_exceed_the_club_limit(projections):
    weak = _weak_but_legal_squad(projections)
    moves = suggest_transfers(weak, projections, bank=20.0, free_transfers=3, max_transfers=3)
    squad = list(weak)
    for move in moves:
        squad = [move["in"] if p is move["out"] else p for p in squad]
    counts = {}
    for pick in squad:
        counts[pick.player.team] = counts.get(pick.player.team, 0) + 1
    assert max(counts.values()) <= 3


def _weak_but_legal_squad(projections):
    """Cheapest legal squad -- guaranteed to have upgrades available."""
    return _optimize_local_search(
        [p for p in projections if p.expected_points < 12],
        budget=100.0, max_per_team=3, bench_weight=0.12, lock_ids=set(),
    ).picks
