import copy

import pytest

from fpl_agent.analysis import (
    ModelConfig, ProjectionModel, _shrunk_rate, best_value, differentials, top_by_position,
)
from fpl_agent.api import GameData
from fpl_agent.models import Player


def test_shrunk_rate_pulls_small_samples_toward_the_prior():
    # 90 minutes with 2 goals is a 2.0 xG/90 rate; shrinkage must temper it.
    raw = 2.0
    shrunk = _shrunk_rate(total=2.0, minutes=90, prior_rate=0.35, prior_minutes=270)
    assert shrunk < raw
    assert shrunk > 0.35          # but still above the prior


def test_shrunk_rate_converges_to_observed_with_many_minutes():
    shrunk = _shrunk_rate(total=20.0, minutes=3000, prior_rate=0.35, prior_minutes=270)
    assert abs(shrunk - 0.6) < 0.05   # 20 goals / 3000 min = 0.6 per 90


def test_shrunk_rate_returns_prior_with_no_evidence():
    assert _shrunk_rate(total=0.0, minutes=0, prior_rate=0.25, prior_minutes=270) == 0.25


def test_every_player_gets_a_projection(data, projections):
    assert len(projections) == len(data.players)
    assert all(p.expected_points is not None for p in projections)


def test_projections_are_sorted_descending(projections):
    points = [p.expected_points for p in projections]
    assert points == sorted(points, reverse=True)


def test_projected_points_are_in_a_plausible_range(projections):
    # Nobody averages more than ~15 points a gameweek over a 5-GW horizon.
    for projection in projections:
        assert -5 <= projection.expected_points <= 75, projection.player.web_name
        assert 0.0 <= projection.start_probability <= 1.0


def test_injured_players_score_near_zero(data):
    model = ProjectionModel(data, ModelConfig(horizon=5))
    injured = [p for p in data.players if p.status == "i"]
    assert injured, "sample data should contain injured players"
    for player in injured[:5]:
        projection = model.project(player, data.next_gameweek)
        assert projection.start_probability == 0.0
        assert projection.expected_points < 3.0
        assert any("injured" in note for note in projection.notes)


def test_doubtful_player_is_discounted_not_zeroed(data):
    model = ProjectionModel(data, ModelConfig(horizon=5))
    doubtful = [p for p in data.players if p.status == "d"]
    assert doubtful
    projection = model.project(doubtful[0], data.next_gameweek)
    assert 0 < projection.start_probability < 1.0


def test_horizon_scales_expected_points(data):
    player = max(data.players, key=lambda p: p.total_points)
    short = ProjectionModel(data, ModelConfig(horizon=1)).project(player, data.next_gameweek)
    long = ProjectionModel(data, ModelConfig(horizon=6)).project(player, data.next_gameweek)
    assert long.expected_points > short.expected_points
    assert long.num_fixtures > short.num_fixtures


def test_blank_gameweek_is_flagged(data):
    """A team with no fixtures in the horizon yields zero points and a note."""
    model = ProjectionModel(data, ModelConfig(horizon=2))
    player = data.players[0]
    # Point the player at a far-future window with no scheduled fixtures.
    projection = model.project(player, start_gw=500)
    assert projection.num_fixtures == 0
    assert projection.expected_points == 0.0
    assert any("blank" in note for note in projection.notes)


def test_double_gameweek_doubles_the_fixture_count(data):
    """Duplicating a fixture must roughly double that player's projection."""
    doubled = GameData(copy.deepcopy(data.raw_bootstrap),
                       [f.__dict__ for f in data.fixtures])
    start_gw = doubled.next_gameweek
    target_team = doubled.players[0].team
    extra = next(
        f for f in doubled.fixtures if f.event == start_gw and f.involves(target_team)
    )
    clone = copy.deepcopy(extra)
    clone.id = 99999
    doubled.fixtures.append(clone)

    player = doubled.players[0]
    base = ProjectionModel(data, ModelConfig(horizon=1)).project(player, start_gw)
    dgw = ProjectionModel(doubled, ModelConfig(horizon=1)).project(player, start_gw)

    assert base.num_fixtures == 1
    assert dgw.num_fixtures == 2
    assert dgw.expected_points > base.expected_points * 1.8
    assert any("double gameweek" in note for note in dgw.notes)


def test_fixture_multiplier_favours_weak_opponents(data):
    model = ProjectionModel(data, ModelConfig(horizon=5))
    strong = max(data.teams.values(), key=lambda t: t.strength_attack_home)
    weak = min(data.teams.values(), key=lambda t: t.strength_attack_home)
    fixture = next(
        f for f in data.fixtures
        if not f.finished and f.involves(strong.id) and f.involves(weak.id)
    ) if any(
        not f.finished and f.involves(strong.id) and f.involves(weak.id)
        for f in data.fixtures
    ) else None
    if fixture is None:
        pytest.skip("no remaining fixture between the strongest and weakest teams")
    assert model.attack_multiplier(strong, fixture) > model.attack_multiplier(weak, fixture)


def test_fixture_multiplier_is_clamped(data):
    config = ModelConfig(horizon=5, max_fixture_swing=1.4)
    model = ProjectionModel(data, config)
    for team in data.teams.values():
        for fixture in data.fixtures[:100]:
            if fixture.involves(team.id):
                multiplier = model.attack_multiplier(team, fixture)
                assert 1 / 1.4 <= multiplier <= 1.4


def test_defcon_toggle_changes_defender_scores(data):
    defender = max(
        (p for p in data.players if p.position == "DEF" and p.minutes > 500),
        key=lambda p: p.defensive_contribution,
    )
    with_dc = ProjectionModel(data, ModelConfig(horizon=5, include_defcon=True))
    without_dc = ProjectionModel(data, ModelConfig(horizon=5, include_defcon=False))
    assert (with_dc.project(defender, data.next_gameweek).expected_points
            > without_dc.project(defender, data.next_gameweek).expected_points)


def test_goalkeepers_earn_save_points_but_outfielders_do_not(data):
    model = ProjectionModel(data, ModelConfig(horizon=5))
    # Must be a keeper who will actually play -- an injured one correctly scores zero.
    keeper = max(
        (p for p in data.players if p.position == "GK" and p.status == "a" and p.minutes > 500),
        key=lambda p: p.saves,
    )
    striker = max(
        (p for p in data.players if p.position == "FWD" and p.status == "a"),
        key=lambda p: p.minutes,
    )
    assert model.project(keeper, data.next_gameweek).breakdown["saves"] > 0
    assert model.project(striker, data.next_gameweek).breakdown["saves"] == 0


def test_weights_are_normalised():
    config = ModelConfig(weight_model=3, weight_form=1, weight_ep=1)
    weights = config.normalised_weights()
    assert abs(sum(weights) - 1.0) < 1e-9
    assert abs(weights[0] - 0.6) < 1e-9


def test_zero_weights_fall_back_to_the_model():
    assert ModelConfig(weight_model=0, weight_form=0, weight_ep=0).normalised_weights() == (1.0, 0.0, 0.0)


def test_value_is_points_per_million(projections):
    for projection in projections[:20]:
        expected = projection.expected_points / projection.player.cost
        assert abs(projection.value - expected) < 0.01


def test_top_by_position_filters_and_limits(projections):
    top = top_by_position(projections, "MID", limit=5)
    assert len(top) == 5
    assert all(p.player.position == "MID" for p in top)


def test_differentials_respect_the_ownership_cap(projections):
    for projection in differentials(projections, max_ownership=5.0, limit=10):
        assert projection.player.selected_by <= 5.0


def test_best_value_is_sorted_by_value(projections):
    values = [p.value for p in best_value(projections, min_points=2.0, limit=10)]
    assert values == sorted(values, reverse=True)


def test_unavailable_keeper_scores_no_saves(data):
    """Regression: breakdown components are weighted by start probability."""
    model = ProjectionModel(data, ModelConfig(horizon=5))
    injured_keepers = [p for p in data.players if p.position == "GK" and p.status == "i"]
    if not injured_keepers:
        return
    projection = model.project(injured_keepers[0], data.next_gameweek)
    assert projection.breakdown["saves"] == 0.0


def _one_provisional_round(data, event=1):
    clone = copy.deepcopy(data)
    for fixture in clone.fixtures:
        fixture.finished = False
        fixture.finished_provisional = fixture.event == event
    return clone


def test_starters_outrank_absentees_in_a_provisionally_finished_round(data):
    """Regression: `team_games_played` once counted only `finished` fixtures.

    In the days after a round ends the API reports it as provisional but not
    finished, so every team read as having played nothing. That sent
    `start_probability` down the pre-season branch, dividing real minutes by a
    full 38-game season -- a nailed-on starter scored ~3% while a player yet to
    feature fell back to ep_next and scored ~58%. The ranking was inverted.
    """
    clone = _one_provisional_round(data)
    model = ProjectionModel(clone, ModelConfig(horizon=5))
    fit = [p for p in clone.players if p.status == "a" and p.ep_next > 0]

    nailed_on = max(fit, key=lambda p: p.minutes)
    # Everyone in the sample data has minutes; make a player who has none.
    absent = copy.deepcopy(nailed_on)
    absent.minutes = 0
    absent.starts = 0

    started = model.start_probability(nailed_on)
    assert started > 0.5, started
    assert model.start_probability(absent) < started


def test_a_single_round_does_not_certify_a_starter(data):
    # One game of evidence can only say 0% or 100%; shrinkage keeps it honest.
    clone = _one_provisional_round(data)
    model = ProjectionModel(clone, ModelConfig(horizon=5))
    fit = [p for p in clone.players if p.status == "a" and p.ep_next > 0]
    nailed_on = max(fit, key=lambda p: p.minutes)
    assert model.start_probability(nailed_on) < 1.0


def test_preseason_still_falls_back_to_ep_next(data):
    # Before a ball is kicked there are no minutes to read, so ep_next is the
    # only signal and the fallback must survive.
    clone = copy.deepcopy(data)
    for fixture in clone.fixtures:
        fixture.finished = False
        fixture.finished_provisional = False
    model = ProjectionModel(clone, ModelConfig(horizon=5))
    fit = [p for p in clone.players if p.status == "a" and p.ep_next > 0]
    fresh = copy.deepcopy(fit[0])
    fresh.minutes = 0
    fresh.starts = 0
    assert model.start_probability(fresh) > 0.0
