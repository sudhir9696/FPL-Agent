"""Rolling form: windowing, blending, and the no-history default."""
import pytest

from fpl_agent.analysis import ModelConfig, ProjectionModel
from fpl_agent.history import HistoryStore, PlayerHistory


def rows(*specs):
    """Build API-shaped history rows from (round, minutes, xg) triples."""
    return [
        {"round": r, "minutes": m, "starts": 1 if m >= 60 else 0, "total_points": 0,
         "expected_goals": xg, "expected_assists": 0, "expected_goals_conceded": 0,
         "defensive_contribution": 0, "bonus": 0, "saves": 0}
        for r, m, xg in specs
    ]


def test_window_takes_the_most_recent_gameweeks():
    h = PlayerHistory(rows((1, 90, 0.1), (2, 90, 0.2), (3, 90, 0.9)))
    w = h.window(2)
    assert w.gameweeks == 2
    assert w.expected_goals == pytest.approx(1.1)   # gw2 + gw3, not gw1


def test_window_is_ordered_by_round_not_input_order():
    h = PlayerHistory(rows((3, 90, 0.9), (1, 90, 0.1), (2, 90, 0.2)))
    assert h.window(1).expected_goals == pytest.approx(0.9)


def test_blank_gameweeks_stay_in_the_window():
    """A player who did not feature had no chance to score; dropping those
    rows would flatter someone who has lost their place."""
    h = PlayerHistory(rows((1, 90, 0.5), (2, 0, 0.0), (3, 0, 0.0)))
    w = h.window(3)
    assert w.gameweeks == 3
    assert w.minutes == 90
    assert w.per90("expected_goals") == pytest.approx(0.5)


def test_window_of_empty_history_is_none():
    assert PlayerHistory([]).window(5) is None


def test_store_lookup_misses_are_none():
    assert HistoryStore().window(999, 5) is None


def test_store_load_survives_a_failing_player(monkeypatch):
    class Client:
        def element_summary(self, pid, use_cache=True):
            if pid == 2:
                raise RuntimeError("boom")
            return {"history": rows((1, 90, 0.3))}

    store = HistoryStore.load(Client(), [1, 2, 3], workers=2)
    assert 1 in store and 3 in store
    assert 2 not in store, "a failed fetch should be absent, not fatal"


# --- integration with the projection model -------------------------------

def _model(data, history=None, **cfg):
    return ProjectionModel(data, ModelConfig(horizon=5, **cfg), history=history)


def test_no_history_matches_the_season_only_projection(data):
    """The default must not move a projection when no history is present."""
    player = max((p for p in data.players if p.minutes > 500),
                 key=lambda p: p.expected_goals)
    gw = data.next_gameweek
    plain = _model(data, recent_window=0).project(player, gw)
    empty = _model(data, HistoryStore()).project(player, gw)
    assert plain.expected_points == pytest.approx(empty.expected_points)


def test_cold_recent_form_drags_a_hot_season_down(data):
    player = max((p for p in data.players if p.minutes > 500),
                 key=lambda p: p.expected_goals)
    gw = data.next_gameweek
    cold = HistoryStore({player.id: PlayerHistory(
        rows((1, 90, 0.0), (2, 90, 0.0), (3, 90, 0.0), (4, 90, 0.0), (5, 90, 0.0))
    )})
    assert (_model(data, cold).project(player, gw).expected_points
            < _model(data, recent_window=0).project(player, gw).expected_points)


def test_hot_recent_form_lifts_a_quiet_season(data):
    player = min((p for p in data.players if p.minutes > 500 and p.position == "MID"),
                 key=lambda p: p.expected_goals)
    gw = data.next_gameweek
    hot = HistoryStore({player.id: PlayerHistory(
        rows((1, 90, 1.2), (2, 90, 1.1), (3, 90, 1.3), (4, 90, 1.0), (5, 90, 1.4))
    )})
    assert (_model(data, hot).project(player, gw).expected_points
            > _model(data, recent_window=0).project(player, gw).expected_points)


def test_one_cameo_barely_moves_the_estimate(data):
    """Weight scales with the minutes behind the window, so a single
    substitute appearance must not overturn a season."""
    player = max((p for p in data.players if p.minutes > 500),
                 key=lambda p: p.expected_goals)
    gw = data.next_gameweek
    base = _model(data, recent_window=0).project(player, gw).expected_points
    cameo = HistoryStore({player.id: PlayerHistory(rows((5, 8, 0.0)))})
    shifted = _model(data, cameo).project(player, gw).expected_points
    assert abs(shifted - base) < 0.06 * base


def test_weight_recent_zero_disables_the_blend(data):
    player = max((p for p in data.players if p.minutes > 500),
                 key=lambda p: p.expected_goals)
    gw = data.next_gameweek
    cold = HistoryStore({player.id: PlayerHistory(
        rows((1, 90, 0.0), (2, 90, 0.0), (3, 90, 0.0))
    )})
    assert (_model(data, cold, weight_recent=0.0).project(player, gw).expected_points
            == pytest.approx(_model(data, recent_window=0).project(player, gw).expected_points))
