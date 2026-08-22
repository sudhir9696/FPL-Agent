"""European congestion: rest days, rotation multipliers, and the no-op default."""
import datetime as dt

import pytest

from fpl_agent.european import EuropeanCalendar, parse_kickoff


def calendar(ucl_teams=(), uel_teams=()):
    return EuropeanCalendar({
        "_teams_verified": True,
        "competitions": {
            "UCL": {
                "teams": list(ucl_teams),
                "matchdays": [{"md": 1, "dates": ["2026-09-08", "2026-09-09"]}],
            },
            "UEL": {
                "teams": list(uel_teams),
                "matchdays": [{"md": 1, "dates": ["2026-09-17"]}],
            },
        },
    })


def kickoff(day, hour=14):
    return dt.datetime(2026, 9, day, hour, tzinfo=dt.timezone.utc)


def test_shipped_calendar_parses():
    cal = EuropeanCalendar()
    assert "UCL" in cal.competitions
    assert cal.competitions["UCL"]["matchdays"], "matchday dates should be populated"


def test_inactive_until_teams_are_filled_in():
    """Dates alone must not move any projection."""
    cal = EuropeanCalendar({"competitions": {"UCL": {"teams": [],
                            "matchdays": [{"md": 1, "dates": ["2026-09-09"]}]}}})
    assert cal.active is False
    assert cal.rotation_multiplier("ARS", kickoff(12)) == (1.0, None)


def test_team_not_in_europe_is_unaffected():
    cal = calendar(ucl_teams=["ARS"])
    assert cal.rotation_multiplier("EVE", kickoff(12)) == (1.0, None)


def test_thursday_to_saturday_is_penalised_hardest():
    """UEL Thursday -> Saturday is two days and the real rotation driver."""
    cal = calendar(uel_teams=["TOT"])
    mult, note = cal.rotation_multiplier("TOT", kickoff(19))
    assert mult < 0.8
    assert note and "UEL" in note and "2d" in note


def test_midweek_ucl_is_penalised_less_than_thursday_uel():
    cal = calendar(ucl_teams=["ARS"], uel_teams=["TOT"])
    ucl, _ = cal.rotation_multiplier("ARS", kickoff(12))   # Wed 9th -> Sat 12th
    uel, _ = cal.rotation_multiplier("TOT", kickoff(19))   # Thu 17th -> Sat 19th
    assert uel < ucl < 1.0


def test_effect_decays_to_nothing_by_five_days():
    cal = calendar(ucl_teams=["ARS"])
    assert cal.rotation_multiplier("ARS", kickoff(14))[0] == 1.0


def test_fixture_before_the_european_tie_is_unaffected():
    cal = calendar(ucl_teams=["ARS"])
    assert cal.rotation_multiplier("ARS", kickoff(5))[0] == 1.0


def test_rest_days_uses_latest_date_in_the_matchday_window():
    """Which night of the window a club plays is unknown before the draw, so
    the least-rest reading is taken."""
    cal = calendar(ucl_teams=["ARS"])
    assert cal.rest_days_before("ARS", kickoff(12)) == 3   # from the 9th, not the 8th


@pytest.mark.parametrize("raw", [None, "", "not-a-date"])
def test_bad_kickoff_is_survivable(raw):
    assert parse_kickoff(raw) is None
    assert calendar(ucl_teams=["ARS"]).rotation_multiplier("ARS", None) == (1.0, None)


def test_projection_drops_when_a_club_is_congested(data):
    """End to end: same player, same fixtures, only the calendar differs."""
    from fpl_agent.analysis import ModelConfig, ProjectionModel

    player = max((p for p in data.players if p.minutes > 500),
                 key=lambda p: p.expected_goals)
    team = data.teams[player.team]
    fixtures = data.upcoming_fixtures(player.team, data.next_gameweek, 5)
    if not fixtures:
        pytest.skip("no fixtures for this player in the horizon")

    dates = sorted({parse_kickoff(f.kickoff_time).date().isoformat()
                    for f in fixtures if f.kickoff_time})
    if not dates:
        pytest.skip("fixtures carry no kickoff times")
    # Put a Europa tie two days before each of this club's fixtures.
    congested = EuropeanCalendar({
        "_teams_verified": True,
        "competitions": {"UEL": {
            "teams": [team.short_name],
            "matchdays": [{"md": i, "dates": [
                (dt.date.fromisoformat(d) - dt.timedelta(days=2)).isoformat()]}
                for i, d in enumerate(dates, 1)],
        }},
    })
    cfg = ModelConfig(horizon=5)
    clear = ProjectionModel(data, cfg, european=calendar()).project(player, data.next_gameweek)
    tired = ProjectionModel(data, cfg, european=congested).project(player, data.next_gameweek)

    assert tired.expected_points < clear.expected_points
    assert any("UEL" in n for n in tired.notes)
