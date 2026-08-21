from fpl_agent.models import Player, Team, Fixture


RAW_PLAYER = {
    "id": 1, "web_name": "Salah", "first_name": "Mohamed", "second_name": "Salah",
    "team": 12, "element_type": 3, "now_cost": 125, "total_points": 120,
    "form": "7.5", "points_per_game": "6.3", "selected_by_percent": "58.2",
    "minutes": 900, "starts": 10, "goals_scored": 8, "assists": 6,
    "clean_sheets": 4, "goals_conceded": 8, "saves": 0, "bonus": 12, "bps": 400,
    "ict_index": "150.2", "expected_goals": "6.50", "expected_assists": "4.50",
    "expected_goals_conceded": "9.00", "defensive_contribution": 2.0,
    "status": "a", "chance_of_playing_next_round": None, "ep_next": "6.8",
    "news": "", "transfers_in_event": 100, "transfers_out_event": 50,
}


def test_player_parses_string_numerics():
    player = Player.from_api(RAW_PLAYER)
    assert player.cost == 12.5           # now_cost is in tenths
    assert player.form == 7.5
    assert player.selected_by == 58.2
    assert player.position == "MID"
    assert player.full_name == "Mohamed Salah"


def test_player_handles_missing_and_null_fields():
    player = Player.from_api({"id": 2, "team": 1, "element_type": 4})
    assert player.cost == 0.0
    assert player.form == 0.0
    assert player.expected_goals == 0.0
    assert player.position == "FWD"


def test_per_90_rates():
    player = Player.from_api(RAW_PLAYER)
    assert player.xg90 == 0.65          # 6.5 xG in 900 minutes
    assert player.xa90 == 0.45
    assert round(player.xgi90, 2) == 1.10


def test_per_90_is_zero_without_minutes():
    player = Player.from_api({**RAW_PLAYER, "minutes": 0})
    assert player.xg90 == 0.0
    assert player.bonus90 == 0.0


def test_availability_from_status_and_chance():
    assert Player.from_api(RAW_PLAYER).availability == 1.0
    injured = Player.from_api({**RAW_PLAYER, "status": "i"})
    assert injured.availability == 0.0
    # An explicit chance_of_playing overrides the status flag.
    doubtful = Player.from_api({**RAW_PLAYER, "status": "d",
                                "chance_of_playing_next_round": 75})
    assert doubtful.availability == 0.75


def test_fixture_perspective_helpers():
    fixture = Fixture.from_api({
        "id": 1, "event": 5, "team_h": 3, "team_a": 7,
        "team_h_difficulty": 2, "team_a_difficulty": 4, "finished": False,
    })
    assert fixture.involves(3) and fixture.involves(7)
    assert not fixture.involves(9)
    assert fixture.opponent_of(3) == 7
    assert fixture.is_home_for(3) and not fixture.is_home_for(7)
    assert fixture.difficulty_for(3) == 2
    assert fixture.difficulty_for(7) == 4


def test_team_strength_by_venue():
    team = Team.from_api({
        "id": 1, "name": "Arsenal", "short_name": "ARS", "strength": 5,
        "strength_attack_home": 1300, "strength_attack_away": 1250,
        "strength_defence_home": 1280, "strength_defence_away": 1220,
    })
    assert team.attack_strength(home=True) == 1300
    assert team.attack_strength(home=False) == 1250
    assert team.defence_strength(home=True) == 1280
