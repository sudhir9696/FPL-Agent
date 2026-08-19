import json
import time

import pytest
import requests

from fpl_builder.models import Player
from fpl_builder.reddit import (
    RedditClient, RedditError, Thread, cross_reference, extract_player_buzz,
    gather_threads, score_thread,
)


def make_thread(**kwargs):
    base = {
        "id": "abc123", "title": "Gameweek 13 Team News", "author": "u/x",
        "score": 500, "num_comments": 200, "created_utc": time.time() - 3600,
        "permalink": "/r/FantasyPL/comments/abc123/gw13/", "subreddit": "FantasyPL",
    }
    base.update(kwargs)
    return Thread.from_api(base)


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
    """Stands in for requests.Session so tests never touch the network."""

    def __init__(self, routes, status_code=200):
        self.routes = routes
        self.status_code = status_code
        self.headers = {}
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(url)
        for fragment, payload in self.routes.items():
            if fragment in url:
                return FakeResponse(payload, self.status_code)
        return FakeResponse({"data": {"children": []}}, self.status_code)

    def post(self, url, auth=None, data=None, timeout=None):
        return FakeResponse({"access_token": "fake-token"})


def listing_payload(threads):
    return {"data": {"children": [
        {"kind": "t3", "data": {
            "id": t.id, "title": t.title, "author": t.author, "score": t.score,
            "num_comments": t.num_comments, "created_utc": t.created_utc,
            "permalink": t.permalink, "selftext": t.selftext,
            "link_flair_text": t.flair, "subreddit": t.subreddit,
        }} for t in threads
    ]}}


# --- thread scoring ---------------------------------------------------
def test_topical_threads_outrank_generic_ones():
    news = score_thread(make_thread(title="GW13 Team News and Press Conference Notes"))
    generic = score_thread(make_thread(id="z", title="Just saying hello everyone"))
    assert news.relevance > generic.relevance
    assert news.matched_topics


def test_recent_threads_outrank_stale_ones():
    fresh = score_thread(make_thread(created_utc=time.time() - 3600))
    stale = score_thread(make_thread(id="old", created_utc=time.time() - 3600 * 24 * 20))
    assert fresh.relevance > stale.relevance


def test_engagement_drives_relevance():
    busy = score_thread(make_thread(score=5000, num_comments=2000))
    quiet = score_thread(make_thread(id="q", score=5, num_comments=1))
    assert busy.relevance > quiet.relevance


def test_thread_url_and_age():
    thread = make_thread(created_utc=time.time() - 7200)
    assert thread.url.startswith("https://www.reddit.com/r/FantasyPL/")
    assert 1.8 < thread.age_hours < 2.2


# --- client -----------------------------------------------------------
def test_listing_parses_threads(tmp_path):
    threads = [make_thread(id="t1"), make_thread(id="t2", title="Captain picks GW13")]
    session = FakeSession({"/r/FantasyPL/hot.json": listing_payload(threads)})
    client = RedditClient(cache_dir=tmp_path, session=session)
    client._last_request = time.time() - 5      # skip the throttle sleep
    result = client.listing("FantasyPL", "hot")
    assert [t.id for t in result] == ["t1", "t2"]


def test_403_raises_a_helpful_error(tmp_path):
    session = FakeSession({}, status_code=403)
    client = RedditClient(cache_dir=tmp_path, session=session)
    client._last_request = time.time() - 5
    with pytest.raises(RedditError, match="REDDIT_CLIENT_ID"):
        client.listing("FantasyPL", "hot")


def test_429_raises_rate_limit_error(tmp_path):
    session = FakeSession({}, status_code=429)
    client = RedditClient(cache_dir=tmp_path, session=session)
    client._last_request = time.time() - 5
    with pytest.raises(RedditError, match="rate limit"):
        client.listing("FantasyPL", "hot")


def test_responses_are_cached(tmp_path):
    session = FakeSession({"/r/FantasyPL/hot.json": listing_payload([make_thread()])})
    client = RedditClient(cache_dir=tmp_path, session=session)
    client._last_request = time.time() - 5
    client.listing("FantasyPL", "hot")
    calls_after_first = len(session.calls)
    client.listing("FantasyPL", "hot")
    assert len(session.calls) == calls_after_first   # served from cache


def test_comment_tree_is_flattened(tmp_path):
    payload = [
        {"kind": "Listing", "data": {"children": []}},
        {"kind": "Listing", "data": {"children": [
            {"kind": "t1", "data": {
                "body": "Salah is essential",
                "replies": {"kind": "Listing", "data": {"children": [
                    {"kind": "t1", "data": {"body": "Agreed, captain him", "replies": ""}},
                    {"kind": "t1", "data": {"body": "[deleted]", "replies": ""}},
                ]}},
            }},
        ]}},
    ]
    session = FakeSession({"/comments/": payload})
    client = RedditClient(cache_dir=tmp_path, session=session)
    client._last_request = time.time() - 5
    bodies = client.top_comments("FantasyPL", "abc123")
    assert "Salah is essential" in bodies
    assert "Agreed, captain him" in bodies
    assert "[deleted]" not in bodies


def test_gather_threads_survives_a_failing_subreddit(tmp_path):
    session = FakeSession({}, status_code=403)
    client = RedditClient(cache_dir=tmp_path, session=session)
    client._last_request = time.time() - 5
    assert gather_threads(client, ["FantasyPL"], limit=5, fetch_comments=0) == []


# --- buzz extraction --------------------------------------------------
def make_player(pid, web_name, second_name=None, first_name="Test"):
    return Player.from_api({
        "id": pid, "web_name": web_name, "first_name": first_name,
        "second_name": second_name or web_name, "team": 1, "element_type": 3,
        "now_cost": 70, "minutes": 900,
    })


def test_player_mentions_are_counted():
    players = [make_player(1, "Salah"), make_player(2, "Haaland")]
    thread = make_thread(selftext="Salah is on fire. Salah again! Haaland blanked.")
    buzz = extract_player_buzz([score_thread(thread)], players)
    assert buzz[1].mentions == 2
    assert buzz[2].mentions == 1


def test_sentiment_separates_bullish_from_bearish():
    players = [make_player(1, "Salah"), make_player(2, "Nunez")]
    thread = make_thread(
        selftext="Salah is essential, captain him, he is elite. "
                 "Nunez is a trap, sell him, total liability."
    )
    buzz = extract_player_buzz([score_thread(thread)], players)
    assert buzz[1].sentiment > 0
    assert buzz[1].sentiment_label == "bullish"
    assert buzz[2].sentiment < 0
    assert buzz[2].sentiment_label == "bearish"


def test_neutral_mentions_have_zero_sentiment():
    players = [make_player(1, "Salah")]
    thread = make_thread(selftext="Salah played 90 minutes yesterday.")
    buzz = extract_player_buzz([score_thread(thread)], players)
    assert buzz[1].sentiment == 0.0
    assert buzz[1].sentiment_label == "mixed"


def test_ambiguous_names_need_a_capital_letter():
    """'king' the common word must not count as a mention of King."""
    players = [make_player(1, "King")]
    lowercase = make_thread(selftext="he is the king of assists this season")
    assert extract_player_buzz([score_thread(lowercase)], players) == {}

    proper = make_thread(id="p", selftext="King is nailed on right now")
    assert extract_player_buzz([score_thread(proper)], players)[1].mentions == 1


def test_substring_collisions_are_avoided():
    """'Rice' must not match inside 'price'."""
    players = [make_player(1, "Rice")]
    thread = make_thread(selftext="his price is rising and the price change is due")
    assert extract_player_buzz([score_thread(thread)], players) == {}


def test_mentions_are_aggregated_across_threads():
    players = [make_player(1, "Salah")]
    threads = [
        score_thread(make_thread(id="a", selftext="Salah haul")),
        score_thread(make_thread(id="b", selftext="Salah again")),
    ]
    buzz = extract_player_buzz(threads, players)
    assert buzz[1].mentions == 2
    assert len(buzz[1].threads) == 2


def test_full_name_matches_as_well_as_web_name():
    players = [make_player(1, "Saka", second_name="Saka", first_name="Bukayo")]
    thread = make_thread(selftext="Bukayo Saka is essential")
    assert extract_player_buzz([score_thread(thread)], players)[1].mentions >= 1


def test_min_mentions_filter():
    players = [make_player(1, "Salah")]
    thread = make_thread(selftext="Salah")
    assert extract_player_buzz([score_thread(thread)], players, min_mentions=2) == {}


# --- cross reference --------------------------------------------------
def test_cross_reference_buckets(projections):
    top = projections[:5]
    weak = projections[-60:-55]
    buzz = {}
    for i, projection in enumerate(weak):
        entry = extract_player_buzz(
            [score_thread(make_thread(id=f"w{i}", selftext=f"{projection.player.web_name} " * 6))],
            [projection.player],
        )
        buzz.update(entry)
    for projection in top:
        buzz.pop(projection.player.id, None)

    result = cross_reference(projections, buzz)
    assert set(result) == {"hype_traps", "under_radar", "consensus"}
    # Heavily-discussed but poorly-projected players are flagged as hype.
    flagged = {item["projection"].player.id for item in result["hype_traps"]}
    assert flagged & {p.player.id for p in weak}


def test_cross_reference_handles_no_buzz(projections):
    result = cross_reference(projections, {})
    assert result["hype_traps"] == []
    assert result["consensus"] == []


def test_cross_reference_handles_empty_projections():
    result = cross_reference([], {})
    assert result == {"hype_traps": [], "under_radar": [], "consensus": []}


def test_sentiment_does_not_leak_between_adjacent_players():
    """Regression: a nearby player's negative terms must not taint this one."""
    players = [make_player(1, "Salah"), make_player(2, "Nunez")]
    thread = make_thread(
        selftext="Salah is elite and essential. Nunez is a trap, sell him, liability."
    )
    buzz = extract_player_buzz([score_thread(thread)], players)
    assert buzz[1].negative == 0
    assert buzz[2].positive == 0


def test_sentiment_splits_on_contrastive_conjunctions():
    players = [make_player(1, "Salah")]
    thread = make_thread(selftext="I wanted Haaland but Salah is essential")
    buzz = extract_player_buzz([score_thread(thread)], players)
    assert buzz[1].positive > 0


# --- alias ambiguity -------------------------------------------------
def test_shared_surname_is_not_credited_to_every_player():
    """Regression: 'Silva' matched several players and inflated all of them."""
    players = [
        make_player(1, "B.Silva", second_name="Silva", first_name="Bernardo"),
        make_player(2, "T.Silva", second_name="Silva", first_name="Thiago"),
    ]
    for player in players:
        player.selected_by = 10.0        # equally owned: genuinely ambiguous
    thread = make_thread(selftext="Silva is essential this week")
    buzz = extract_player_buzz([score_thread(thread)], players)
    # Neither is credited, rather than both.
    assert buzz == {}


def test_dominant_player_wins_an_ambiguous_surname():
    players = [
        make_player(1, "B.Silva", second_name="Silva", first_name="Bernardo"),
        make_player(2, "T.Silva", second_name="Silva", first_name="Thiago"),
    ]
    players[0].selected_by = 30.0
    players[1].selected_by = 2.0
    thread = make_thread(selftext="Silva is essential this week")
    buzz = extract_player_buzz([score_thread(thread)], players)
    assert list(buzz) == [1]


def test_benchwarmer_does_not_steal_an_ambiguous_surname():
    """Only one candidate has minutes, so the alias resolves to them."""
    starter = make_player(1, "Gomes", second_name="Gomes")
    reserve = make_player(2, "Gomes 2", second_name="Gomes")
    reserve.minutes = 0
    starter.selected_by = reserve.selected_by = 5.0
    thread = make_thread(selftext="Gomes is nailed on")
    buzz = extract_player_buzz([score_thread(thread)], [starter, reserve])
    assert list(buzz) == [1]


def test_longer_alias_wins_over_the_bare_surname():
    """Regression: 'Watkins 8' counted once as itself, not also as 'Watkins'."""
    specific = make_player(1, "Watkins 8", second_name="Watkins")
    other = make_player(2, "Watkins 3", second_name="Watkins")
    specific.selected_by = other.selected_by = 5.0
    thread = make_thread(selftext="Watkins 8 is essential")
    buzz = extract_player_buzz([score_thread(thread)], [specific, other])
    assert list(buzz) == [1]
    assert buzz[1].mentions == 1        # not 2


def test_unique_web_name_still_resolves_when_surname_is_shared():
    a = make_player(1, "Watkins 8", second_name="Watkins")
    b = make_player(2, "Watkins 3", second_name="Watkins")
    a.selected_by = b.selected_by = 5.0
    thread = make_thread(selftext="Watkins 3 hauled again, he is elite")
    buzz = extract_player_buzz([score_thread(thread)], [a, b])
    assert list(buzz) == [2]
    assert buzz[2].sentiment > 0


def test_full_name_disambiguates_a_shared_surname():
    a = make_player(1, "B.Silva", second_name="Silva", first_name="Bernardo")
    b = make_player(2, "T.Silva", second_name="Silva", first_name="Thiago")
    a.selected_by = b.selected_by = 8.0
    thread = make_thread(selftext="Bernardo Silva is essential")
    buzz = extract_player_buzz([score_thread(thread)], [a, b])
    assert list(buzz) == [1]
