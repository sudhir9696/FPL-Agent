"""Mine r/FantasyPL (and friends) for community signal on players.

Reddit's JSON endpoints are public and need no credentials, but they rate-limit
anonymous traffic hard and often block datacenter IPs outright. If you hit 403s,
set REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET (create a free "script" app at
https://www.reddit.com/prefs/apps) and this module will use OAuth automatically.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import requests

log = logging.getLogger(__name__)

PUBLIC_BASE = "https://www.reddit.com"
OAUTH_BASE = "https://oauth.reddit.com"
DEFAULT_SUBREDDITS = ("FantasyPL",)

# Reddit requires a descriptive, unique User-Agent.
USER_AGENT = "python:fpl-team-builder:0.1 (by /u/fpl-team-builder)"

# Threads matching these carry the most decision-relevant discussion.
NOTEWORTHY_PATTERNS = [
    (re.compile(r"\bgameweek\s*\d+", re.I), 2.0),
    (re.compile(r"\bgw\s*\d+", re.I), 1.8),
    (re.compile(r"team\s*news|press\s*conference|presser", re.I), 2.5),
    (re.compile(r"\binjur|\bdoubt|\bfitness\b|\bsuspend", re.I), 2.2),
    (re.compile(r"captain|armband|\b(tc|triple\s*captain)\b", re.I), 2.0),
    (re.compile(r"wildcard|free\s*hit|bench\s*boost|chip", re.I), 1.8),
    (re.compile(r"transfer|\bin\s*or\s*out\b|\bsell\b|\bbuy\b", re.I), 1.6),
    (re.compile(r"rate\s*my\s*team|rmt", re.I), 0.8),
    (re.compile(r"daily\s*discussion|weekly\s*discussion|megathread", re.I), 1.5),
    (re.compile(r"price\s*change|\bpricewatch\b", re.I), 1.2),
    (re.compile(r"differential|template|eo\b|effective\s*ownership", re.I), 1.4),
    (re.compile(r"\bstat|\bxg|\bunderlying|\bdata\b|\bmodel\b", re.I), 1.3),
]

POSITIVE_TERMS = {
    "essential", "must", "haul", "hauling", "nailed", "explosive", "elite",
    "bargain", "steal", "form", "eyeing", "bringing in", "buy", "in for",
    "get him", "great fixtures", "easy fixtures", "differential", "punt",
    "premium", "captaining", "captain", "on fire", "unreal", "monster",
    "underlying", "value", "set and forget", "locked in", "bandwagon",
}
NEGATIVE_TERMS = {
    "injured", "injury", "doubt", "benched", "bench", "rotation", "rotated",
    "sell", "selling", "ditch", "avoid", "trap", "blank", "blanked",
    "overpriced", "dropped", "suspended", "knock", "hamstring", "out for",
    "disappointing", "cursed", "sideways", "regret", "hard fixtures",
    "tough fixtures", "get rid", "cut", "fodder", "liability",
}

# Short surnames that collide with ordinary English. Only matched when they
# appear capitalised in the source text.
AMBIGUOUS_NAMES = {
    "king", "long", "ward", "green", "bright", "young", "moore", "rice",
    "brooks", "wood", "reed", "white", "bell", "cash", "may", "gray", "grey",
    "west", "man", "son", "sa", "mac", "ake", "ali", "are", "ings", "hall",
    "walker", "wise", "best", "law", "may", "burn", "carter", "field",
}


class RedditError(RuntimeError):
    """Raised when Reddit cannot be reached."""


@dataclass
class Thread:
    id: str
    title: str
    author: str
    score: int
    num_comments: int
    created_utc: float
    permalink: str
    selftext: str = ""
    flair: str = ""
    subreddit: str = ""
    comments: list = field(default_factory=list)
    relevance: float = 0.0
    matched_topics: list = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"https://www.reddit.com{self.permalink}"

    @property
    def age_hours(self) -> float:
        return max(0.0, (time.time() - self.created_utc) / 3600.0)

    @classmethod
    def from_api(cls, raw: dict) -> "Thread":
        return cls(
            id=raw.get("id", ""),
            title=raw.get("title", ""),
            author=raw.get("author", "") or "[deleted]",
            score=int(raw.get("score", 0) or 0),
            num_comments=int(raw.get("num_comments", 0) or 0),
            created_utc=float(raw.get("created_utc", 0) or 0),
            permalink=raw.get("permalink", ""),
            selftext=raw.get("selftext", "") or "",
            flair=raw.get("link_flair_text", "") or "",
            subreddit=raw.get("subreddit", "") or "",
        )

    def text_blob(self) -> str:
        return "\n".join([self.title, self.selftext] + self.comments)


@dataclass
class PlayerBuzz:
    """Aggregated community chatter about a single player."""

    player_id: int
    name: str
    mentions: int = 0
    weighted_score: float = 0.0
    positive: int = 0
    negative: int = 0
    threads: set = field(default_factory=set)
    quotes: list = field(default_factory=list)

    @property
    def sentiment(self) -> float:
        """-1 (sell) .. +1 (buy). 0 when opinion is balanced or absent."""
        total = self.positive + self.negative
        if total == 0:
            return 0.0
        return round((self.positive - self.negative) / total, 3)

    @property
    def sentiment_label(self) -> str:
        s = self.sentiment
        if s >= 0.35:
            return "bullish"
        if s <= -0.35:
            return "bearish"
        return "mixed"


class RedditClient:
    """Public JSON reader with optional OAuth and an on-disk cache."""

    def __init__(
        self,
        user_agent: str = USER_AGENT,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        cache_ttl: int = 1800,
        timeout: int = 30,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "fpl_builder" / "reddit"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client_id = client_id or os.environ.get("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("REDDIT_CLIENT_SECRET")
        self._token: Optional[str] = None
        self._last_request = 0.0

    # --- auth ---------------------------------------------------------
    def _authenticate(self) -> Optional[str]:
        if self._token or not (self.client_id and self.client_secret):
            return self._token
        try:
            response = self.session.post(
                f"{PUBLIC_BASE}/api/v1/access_token",
                auth=(self.client_id, self.client_secret),
                data={"grant_type": "client_credentials"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            self._token = response.json().get("access_token")
            if self._token:
                log.info("authenticated to Reddit via OAuth")
        except (requests.RequestException, ValueError) as exc:
            log.warning("Reddit OAuth failed, continuing anonymously: %s", exc)
        return self._token

    # --- plumbing -----------------------------------------------------
    def _cache_path(self, key: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", key)[:180]
        return self.cache_dir / f"{safe}.json"

    def _throttle(self) -> None:
        # Reddit asks for <= 1 request/second from anonymous clients.
        elapsed = time.time() - self._last_request
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self._last_request = time.time()

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        params = dict(params or {})
        params.setdefault("raw_json", 1)
        cache_key = path + "_" + "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
        cache_file = self._cache_path(cache_key)

        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < self.cache_ttl:
            try:
                return json.loads(cache_file.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        token = self._authenticate()
        base = OAUTH_BASE if token else PUBLIC_BASE
        headers = {"Authorization": f"bearer {token}"} if token else {}

        self._throttle()
        url = f"{base}{path}"
        log.info("GET %s", url)
        try:
            response = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
            if response.status_code in (401, 403):
                raise RedditError(
                    f"Reddit returned {response.status_code} for {url}. Anonymous access is "
                    "often blocked from cloud/datacenter IPs -- set REDDIT_CLIENT_ID and "
                    "REDDIT_CLIENT_SECRET for authenticated access, or run with --no-reddit."
                )
            if response.status_code == 429:
                raise RedditError("Reddit rate limit hit (429). Wait a minute and retry.")
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.ProxyError as exc:
            raise RedditError(
                f"Could not reach {url}: blocked by a network proxy, not by Reddit. "
                f"Run with --no-reddit to skip community analysis. ({exc})"
            ) from exc
        except requests.RequestException as exc:
            raise RedditError(f"Reddit request to {url} failed: {exc}") from exc
        except ValueError as exc:
            raise RedditError(f"Reddit returned invalid JSON from {url}: {exc}") from exc

        try:
            cache_file.write_text(json.dumps(payload))
        except OSError:
            pass
        return payload

    # --- endpoints ----------------------------------------------------
    def listing(
        self, subreddit: str, sort: str = "hot", limit: int = 25, time_filter: str = "week"
    ) -> list:
        params = {"limit": min(limit, 100)}
        if sort in ("top", "controversial"):
            params["t"] = time_filter
        payload = self._get(f"/r/{subreddit}/{sort}.json", params)
        children = payload.get("data", {}).get("children", [])
        return [Thread.from_api(c["data"]) for c in children if c.get("kind") == "t3"]

    def search(self, subreddit: str, query: str, limit: int = 25, sort: str = "relevance") -> list:
        payload = self._get(
            f"/r/{subreddit}/search.json",
            {"q": query, "restrict_sr": 1, "limit": min(limit, 100), "sort": sort, "t": "month"},
        )
        children = payload.get("data", {}).get("children", [])
        return [Thread.from_api(c["data"]) for c in children if c.get("kind") == "t3"]

    def top_comments(self, subreddit: str, post_id: str, limit: int = 100) -> list:
        """Flatten a thread's comment tree into a list of body strings."""
        payload = self._get(
            f"/r/{subreddit}/comments/{post_id}.json",
            {"limit": limit, "sort": "top", "depth": 3},
        )
        bodies: list = []

        def walk(node) -> None:
            if isinstance(node, list):
                for item in node:
                    walk(item)
                return
            if not isinstance(node, dict):
                return
            if node.get("kind") == "Listing":
                walk(node.get("data", {}).get("children", []))
                return
            if node.get("kind") == "t1":
                data = node.get("data", {})
                body = data.get("body")
                if body and body not in ("[deleted]", "[removed]"):
                    bodies.append(body)
                replies = data.get("replies")
                if replies:
                    walk(replies)

        walk(payload)
        return bodies[:limit]


def score_thread(thread: Thread) -> Thread:
    """Attach a relevance score: engagement x topical weight x recency."""
    topics, weight = [], 1.0
    haystack = f"{thread.title} {thread.flair}"
    for pattern, bump in NOTEWORTHY_PATTERNS:
        if pattern.search(haystack):
            weight = max(weight, bump)
            topics.append(pattern.pattern.split("|")[0].strip("\\b"))

    engagement = thread.score + 2.0 * thread.num_comments
    # Half-life of roughly three days: FPL advice decays fast.
    recency = 0.5 ** (thread.age_hours / 72.0) if thread.created_utc else 0.5
    thread.relevance = round(engagement * weight * (0.35 + 0.65 * recency), 1)
    thread.matched_topics = topics
    return thread


def gather_threads(
    client: RedditClient,
    subreddits: Iterable[str] = DEFAULT_SUBREDDITS,
    limit: int = 25,
    gameweek: Optional[int] = None,
    fetch_comments: int = 5,
    comment_limit: int = 80,
) -> list:
    """Collect and rank noteworthy threads, pulling comments for the top few."""
    threads: dict = {}

    for subreddit in subreddits:
        for sort in ("hot", "top"):
            try:
                for thread in client.listing(subreddit, sort=sort, limit=limit):
                    threads[thread.id] = thread
            except RedditError as exc:
                log.warning("could not read r/%s/%s: %s", subreddit, sort, exc)

        queries = ["team news", "injury news", "captain"]
        if gameweek:
            queries.append(f"GW{gameweek}")
        for query in queries:
            try:
                for thread in client.search(subreddit, query, limit=10):
                    threads.setdefault(thread.id, thread)
            except RedditError as exc:
                log.debug("search '%s' in r/%s failed: %s", query, subreddit, exc)

    ranked = sorted((score_thread(t) for t in threads.values()),
                    key=lambda t: t.relevance, reverse=True)

    for thread in ranked[:fetch_comments]:
        try:
            thread.comments = client.top_comments(
                thread.subreddit or DEFAULT_SUBREDDITS[0], thread.id, limit=comment_limit
            )
        except RedditError as exc:
            log.debug("comments for %s unavailable: %s", thread.id, exc)

    return ranked


def _build_alias_index(players: Iterable) -> dict:
    """Map lowercase alias -> set of candidate player ids."""
    index: dict = {}
    for player in players:
        aliases = {player.web_name, player.second_name, player.full_name}
        aliases = {a.strip() for a in aliases if a and len(a.strip()) >= 3}
        for alias in aliases:
            index.setdefault(alias.lower(), set()).add(player.id)
    return index


def _resolve_alias(candidate_ids: set, by_id: dict) -> Optional[int]:
    """Pick which player an alias refers to, or None if it is too ambiguous.

    Surnames are routinely shared -- several players are called "Silva" -- so
    crediting every candidate would wildly inflate their buzz. Prefer the one
    player who actually features; if several do, accept the clearly dominant
    pick by ownership and otherwise decline to guess.
    """
    if len(candidate_ids) == 1:
        return next(iter(candidate_ids))

    candidates = [by_id[i] for i in candidate_ids if i in by_id]
    if not candidates:
        return None

    playing = [p for p in candidates if p.minutes > 0] or candidates
    if len(playing) == 1:
        return playing[0].id

    playing.sort(key=lambda p: (-p.selected_by, -p.total_points))
    runner_up = max(playing[1].selected_by, 0.1)
    if playing[0].selected_by >= 3 * runner_up:
        return playing[0].id
    return None


def _dedupe_overlaps(matches: list) -> list:
    """Keep the longest match at each position.

    "Watkins 8" must count once as that player, not also as a bare "Watkins".
    """
    accepted: list = []
    occupied: list = []
    for start, end, alias, text in sorted(matches, key=lambda m: -(m[1] - m[0])):
        if any(start < o_end and end > o_start for o_start, o_end in occupied):
            continue
        occupied.append((start, end))
        accepted.append((alias, text))
    return accepted


def _split_sentences(text: str) -> list:
    """Split text into clause-sized chunks for sentiment attribution.

    Sentiment is scoped to the clause a player is named in, so that
    "Salah is essential. Nunez is a trap." does not credit Salah with the
    negative terms aimed at Nunez.
    """
    chunks = re.split(r"[.!?;\n]+|\s+(?:but|however|whereas|although|though)\s+", text)
    return [c.strip() for c in chunks if c and c.strip()]


def extract_player_buzz(
    threads: Iterable[Thread], players: Iterable, min_mentions: int = 1
) -> dict:
    """Count and sentiment-score player mentions across threads."""
    players = list(players)
    by_id = {p.id: p for p in players}
    alias_index = _build_alias_index(players)

    # Resolve each alias to at most one player up front; drop the hopeless ones.
    alias_owner = {
        alias: _resolve_alias(ids, by_id) for alias, ids in alias_index.items()
    }
    alias_owner = {alias: pid for alias, pid in alias_owner.items() if pid is not None}
    buzz: dict = {}

    patterns = [
        (alias, re.compile(rf"(?<![\w']){re.escape(alias)}(?![\w'])", re.I))
        for alias in alias_owner
    ]

    for thread in threads:
        blob = thread.text_blob()
        if not blob:
            continue
        # A busy thread's opinions carry more weight than a dead one's.
        thread_weight = 1.0 + (thread.relevance / 1000.0)

        for sentence in _split_sentences(blob):
            lowered = sentence.lower()
            positives = sum(1 for term in POSITIVE_TERMS if term in lowered)
            negatives = sum(1 for term in NEGATIVE_TERMS if term in lowered)

            spans = []
            for alias, pattern in patterns:
                for match in pattern.finditer(sentence):
                    if alias in AMBIGUOUS_NAMES and not match.group(0)[0].isupper():
                        # Ordinary-word collision: "his price is rising" is not Rice.
                        continue
                    spans.append((match.start(), match.end(), alias, match.group(0)))

            for alias, _text in _dedupe_overlaps(spans):
                player_id = alias_owner[alias]
                player = by_id.get(player_id)
                if player is None:
                    continue
                entry = buzz.setdefault(
                    player_id, PlayerBuzz(player_id=player_id, name=player.web_name)
                )
                entry.mentions += 1
                entry.weighted_score += thread_weight
                entry.threads.add(thread.id)
                entry.positive += positives
                entry.negative += negatives
                if (positives or negatives) and len(entry.quotes) < 3:
                    entry.quotes.append(" ".join(sentence.split())[:200])

    for entry in buzz.values():
        entry.weighted_score = round(entry.weighted_score, 2)

    return {pid: b for pid, b in buzz.items() if b.mentions >= min_mentions}


def cross_reference(projections: Iterable, buzz: dict, limit: int = 8) -> dict:
    """Compare the model against the crowd.

    * `hype_traps`   - loud on Reddit, weak in the model
    * `under_radar`  - strong in the model, quiet on Reddit
    * `consensus`    - both agree, and the community is bullish
    """
    projections = [p for p in projections if p.player.cost > 0]
    if not projections:
        return {"hype_traps": [], "under_radar": [], "consensus": []}

    ranked = sorted(projections, key=lambda p: p.expected_points, reverse=True)
    model_rank = {p.player.id: i for i, p in enumerate(ranked)}
    total = len(ranked)

    buzz_ranked = sorted(buzz.values(), key=lambda b: b.weighted_score, reverse=True)
    buzz_rank = {b.player_id: i for i, b in enumerate(buzz_ranked)}
    buzz_total = max(1, len(buzz_ranked))

    hype_traps, under_radar, consensus = [], [], []

    for projection in ranked:
        pid = projection.player.id
        model_pct = model_rank[pid] / total          # 0 = best
        entry = buzz.get(pid)

        if entry and entry.mentions >= 3:
            buzz_pct = buzz_rank[pid] / buzz_total
            if buzz_pct < 0.25 and model_pct > 0.35:
                hype_traps.append({"projection": projection, "buzz": entry})
            elif buzz_pct < 0.35 and model_pct < 0.12 and entry.sentiment >= 0:
                consensus.append({"projection": projection, "buzz": entry})
        elif model_pct < 0.06 and projection.player.selected_by < 12:
            under_radar.append({"projection": projection, "buzz": entry})

    return {
        "hype_traps": hype_traps[:limit],
        "under_radar": under_radar[:limit],
        "consensus": consensus[:limit],
    }
