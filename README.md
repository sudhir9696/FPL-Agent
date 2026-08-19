# FPL Team Builder

Build and analyse a Fantasy Premier League squad from the **public FPL API**,
cross-referenced against **r/FantasyPL** discussion.

No API key, no login, no scraping — the FPL endpoints used here are the same
public JSON the official web app calls.

## What it does

1. **Pulls live data** from the FPL API — every player's price, minutes, xG, xA,
   xGC, bonus, defensive contributions, injury status and ownership, plus all
   380 fixtures with difficulty ratings.
2. **Projects expected points** for every player over a multi-gameweek horizon
   by walking their *actual* remaining fixtures, so blanks and doubles are
   handled correctly.
3. **Optimises a legal 15-man squad** with integer linear programming — budget,
   positional quotas, the 3-per-club limit, formation and captaincy all solved
   together, not greedily.
4. **Mines r/FantasyPL** for noteworthy threads (team news, press conferences,
   injuries, captaincy, chips), extracts which players are being discussed and
   whether the mood is bullish or bearish.
5. **Compares the model against the crowd** to surface consensus buys, possible
   hype traps, and under-the-radar picks.

## Install

```bash
git clone <this repo> && cd fpl-team-builder
pip install -e ".[dev]"
```

Only `requests` is required. `pulp` is optional but recommended — with it the
squad is provably optimal; without it a local search runs instead.

## Quick start

```bash
# Build the best squad for the next 5 gameweeks
fpl-builder build

# Analyse YOUR team and get transfer suggestions
fpl-builder team 1234567

# Rank players
fpl-builder players --position MID --limit 20
fpl-builder players --sort value --max-price 6.0

# Reddit digest on its own
fpl-builder reddit
```

Every command also runs as `python -m fpl_builder.cli ...`.

### Finding your entry ID

Log in to the FPL site, open **Points** or **Pick Team**, and read the number
out of the URL:

```
https://fantasy.premierleague.com/entry/1234567/event/13
                                        ^^^^^^^ this is your entry id
```

Then:

```bash
fpl-builder team 1234567 --horizon 5 --free-transfers 1
```

That prints your current XI and bench with a projection for each player, flags
injuries and rotation risks, and ranks transfers by **net** gain — each move
past your free transfers must beat its own −4 hit to be suggested.

## Commands

| Command | What it does |
|---|---|
| `build` | Optimise a squad from scratch |
| `team <entry-id>` | Show your current team and suggest transfers |
| `players` | Rank players by points, value or differential status |
| `reddit` | Noteworthy threads and community buzz |
| `fetch` | Save API data locally for offline runs |

### Useful flags

```bash
--horizon 8                  # plan further ahead (default 5)
--gw 13                      # plan from a specific gameweek
--budget 100.0               # squad budget in millions
--lock Salah Haaland         # force players into the squad
--exclude "Son"              # keep players out
--max-per-team 3             # club limit
-o report.md                 # write a full Markdown report
--no-reddit                  # skip community analysis
--weight-model 0.8 --weight-form 0.1 --weight-ep 0.1   # retune the blend
```

## How the projection works

For each of a player's remaining fixtures in the horizon:

```
expected points = P(start) x [ appearance
                             + xG/90 x goal points(position)
                             + xA/90 x 3
                             + P(clean sheet) x CS points(position)
                             - xGC/2                (GK and DEF only)
                             + saves/90 / 3         (GK only)
                             + bonus/90
                             + defensive contribution ]
```

with these adjustments:

- **Small samples are shrunk** toward positional priors, using 270 minutes of
  prior evidence. A player with one good game does not top the table.
- **Fixtures scale the attacking rates** via a geometric blend of team strength
  ratings and FDR, clamped so no single fixture dominates, with a home bonus.
- **Clean sheets are Poisson** — `P(CS) = e^(-xGC)`, where expected goals
  conceded comes from the opponent's attack against this team's defence.
- **Minutes drive everything.** Start probability blends start rate with minutes
  rate, multiplied by availability from `status` and `chance_of_playing`.
- **The model is blended** with FPL's own `ep_next` and the player's `form`
  (default 60/20/20), and those anchors are themselves scaled by expected
  minutes so fringe players are not over-rated.

Tune the blend with `--weight-model`, `--weight-form` and `--weight-ep`.

## How the optimiser works

A single ILP maximises

```
sum over players of  xPTS x (in_XI + bench_weight x (in_squad - in_XI) + is_captain)
```

subject to 15 players, the 2/5/5/3 quota, the budget, ≤3 per club, exactly 11
starters in a legal formation, and one captain who must be a starter. Squad, XI
and armband are chosen *together* — picking the squad first and the XI second
gives a worse answer.

## Reddit analysis

Threads are scored by `engagement x topical weight x recency` (a 3-day
half-life — FPL advice goes stale fast). Topical weight favours team news,
press conferences, injuries and captaincy over generic chat.

Player mentions are matched with word boundaries against web names, surnames
and full names. Two guards keep this honest:

- Surnames that collide with ordinary English (`King`, `Rice`, `Ward`, `Long`…)
  only count when **capitalised**, so "his price is rising" is not a Rice mention.
- Sentiment is scoped to the **clause** a player is named in, so
  "Salah is elite. Nunez is a trap." does not make Salah look bearish.

Sentiment uses an FPL-specific lexicon (`essential`, `nailed`, `haul` vs
`trap`, `rotation`, `ditch`…). It is a keyword reading of public comments — it
does not understand sarcasm, and it is a *signal*, not an oracle.

### If Reddit returns 403

Reddit blocks anonymous JSON from many datacenter and cloud IPs. Create a free
"script" app at <https://www.reddit.com/prefs/apps> and export:

```bash
export REDDIT_CLIENT_ID=...
export REDDIT_CLIENT_SECRET=...
```

The client then uses OAuth automatically. Or pass `--no-reddit` to skip it —
every other feature works without Reddit.

## Working offline

```bash
fpl-builder fetch --data-dir fpl-data      # save the API payloads
fpl-builder build --data-dir fpl-data      # replay with no network
```

Responses are cached for an hour by default (60 seconds for live gameweek
data). `--refresh` forces a refetch.

## Endpoints used

| Endpoint | Used for |
|---|---|
| `bootstrap-static/` | players, teams, gameweeks, scoring |
| `fixtures/` | all fixtures, kickoff times, FDR |
| `element-summary/{id}/` | per-player match history |
| `event/{gw}/live/` | live points during a gameweek |
| `entry/{id}/` | manager summary and bank |
| `entry/{id}/event/{gw}/picks/` | your 15 picks |

## Tests

```bash
python -m pytest -q
```

105 tests cover parsing, the projection maths, squad legality under every
constraint, ILP-vs-heuristic optimality, Reddit parsing and sentiment scoping,
and end-to-end CLI runs. They use a generated dataset that mirrors the real API
schema, so the suite runs with no network access:

```bash
python scripts/generate_sample_data.py tests/fixtures
```

## Caveats

- Projections are **model estimates, not predictions**. Minutes are the largest
  source of error; the model cannot know a manager's team sheet.
- Reddit sentiment is automated keyword matching and misreads sarcasm.
- Price-change prediction and chip-timing strategy are not implemented.
- Fixture difficulty uses FPL's own FDR plus team strength ratings, which are
  coarse and update slowly.

## Licence

MIT
