"""Live team & player enrichment from API-Football (free tier).

Two sharpeners, both fallback-safe (any miss -> the engine's base behavior):

* team_rate(code): empirical-Bayes per-match rates (shots-on-target, corners,
  offsides, fouls, cards) built from a *rolling accumulator* of in-window
  finished World Cup fixtures. The free plan only exposes a ~3-day date window,
  so we accumulate finished-fixture stats day by day into data/team_stats.json
  and shrink each team toward the league base by games seen. Thin early in the
  tournament, sharper as the window fills.

* player_info(name, home, away): which side a named player is on, their position
  group, and whether they START (from confirmed lineups, fetched only inside the
  pre-kickoff window). Turns flat player-prop base rates into start/position/
  team-aware estimates.

Budget: Pro tier (7500 req/day). We still cache hard (squads 24h, fixture map 6h,
lineups 10m) and fetch lineups only near kickoff (steady usage ~30-80/day); a
one-time backfill_team_accumulator() sweep folds the full WC history. Everything
is wrapped so a failure never propagates into the daemon loop.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from unidecode import unidecode

from . import config, model
from .httputil import get_json, read_cache, write_cache
from .logging_setup import get_logger
from .mapping import resolve_team

log = get_logger("jumpbot.stats")

_BASE = "https://v3.football.api-sports.io"
_WC_LEAGUE = 1
_ACCUM_FILE = config.DATA_DIR / "team_stats.json"

_ENABLED = bool(config.APIFOOTBALL_KEY)
_HDR = {"x-apisports-key": config.APIFOOTBALL_KEY} if _ENABLED else {}

# In-memory caches (per process); disk caches back them across restarts.
_fixture_map: Dict[Tuple[str, str], dict] = {}
_fixture_map_ts = 0.0
_squad_cache: Dict[int, Dict[str, str]] = {}
_lineup_cache: Dict[int, dict] = {}  # fixture_id -> {ts, home:{starters,formation}, away:{...}}

# Soft daily budget so a bug can't burn the quota.
_calls_today = 0
_calls_day = ""
_CALL_CAP = 1000  # Pro (7500/day); generous ceiling, real usage stays ~30-80/day
# Free tier is rate-limited per minute; throttle so bursts don't 429.
_SQUAD_WINDOW_S = 30 * 3600  # only resolve players for matches this soon


class _RateLimiter:
    def __init__(self, n: int, w: float):
        self.n, self.w, self.q, self.lock = n, w, deque(), threading.Lock()

    def acquire(self) -> None:
        with self.lock:
            now = time.time()
            while self.q and now - self.q[0] > self.w:
                self.q.popleft()
            if len(self.q) >= self.n:
                time.sleep(max(self.w - (now - self.q[0]) + 0.1, 0))
                now = time.time()
                while self.q and now - self.q[0] > self.w:
                    self.q.popleft()
            self.q.append(time.time())


_LIMITER = _RateLimiter(150, 60.0)  # 150/min — Pro tier (7500/day); was 8/min (free)


def _api(path: str, params: dict) -> Optional[dict]:
    global _calls_today, _calls_day
    if not _ENABLED:
        return None
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if day != _calls_day:
        _calls_day, _calls_today = day, 0
    if _calls_today >= _CALL_CAP:
        log.warning("stats: daily call cap reached (%d); using fallbacks", _CALL_CAP)
        return None
    try:
        _LIMITER.acquire()
        _calls_today += 1
        return get_json(f"{_BASE}{path}", params=params, headers=_HDR)
    except Exception as exc:  # noqa: BLE001
        log.warning("stats api %s failed: %s", path, exc)
        return None


def _norm(name: str) -> str:
    return unidecode(name or "").lower().strip()


def _last(name: str) -> str:
    n = _norm(name)
    return n.split()[-1] if n else ""


# --- fixture map -----------------------------------------------------------

def _ensure_fixture_map(now: float) -> None:
    """Map (home_code, away_code) -> {fixture_id, home_id, away_id} for nearby days."""
    global _fixture_map, _fixture_map_ts
    if _fixture_map and now - _fixture_map_ts < 6 * 3600:
        return
    cached = read_cache("af_fixture_map.json", 6 * 3600)
    if cached:
        _fixture_map = {tuple(k.split("|")): v for k, v in cached.items()}
        _fixture_map_ts = now
        return
    today = datetime.now(timezone.utc).date()
    mp: Dict[str, dict] = {}
    # +2/+3 too: career-prior pid resolution needs team ids for every OPEN match
    # being priced, and the board floors matches 2-3 days ahead of lock.
    for delta in (-1, 0, 1, 2, 3):
        d = (today + timedelta(days=delta)).strftime("%Y-%m-%d")
        data = _api("/fixtures", {"date": d})
        if not data:
            continue
        for f in data.get("response", []):
            if f.get("league", {}).get("id") != _WC_LEAGUE:
                continue
            hc = resolve_team(f["teams"]["home"]["name"])
            ac = resolve_team(f["teams"]["away"]["name"])
            if hc and ac:
                mp[f"{hc}|{ac}"] = {
                    "fixture_id": f["fixture"]["id"],
                    "home_id": f["teams"]["home"]["id"],
                    "away_id": f["teams"]["away"]["id"],
                    "status": f["fixture"]["status"]["short"],
                    "referee": f["fixture"].get("referee"),
                }
    if mp:
        write_cache("af_fixture_map.json", mp)
        _fixture_map = {tuple(k.split("|")): v for k, v in mp.items()}
        _fixture_map_ts = now


def _fixture_for(home_code: str, away_code: str) -> Optional[dict]:
    return _fixture_map.get((home_code, away_code))


def match_referee(home_code: str, away_code: str) -> Optional[str]:
    """Assigned referee name for a mapped match (fixture.referee, set once the
    appointment is announced — typically a day or two pre-KO), or None. Reuses the
    cached fixture map, so no extra API call."""
    fx = _fixture_map.get((home_code, away_code))
    return fx.get("referee") if fx else None


# --- team-stats rolling accumulator ---------------------------------------

_accum_cache: Optional[dict] = None
_league_cache: Optional[Dict[str, float]] = None  # empirical league per-game means


def _load_accum() -> dict:
    global _accum_cache
    if _accum_cache is not None:
        return _accum_cache
    try:
        with open(_ACCUM_FILE) as f:
            _accum_cache = json.load(f)
    except (OSError, json.JSONDecodeError):
        _accum_cache = {"processed_fixtures": [], "teams": {}}
    return _accum_cache


def _save_accum(acc: dict) -> None:
    global _accum_cache, _league_cache
    _accum_cache = acc
    _league_cache = None  # accumulator changed -> recompute league means lazily
    tmp = _ACCUM_FILE.with_suffix(".tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(acc, f)
        tmp.replace(_ACCUM_FILE)
    except OSError as exc:
        log.warning("accum save failed: %s", exc)


_METRICS = {"Shots on Goal": "sot", "Corner Kicks": "corners",
            "Offsides": "offsides", "Fouls": "fouls"}


def _new_team_rec() -> dict:
    """Fresh per-team record: own counts plus conceded ('_against') counts."""
    rec = {"games": 0}
    for k in (*_METRICS.values(), "cards"):
        rec[k] = 0
        rec[k + "_against"] = 0
    return rec


def _fold_fixture_stats(blocks_resp, teams) -> None:
    """Fold one fixture's /fixtures/statistics response into the accumulator.

    The response carries one stat block per team; pairing the two lets each team
    accrue both its own counts and the opponent's as conceded ('_against'), which
    is what the compare markets need to opponent-adjust (corners-for vs -against).
    Backward-compatible: legacy records missing the '_against' keys are upgraded
    in place via setdefault, so re-runs over an old accumulator never KeyError."""
    parsed = []
    for tb in blocks_resp or []:
        code = resolve_team(tb["team"]["name"])
        if not code:
            continue
        parsed.append((code, {x["type"]: x["value"] for x in tb["statistics"]}))
    for i, (code, s) in enumerate(parsed):
        opp = parsed[1 - i][1] if len(parsed) == 2 else {}
        rec = teams.setdefault(code, _new_team_rec())
        rec.setdefault("games", 0)
        for our_key in (*_METRICS.values(), "cards"):
            rec.setdefault(our_key, 0)
            rec.setdefault(our_key + "_against", 0)
        rec["games"] += 1
        for api_key, our_key in _METRICS.items():
            rec[our_key] += (s.get(api_key) or 0)
            rec[our_key + "_against"] += (opp.get(api_key) or 0)
        rec["cards"] += (s.get("Yellow Cards") or 0) + (s.get("Red Cards") or 0)
        rec["cards_against"] += (opp.get("Yellow Cards") or 0) + (opp.get("Red Cards") or 0)


# --- referee discipline accumulator (sharpens ABSOLUTE total_cards only) -----
# Referee identity barely moves the *compare* markets — a stricter ref cards both
# teams, so "X gets more cards than Y" is ~unchanged — but it does move the
# absolute total-cards line. WC-only data is thin (~1.5 games/ref), so
# referee_rate() EB-shrinks hard toward the league mean (K tuned to the measured
# signal-to-noise) => strictly no-harm, sharpening only as a ref's game count
# grows through the tournament. Folded for free from the same /fixtures/statistics
# we already pull for team rates; the assigned referee comes from fixture.referee.
_REF_EB = 4.0  # prior weight in games; from between-ref vs sampling variance on cards


def _ref_key(name: str) -> str:
    """Normalized referee key: drop the ', Country' suffix, unidecode + lower.
    So an accumulated 'Danny Makkelie, Netherlands' and an assignment 'D. Makkelie'
    both reduce to a stable last-name-matchable form."""
    return _norm((name or "").split(",")[0])


def _fold_fixture_referee(referee: Optional[str], blocks, referees: dict) -> None:
    """Fold one fixture's total cards & fouls under its referee."""
    key = _ref_key(referee or "")
    if not key:
        return
    cards = fouls = 0
    for tb in blocks or []:
        d = {x["type"]: (x["value"] or 0) for x in tb.get("statistics", [])}
        cards += (d.get("Yellow Cards") or 0) + (d.get("Red Cards") or 0)
        fouls += (d.get("Fouls") or 0)
    rec = referees.setdefault(key, {"g": 0, "cards": 0, "fouls": 0, "name": referee})
    for fld in ("g", "cards", "fouls"):
        rec.setdefault(fld, 0)
    rec["g"] += 1
    rec["cards"] += cards
    rec["fouls"] += fouls


def update_team_accumulator(now: float, force: bool = False) -> None:
    """Fetch in-window finished WC fixtures' stats and fold into the accumulator.

    Runs at most once every 6h (cheap, idempotent via processed_fixtures) unless
    ``force`` — the settlement detector folds immediately when a game ends."""
    acc = _load_accum()
    last = acc.get("last_update", 0)
    if not force and now - last < 6 * 3600:
        return
    today = datetime.now(timezone.utc).date()
    processed = set(acc.get("processed_fixtures", []))
    ref_processed = set(acc.get("ref_processed", []))
    teams = acc.get("teams", {})
    referees = acc.get("referees", {})
    for delta in (-1, 0):  # in-window dates only (free tier ~ today +/- 1)
        d = (today + timedelta(days=delta)).strftime("%Y-%m-%d")
        data = _api("/fixtures", {"date": d})
        if not data:
            continue
        for f in data.get("response", []):
            if f.get("league", {}).get("id") != _WC_LEAGUE:
                continue
            if f["fixture"]["status"]["short"] not in ("FT", "AET", "PEN"):
                continue
            fid = f["fixture"]["id"]
            need_team, need_ref = fid not in processed, fid not in ref_processed
            if not need_team and not need_ref:
                continue
            st = _api("/fixtures/statistics", {"fixture": fid})
            if not st or not st.get("response"):
                continue
            if need_team:
                _fold_fixture_stats(st["response"], teams)
                processed.add(fid)
            if need_ref:
                _fold_fixture_referee(f["fixture"].get("referee"), st["response"], referees)
                ref_processed.add(fid)
    acc.update({"processed_fixtures": sorted(processed), "ref_processed": sorted(ref_processed),
                "teams": teams, "referees": referees, "last_update": now})
    _save_accum(acc)
    log.info("team accumulator: %d teams, %d fixtures processed", len(teams), len(processed))


def backfill_team_accumulator() -> dict:
    """One-time Pro-tier sweep: fold EVERY finished WC fixture's stats into the
    accumulator (not just the free-tier ~3-day window). One /fixtures call for the
    whole season, then /fixtures/statistics per finished fixture. Idempotent via
    processed_fixtures, so re-running only adds newly-finished games."""
    acc = _load_accum()
    processed = set(acc.get("processed_fixtures", []))
    ref_processed = set(acc.get("ref_processed", []))
    teams = acc.get("teams", {})
    referees = acc.get("referees", {})
    data = _api("/fixtures", {"league": _WC_LEAGUE, "season": 2026})
    fixtures = (data or {}).get("response", [])
    finished = [f for f in fixtures
                if f["fixture"]["status"]["short"] in ("FT", "AET", "PEN")]
    n_new = 0
    for f in finished:
        fid = f["fixture"]["id"]
        need_team, need_ref = fid not in processed, fid not in ref_processed
        if not need_team and not need_ref:
            continue
        st = _api("/fixtures/statistics", {"fixture": fid})
        if not st or not st.get("response"):
            continue
        if need_team:
            _fold_fixture_stats(st["response"], teams)
            processed.add(fid)
            n_new += 1
        if need_ref:
            _fold_fixture_referee(f["fixture"].get("referee"), st["response"], referees)
            ref_processed.add(fid)
    acc.update({"processed_fixtures": sorted(processed), "ref_processed": sorted(ref_processed),
                "teams": teams, "referees": referees, "last_update": time.time()})
    _save_accum(acc)
    log.info("backfill: %d WC fixtures (%d finished), %d newly folded, %d teams, %d referees",
             len(fixtures), len(finished), n_new, len(teams), len(referees))
    return acc


def finished_fixture_ids() -> set:
    """The set of finished WC fixture ids — one cheap /fixtures call. The daemon's
    settlement detector diffs this each cycle to fold calibration the moment a game
    ends, instead of waiting on the slow fixed-interval refreshes."""
    data = _api("/fixtures", {"league": _WC_LEAGUE, "season": 2026})
    return {f["fixture"]["id"] for f in (data or {}).get("response", [])
            if f["fixture"]["status"]["short"] in ("FT", "AET", "PEN")}


# --- per-player shots-on-target accumulator (player_sot differentiation) -----
_PLAYER_FILE = config.DATA_DIR / "player_stats.json"
_player_cache: Optional[dict] = None
# position prior SoT/90 = -ln(1 - P(>=1 SoT)); EB shrinks a player toward this.
_POS_SOT90_PRIOR = {"att": 0.97, "mid": 0.54, "def": 0.22, "gk": 0.02}
_PLAYER_EB_K = 1.5  # prior weight, in 90-min appearances
# goals & (goals+assists) per 90 priors; rarer/noisier than SoT -> shrink harder.
_POS_GOAL90_PRIOR = {"att": 0.42, "mid": 0.16, "def": 0.05, "gk": 0.004}
_POS_INV90_PRIOR = {"att": 0.62, "mid": 0.30, "def": 0.10, "gk": 0.008}
_GOAL_EB_K = 2.0
_FIXPOS = {"G": "gk", "D": "def", "M": "mid", "F": "att"}  # /fixtures/players code


def _load_players() -> dict:
    global _player_cache
    if _player_cache is not None:
        return _player_cache
    try:
        with open(_PLAYER_FILE) as f:
            _player_cache = json.load(f)
    except (OSError, json.JSONDecodeError):
        _player_cache = {"processed_fixtures": [], "players": {}}
    return _player_cache


def _save_players(acc: dict) -> None:
    global _player_cache
    _player_cache = acc
    tmp = _PLAYER_FILE.with_suffix(".tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(acc, f)
        tmp.replace(_PLAYER_FILE)
    except OSError as exc:
        log.warning("player save failed: %s", exc)


def backfill_player_accumulator() -> dict:
    """Sweep finished WC fixtures' per-player box-score lines (shots-on-target,
    goals, assists, minutes, modal position) into a per-player rate accumulator
    (Pro tier; /fixtures/players). Idempotent via processed_fixtures."""
    acc = _load_players()
    processed = set(acc.get("processed_fixtures", []))
    players = acc.get("players", {})
    data = _api("/fixtures", {"league": _WC_LEAGUE, "season": 2026})
    finished = [f for f in (data or {}).get("response", [])
                if f["fixture"]["status"]["short"] in ("FT", "AET", "PEN")]
    n_new = 0
    for f in finished:
        fid = f["fixture"]["id"]
        if fid in processed:
            continue
        pd = _api("/fixtures/players", {"fixture": fid})
        if not pd or not pd.get("response"):
            continue
        for tb in pd["response"]:
            for p in tb.get("players", []):
                st = (p.get("statistics") or [{}])[0]
                games = st.get("games") or {}
                mins = games.get("minutes") or 0
                if not mins:
                    continue
                son = (st.get("shots") or {}).get("on") or 0
                gd = st.get("goals") or {}
                nm = _norm(p["player"]["name"])
                rec = players.setdefault(nm, {"sot": 0, "minutes": 0, "matches": 0,
                                              "goals": 0, "assists": 0, "pos": None, "pmin": 0})
                rec["sot"] += son
                rec["minutes"] += mins
                rec["matches"] += 1
                rec["goals"] += gd.get("total") or 0
                rec["assists"] += gd.get("assists") or 0
                posc = _FIXPOS.get((games.get("position") or " ")[:1].upper())
                if posc and mins > rec.get("pmin", 0):  # modal position by minutes
                    rec["pos"], rec["pmin"] = posc, mins
        processed.add(fid)
        n_new += 1
    acc.update({"processed_fixtures": sorted(processed), "players": players})
    _save_players(acc)
    log.info("player backfill: %d finished fixtures, %d players, %d newly folded",
             len(finished), len(players), n_new)
    return acc


def _player_rec(name: str) -> Optional[dict]:
    """Resolve a name -> accumulated record (exact, then last-name fallback)."""
    players = _load_players().get("players", {})
    rec = players.get(_norm(name))
    if rec is None:
        ln = _last(name)
        for k, v in players.items():
            if ln and k.split()[-1:] == [ln]:
                rec = v
                break
    return rec if (rec and rec.get("minutes", 0) > 0) else None


# --- career priors (last completed club season; the "Bellingham fix") -------
# The this-WC accumulator is blind to career class: 270 cold minutes crushed
# Semenyo's 2+SoT to .02 (field .33), and pre-game Bellingham's model side to
# ~.20 while the field priced his career at .40 (settled YES, -23 RBP). Blend
# the player's LAST-SEASON per-90 rates (all competitions, club + country) into
# the EB prior and give that prior real pseudo-count weight: a full season is
# worth far more than 2-4 tournament games, but K stops short of drowning
# genuine tournament signal (role change, form, minutes share).
_CAREER_W = 0.7        # career share inside the prior (rest = position mean)
_CAREER_SOT_K = 5.0    # prior pseudo-90s when career-backed (legacy 1.5 without)
_CAREER_GOAL_K = 6.0   # goals/involve are rarer -> lean harder on the season rate
_CAREER_MIN_MINUTES = 500  # below this a season rate is noise -> unusable
_CAREER_FETCH_CAP = 12     # max NEW career fetches per pricing cycle (budget guard)
_career_season: Optional[int] = None  # probed once: 2025 if the plan serves it, else 2024
_career_fetches = 0
_squadids_cache: Dict[int, Dict[str, int]] = {}


def reset_cycle_budget() -> None:
    """Called at the top of each engine cycle; re-arms the career fetch cap."""
    global _career_fetches
    _career_fetches = 0


def _squad_ids(team_id: int) -> Dict[str, int]:
    """norm name -> af player id for a squad (the squads endpoint carries ids;
    the pos-only _squad cache predates needing them)."""
    if team_id in _squadids_cache:
        return _squadids_cache[team_id]
    cached = read_cache(f"af_squadids_{team_id}.json", 7 * 24 * 3600)
    if cached is None:
        data = _api("/players/squads", {"team": team_id})
        if not data or not data.get("response"):
            return {}  # don't cache a failed fetch
        cached = {}
        for tb in data.get("response", []):
            for p in tb.get("players", []):
                if p.get("id"):
                    cached[_norm(p["name"])] = p["id"]
        write_cache(f"af_squadids_{team_id}.json", cached)
    _squadids_cache[team_id] = cached
    return cached


def _career(pid: int) -> Optional[dict]:
    """Last-season per-90 rates for one player, aggregated across ALL competitions
    ({} is cached as known-missing so we don't re-ask for 30 days)."""
    global _career_season, _career_fetches
    cached = read_cache(f"af_career_{pid}.json", 30 * 24 * 3600)
    if cached is not None:
        return cached or None
    if _career_fetches >= _CAREER_FETCH_CAP:
        return None  # over this cycle's budget; a later cycle picks it up
    _career_fetches += 1
    for season in ([_career_season] if _career_season else [2025, 2024]):
        data = _api("/players", {"id": pid, "season": season})
        resp = (data or {}).get("response") or []
        if not resp:
            continue
        _career_season = _career_season or season
        mins = g = a = sot = sot_min = 0
        for st in resp[0].get("statistics") or []:
            m = (st.get("games") or {}).get("minutes") or 0
            if m <= 0:
                continue
            mins += m
            gd = st.get("goals") or {}
            g += gd.get("total") or 0
            a += gd.get("assists") or 0
            son = (st.get("shots") or {}).get("on")
            if son is not None:  # competitions without shot data don't dilute the rate
                sot += son
                sot_min += m
        out = {"season": season, "minutes": mins,
               "sot90": round(sot / sot_min * 90.0, 3) if sot_min >= _CAREER_MIN_MINUTES else None,
               "goal90": round(g / mins * 90.0, 3) if mins >= _CAREER_MIN_MINUTES else None,
               "inv90": round((g + a) / mins * 90.0, 3) if mins >= _CAREER_MIN_MINUTES else None}
        write_cache(f"af_career_{pid}.json", out)
        log.info("career %s season %s: %dmin sot90=%s goal90=%s", pid, season,
                 mins, out["sot90"], out["goal90"])
        return out
    write_cache(f"af_career_{pid}.json", {})
    return None


def career_rates(name: str, home_code: Optional[str] = None,
                 away_code: Optional[str] = None) -> Optional[dict]:
    """{sot90, goal90, inv90} from the player's last season, or None. Resolves the
    af player id via the two squads' id maps (exact name, then last-name)."""
    if not _ENABLED or not name or not (home_code and away_code):
        return None
    try:
        _ensure_fixture_map(time.time())
        fx = _fixture_for(home_code, away_code)
        if not fx:
            return None
        nm, ln = _norm(name), _last(name)
        pid = None
        for tid in (fx["home_id"], fx["away_id"]):
            ids = _squad_ids(tid)
            pid = ids.get(nm)
            if pid is None and ln:
                for k, v in ids.items():
                    if k.split()[-1:] == [ln]:
                        pid = v
                        break
            if pid:
                break
        return _career(pid) if pid else None
    except Exception as exc:  # noqa: BLE001 - prior enrichment must never block pricing
        log.debug("career_rates(%s) failed: %s", name, exc)
        return None


def _eb_rate(rec: Optional[dict], obs_key, career_val: Optional[float],
             pos_prior: float, legacy_k: float, career_k: float) -> Optional[float]:
    """EB-shrunk per-90 rate: observed this-WC counts against a prior that is the
    position mean, upgraded to a career-blended prior (with more pseudo-weight)
    when last-season data exists. Career-only players (no WC minutes yet) price
    at the prior itself — far better than the flat position share."""
    if rec is None and career_val is None:
        return None
    prior, k = pos_prior, legacy_k
    if career_val is not None:
        prior, k = _CAREER_W * career_val + (1.0 - _CAREER_W) * pos_prior, career_k
    obs = obs_key(rec) if rec else 0
    n90 = rec["minutes"] / 90.0 if rec else 0.0
    return (obs + prior * k) / (n90 + k)


def player_sot_rate(name: str, pos: str = "att", home_code: Optional[str] = None,
                    away_code: Optional[str] = None) -> Optional[float]:
    """EB-shrunk shots-on-target per 90 min for a player (career-informed prior
    when available), or None if the player has no data at all.
    Prefers the player's own modal position over the caller's `pos` hint."""
    rec = _player_rec(name)
    car = career_rates(name, home_code, away_code)
    prior = _POS_SOT90_PRIOR.get((rec or {}).get("pos") or pos, 0.54)
    return _eb_rate(rec, lambda r: r["sot"], (car or {}).get("sot90"),
                    prior, _PLAYER_EB_K, _CAREER_SOT_K)


def player_goal_rate(name: str, pos: str = "att", home_code: Optional[str] = None,
                     away_code: Optional[str] = None) -> Optional[float]:
    """EB-shrunk goals per 90 min for a player, or None if no data."""
    rec = _player_rec(name)
    car = career_rates(name, home_code, away_code)
    prior = _POS_GOAL90_PRIOR.get((rec or {}).get("pos") or pos, 0.16)
    return _eb_rate(rec, lambda r: r.get("goals", 0), (car or {}).get("goal90"),
                    prior, _GOAL_EB_K, _CAREER_GOAL_K)


def player_involve_rate(name: str, pos: str = "att", home_code: Optional[str] = None,
                        away_code: Optional[str] = None) -> Optional[float]:
    """EB-shrunk (goals + assists) per 90 min for a player, or None if no data."""
    rec = _player_rec(name)
    car = career_rates(name, home_code, away_code)
    prior = _POS_INV90_PRIOR.get((rec or {}).get("pos") or pos, 0.30)
    return _eb_rate(rec, lambda r: r.get("goals", 0) + r.get("assists", 0),
                    (car or {}).get("inv90"), prior, _GOAL_EB_K, _CAREER_GOAL_K)


# League base means (per team per match) the EB prior shrinks toward.
_LEAGUE_BASE = {
    "sot": model.BASE["team_sot"], "corners": model.BASE["team_corners"],
    "offsides": model.BASE["team_offsides"], "fouls": model.BASE["team_fouls"],
    "cards": model.BASE["team_cards"],
}
_EB_STRENGTH = 3.0  # prior weight in "games"; >=3 real games before we lean on data


def league_rate(metric: str) -> float:
    """Empirical league average per-team-per-game for a metric. By symmetry this
    is both the average 'for' and 'against', so it's the right normalizer for the
    opponent-adjusted (attack x defense) compare pricing. Falls back to the model
    base before any data exists. Cached; invalidated on _save_accum."""
    global _league_cache
    if _league_cache is None:
        acc = _load_accum()
        teams = acc.get("teams", {})
        tot_g = sum(r.get("games", 0) for r in teams.values())
        _league_cache = {}
        for key in _LEAGUE_BASE:
            tot = sum(r.get(key, 0) for r in teams.values())
            _league_cache[key] = (tot / tot_g) if tot_g > 0 else _LEAGUE_BASE[key]
    return _league_cache.get(metric) or _LEAGUE_BASE.get(metric, 1.0)


def team_rate(code: str) -> Optional[Dict[str, float]]:
    """Empirical-Bayes per-match rate means for a team, or None if no data.

    Includes both 'for' rates (e.g. 'corners') and conceded rates
    ('corners_against'), each EB-shrunk toward the league mean so a 2-game sample
    can't swing the opponent-adjustment too hard."""
    acc = _load_accum()
    rec = acc.get("teams", {}).get(code)
    if not rec or rec.get("games", 0) <= 0:
        return None
    g = rec["games"]
    out = {}
    for key, base in _LEAGUE_BASE.items():
        observed = rec.get(key, 0) / g
        out[key] = (observed * g + base * _EB_STRENGTH) / (g + _EB_STRENGTH)
        ag_base = league_rate(key)
        ag_obs = rec.get(key + "_against", 0) / g
        out[key + "_against"] = (ag_obs * g + ag_base * _EB_STRENGTH) / (g + _EB_STRENGTH)
    out["games"] = g
    return out


def referee_rate(name: str) -> Optional[Dict[str, float]]:
    """EB-shrunk referee discipline for an assigned referee, or None if we have no
    referee data at all. Returns the per-game cards mean and a ``cards_mult`` (ref
    cards/game ÷ league cards/game) for tilting the absolute total-cards lambda.

    The shrink (toward the league mean, weight ``_REF_EB``) is deliberately heavy:
    measured between-ref card variance is only ~0.6 over a sampling-noise floor of
    ~1.85 at these game counts, so a thin ref barely moves off the mean and a
    well-sampled one moves proportionally. Matches an abbreviated assignment
    ('I. Elfath') to an accumulated full name by last name."""
    referees = _load_accum().get("referees", {})
    if not referees:
        return None
    key = _ref_key(name)
    rec = referees.get(key)
    if rec is None:  # last-name fallback: 'i. elfath' -> 'ismail elfath'
        ln = key.split()[-1] if key else ""
        for k, v in referees.items():
            if ln and k.split()[-1:] == [ln]:
                rec = v
                break
    if not rec or rec.get("g", 0) <= 0:
        return None
    tot_g = sum(r.get("g", 0) for r in referees.values())
    lg_cards = (sum(r.get("cards", 0) for r in referees.values()) / tot_g) if tot_g else 2.5
    lg_fouls = (sum(r.get("fouls", 0) for r in referees.values()) / tot_g) if tot_g else 22.0
    g = rec["g"]
    cards_pg = (rec.get("cards", 0) + lg_cards * _REF_EB) / (g + _REF_EB)
    fouls_pg = (rec.get("fouls", 0) + lg_fouls * _REF_EB) / (g + _REF_EB)
    mult = (cards_pg / lg_cards) if lg_cards > 0 else 1.0
    return {"cards_pg": cards_pg, "fouls_pg": fouls_pg, "lg_cards": lg_cards,
            "lg_fouls": lg_fouls, "games": g, "cards_mult": max(0.75, min(1.35, mult))}


# --- squads & lineups (player resolution) ----------------------------------

_POS = {"Attacker": "att", "Midfielder": "mid", "Defender": "def", "Goalkeeper": "gk"}


def _squad(team_id: int) -> Dict[str, str]:
    if team_id in _squad_cache:
        return _squad_cache[team_id]
    cached = read_cache(f"af_squad_{team_id}.json", 24 * 3600)
    if cached is None:
        data = _api("/players/squads", {"team": team_id})
        if not data or not data.get("response"):
            return {}  # don't cache a failed fetch — retry next cycle
        cached = {}
        for tb in data.get("response", []):
            for p in tb.get("players", []):
                cached[_norm(p["name"])] = _POS.get(p.get("position", ""), "mid")
        write_cache(f"af_squad_{team_id}.json", cached)
    _squad_cache[team_id] = cached
    return cached


_player_stats_cache: Dict[int, Dict[str, list]] = {}


def _player_stats(team_id: int) -> Dict[str, list]:
    """Per-player [goals, appearances] from the latest accessible season (2024).

    Free tier blocks 2026, but a player's recent national-team scoring rate is a
    solid prior for scorer/involve props (far better than a flat position share).
    """
    if team_id in _player_stats_cache:
        return _player_stats_cache[team_id]
    cached = read_cache(f"af_pstats_{team_id}.json", 24 * 3600)
    if cached is None:
        data = _api("/players", {"team": team_id, "season": 2024, "page": 1})
        if not data or not data.get("response"):
            return {}
        cached = {}

        def _ingest(resp):
            for item in resp:
                nm = _norm(item["player"]["name"])
                st = (item.get("statistics") or [{}])[0]
                goals = (st.get("goals") or {}).get("total") or 0
                apps = (st.get("games") or {}).get("appearences") or 0
                cached[nm] = [goals, apps]

        _ingest(data["response"])
        total_pages = (data.get("paging") or {}).get("total", 1)
        for pg in range(2, min(total_pages, 4) + 1):  # cap pages to bound budget
            d2 = _api("/players", {"team": team_id, "season": 2024, "page": pg})
            if d2 and d2.get("response"):
                _ingest(d2["response"])
        write_cache(f"af_pstats_{team_id}.json", cached)
    _player_stats_cache[team_id] = cached
    return cached


def _lineup(fixture_id: int, now: float) -> Optional[dict]:
    cur = _lineup_cache.get(fixture_id)
    if cur and now - cur["ts"] < 600:
        return cur
    data = _api("/fixtures/lineups", {"fixture": fixture_id})
    if not data or not data.get("response"):
        return None
    out = {"ts": now, "sides": {}}
    for i, tb in enumerate(data["response"]):
        starters = {_norm(p["player"]["name"]) for p in tb.get("startXI", [])}
        out["sides"][i] = {"starters": starters, "team_id": tb["team"]["id"],
                           "formation": tb.get("formation")}
    _lineup_cache[fixture_id] = out
    return out


def prefetch(matches: List[dict], now: float, lineup_window_s: float) -> None:
    """Per-cycle warm-up: fixture map, accumulator, squads + lineups near KO."""
    if not _ENABLED:
        return
    try:
        _ensure_fixture_map(now)
        update_team_accumulator(now)
        for m in matches:
            hc, ac, lock = m.get("home_code"), m.get("away_code"), m.get("lock_ts")
            if not (hc and ac) or not lock:
                continue
            ttl = lock - now
            # Only resolve players for near-term matches (keeps the call burst small).
            if ttl <= 0 or ttl > _SQUAD_WINDOW_S:
                continue
            fx = _fixture_for(hc, ac)
            if not fx:
                continue
            _squad(fx["home_id"])   # cached 24h
            _squad(fx["away_id"])
            if 0 < ttl <= lineup_window_s:   # lineups only inside the pre-KO window
                _lineup(fx["fixture_id"], now)
    except Exception as exc:  # noqa: BLE001 - never break the loop
        log.warning("stats.prefetch failed: %s", exc)


def player_info(name: str, home_code: str, away_code: str) -> Optional[dict]:
    """Resolve a named player -> {side, pos, starts}. None if unresolved.

    starts: True/False if a lineup is loaded, else None (squad-only).
    """
    if not _ENABLED:
        return None
    fx = _fixture_for(home_code, away_code)
    if not fx:
        return None
    nm, ln = _norm(name), _last(name)
    sides = {"home": fx["home_id"], "away": fx["away_id"]}
    found_side = found_pos = None
    for side, tid in sides.items():
        sq = _squad_cache.get(tid) or {}
        if nm in sq:
            found_side, found_pos = side, sq[nm]
            break
        # last-name fallback
        for full, pos in sq.items():
            if full.split()[-1:] == [ln] and ln:
                found_side, found_pos = side, pos
                break
        if found_side:
            break
    if not found_side:
        return None
    starts = None
    lu = _lineup_cache.get(fx["fixture_id"])
    if lu:
        tid = sides[found_side]
        for blk in lu["sides"].values():
            if blk["team_id"] == tid:
                starts = (nm in blk["starters"]) or any(
                    s.split()[-1:] == [ln] for s in blk["starters"])
                break
    return {"side": found_side, "pos": found_pos, "starts": starts}
