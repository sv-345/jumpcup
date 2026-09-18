"""the-odds-api client: pull prices, de-vig, expose fair probabilities.

Budget-aware: the free tier is 500 credits, so a single /odds call (which
returns every match for the sport at once) is cached and reused across loop
ticks, refreshing faster only inside the pre-kickoff window. Every result
carries a fetch timestamp so the engine can detect staleness.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from statistics import median
from typing import Dict, List, Optional

from . import config, devig
from .httputil import get_json, get_json_with_headers, read_cache, write_cache
from .logging_setup import get_logger
from .mapping import resolve_team

log = get_logger("jumpbot.odds")

_BASE = "https://api.the-odds-api.com/v4"
_ODDS_CACHE = "oddsapi_wc.json"
_OUTRIGHT_CACHE = "oddsapi_wc_winner.json"
_BUDGET_CACHE = "oddsapi_budget.json"

# Cache TTLs (minutes). Background refresh must stay well under STALE_ODDS_MIN
# (12h) so the board never ages into the Elo fallback — one fetch refreshes every
# match at once, so a proactive ~8h refresh is cheap and keeps the whole board sharp.
_TTL_BACKGROUND = 3 * 60
_TTL_NEAR_KO = 15
_TTL_STALE_OK = 36 * 60  # serve cache this old rather than nothing


def _load_budget() -> dict:
    b = read_cache(_BUDGET_CACHE, 10 ** 9) or {}
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if b.get("day") != today:
        b = {"day": today, "spent": 0.0, "remaining": float(b.get("remaining", 500))}
    return b


def _save_budget(b: dict) -> None:
    write_cache(_BUDGET_CACHE, b)


def _fetch_cost() -> int:
    return 2 * len(config.ODDS_API_REGIONS.split(","))  # markets(h2h,totals) x regions


def _daily_cap(b: dict) -> float:
    try:
        end = date.fromisoformat(config.ODDS_WC_END)
    except ValueError:
        end = date(2026, 7, 19)
    days_left = max(1, (end - datetime.now(timezone.utc).date()).days)
    return max(60.0, b.get("remaining", 500) / days_left)


@dataclass
class MatchOdds:
    home_code: Optional[str]
    away_code: Optional[str]
    p_home: float
    p_draw: float
    p_away: float
    total_line: Optional[float] = None
    p_over: Optional[float] = None
    p_btts: Optional[float] = None
    n_books: int = 0
    fetched_ts: float = 0.0
    commence_ts: Optional[float] = None
    source: str = "the-odds-api"

    def age_min(self) -> float:
        return (time.time() - self.fetched_ts) / 60.0


def _iso_to_epoch(s: str) -> Optional[float]:
    from dateutil import parser as _p
    try:
        return _p.isoparse(s).timestamp()
    except (ValueError, TypeError):
        return None


def fetch_world_cup_odds(force: bool = False, ttl_min: float = 20.0,
                         hard: bool = False) -> Optional[dict]:
    """Fetch (and cache) the odds payload (paid tier, lightly rationed).

    With a paid plan the credit budget is no longer the binding constraint, so we
    refresh on TTL (background ~3h, near-kickoff ~15m). Near-kickoff (force) fetches
    bypass the daily-cap gate entirely to capture closing-line value; background
    fetches keep a generous daily floor as a runaway safety only. ``hard`` bypasses
    the cache entirely (closing window) so the final pre-lock submit is on the
    freshest possible line.
    """
    if not config.ODDS_API_KEY:
        return None
    ttl = 0.0 if hard else (_TTL_NEAR_KO if force else _TTL_BACKGROUND) * 60
    cached = read_cache(_ODDS_CACHE, ttl)
    if cached is not None:
        return cached

    b = _load_budget()
    cost = _fetch_cost()
    cap = _daily_cap(b)
    # One fetch refreshes every match at once, so any fetch within the daily cap
    # keeps the whole board sharp — don't sub-throttle background vs force, or the
    # essential staleness refresh gets blocked and the board decays to Elo.
    allowance = cap
    if not force and b["spent"] + cost > allowance:
        log.info("odds: budget gate (spent %.0f + %d > cap %.0f, force=%s) -> serving cache",
                 b["spent"], cost, allowance, force)
        return read_cache(_ODDS_CACHE, _TTL_STALE_OK * 60)

    try:
        data, headers = get_json_with_headers(
            f"{_BASE}/sports/{config.ODDS_API_SPORT}/odds",
            params={"apiKey": config.ODDS_API_KEY, "regions": config.ODDS_API_REGIONS,
                    "markets": "h2h,totals", "oddsFormat": "decimal", "dateFormat": "iso"},
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("odds fetch failed: %s", exc)
        return read_cache(_ODDS_CACHE, _TTL_STALE_OK * 60)

    b["spent"] += cost
    rem = headers.get("x-requests-remaining")
    if rem is not None:
        try:
            b["remaining"] = float(rem)
        except ValueError:
            pass
    _save_budget(b)
    payload = {"fetched_ts": time.time(), "events": data}
    write_cache(_ODDS_CACHE, payload)
    log.info("fetched odds: %d events, cost %d, spent today %.0f/%.0f, %s remaining",
             len(data) if isinstance(data, list) else 0, cost, b["spent"], cap,
             b.get("remaining", "?"))
    return payload


def _devig_h2h(outcomes: List[dict], home_code, away_code):
    """Return (p_home, p_draw, p_away) from one book's h2h outcomes, or None."""
    by_code = {}
    draw_price = None
    for o in outcomes:
        name = o.get("name", "")
        price = o.get("price")
        if not price or price <= 1.0:
            return None
        if name.lower() == "draw":
            draw_price = price
            continue
        code = resolve_team(name)
        if code:
            by_code[code] = price
    if draw_price is None or home_code not in by_code or away_code not in by_code:
        return None
    res = devig.shin_devig([by_code[home_code], draw_price, by_code[away_code]])
    return res.probs  # [home, draw, away]


def _devig_totals(markets_outcomes: Dict[float, List[dict]]):
    """Pick the most-quoted line and de-vig over/under -> (line, p_over)."""
    best_line, best_n = None, -1
    for line, books in markets_outcomes.items():
        n = len(books)
        # prefer lines with more books, tie-break toward 2.5
        score = n - abs(line - 2.5) * 0.1
        if score > best_n:
            best_n, best_line = score, line
    if best_line is None:
        return None, None
    overs, unders = [], []
    for pair in markets_outcomes[best_line]:
        if pair.get("over") and pair.get("under"):
            overs.append(pair["over"])
            unders.append(pair["under"])
    if not overs:
        return best_line, None
    p_overs = []
    for ov, un in zip(overs, unders):
        try:
            p_overs.append(devig.power_devig([ov, un]).probs[0])
        except Exception:  # noqa: BLE001
            continue
    return best_line, (median(p_overs) if p_overs else None)


def match_odds_for(home_code: str, away_code: str,
                   payload: Optional[dict] = None) -> Optional[MatchOdds]:
    """De-vigged fair probabilities for one fixture, consensus across books."""
    payload = payload or fetch_world_cup_odds()
    if not payload:
        return None
    events = payload.get("events", [])
    fetched_ts = payload.get("fetched_ts", time.time())
    for ev in events:
        h = resolve_team(ev.get("home_team", ""))
        a = resolve_team(ev.get("away_team", ""))
        if {h, a} != {home_code, away_code}:
            continue
        # Orient to the SportsPredict home/away (match-name order).
        swap = (h == away_code)
        h1x2: List[List[float]] = []
        totals_by_line: Dict[float, List[dict]] = defaultdict(list)
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") == "h2h":
                    probs = _devig_h2h(mk.get("outcomes", []), h, a)
                    if probs:
                        h1x2.append(probs)
                elif mk.get("key") == "totals":
                    pair = {}
                    line = None
                    for o in mk.get("outcomes", []):
                        line = o.get("point", line)
                        if o.get("name", "").lower() == "over":
                            pair["over"] = o.get("price")
                        elif o.get("name", "").lower() == "under":
                            pair["under"] = o.get("price")
                    if line is not None and "over" in pair and "under" in pair:
                        totals_by_line[float(line)].append(pair)
        if not h1x2:
            return None
        p_home = median(p[0] for p in h1x2)
        p_draw = median(p[1] for p in h1x2)
        p_away = median(p[2] for p in h1x2)
        s = p_home + p_draw + p_away
        p_home, p_draw, p_away = p_home / s, p_draw / s, p_away / s
        line, p_over = _devig_totals(totals_by_line)
        mo = MatchOdds(
            home_code=home_code, away_code=away_code,
            p_home=(p_away if swap else p_home),
            p_draw=p_draw,
            p_away=(p_home if swap else p_away),
            total_line=line, p_over=p_over,
            n_books=len(h1x2), fetched_ts=fetched_ts,
            commence_ts=_iso_to_epoch(ev.get("commence_time", "")),
        )
        return mo
    return None


def fetch_outrights(force: bool = False, ttl_hours: float = 6.0) -> Optional[Dict[str, float]]:
    """De-vigged tournament-winner probabilities, code -> prob."""
    if not config.ODDS_API_KEY:
        return None
    cached = None if force else read_cache(_OUTRIGHT_CACHE, ttl_hours * 3600)
    if cached is None:
        try:
            data = get_json(
                f"{_BASE}/sports/soccer_fifa_world_cup_winner/odds",
                params={"apiKey": config.ODDS_API_KEY, "regions": config.ODDS_API_REGIONS,
                        "markets": "outrights", "oddsFormat": "decimal"},
            )
            cached = {"fetched_ts": time.time(), "events": data}
            write_cache(_OUTRIGHT_CACHE, cached)
        except Exception as exc:  # noqa: BLE001
            log.warning("outright fetch failed: %s", exc)
            return None
    # Aggregate outcomes across books; de-vig the consensus field.
    prices: Dict[str, List[float]] = defaultdict(list)
    for ev in cached.get("events", []):
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") != "outrights":
                    continue
                for o in mk.get("outcomes", []):
                    code = resolve_team(o.get("name", ""))
                    if code and o.get("price", 0) > 1.0:
                        prices[code].append(o["price"])
    if not prices:
        return None
    cons = {code: median(p) for code, p in prices.items()}
    res = devig.devig(list(cons.values()), method="power")
    return {code: prob for code, prob in zip(cons.keys(), res.probs)}


def remaining_quota() -> Optional[str]:
    """Best-effort read of the cached quota header (set during fetches)."""
    return None  # quota is surfaced via response headers; logged at fetch time


# --- player props (per-event endpoint; market-tier scorer/assist) -----------
# The bulk /odds endpoint only carries h2h/totals; player props live on the
# per-event endpoint and only at us/uk/eu books. Pinnacle quotes a clean Yes/No
# anytime-scorer line we anchor on (sharpest book in the world); matching it beats
# any WC-only model, which can't see career priors (e.g. that a hot 4-in-2 streak
# is unsustainable). Fetched only inside the pre-KO window and cached, so cheap.
_PROP_MARKETS = "player_goal_scorer_anytime,player_assists,btts,player_shots_on_target"
# Shots-on-target books quote Over-only (no Under to de-vig against), so we haircut
# the cross-book consensus implied prob by this factor to strip the (one-sided) margin.
_SOT_VIG = 1.06


def _prop_cost() -> int:
    return len(_PROP_MARKETS.split(",")) * len(config.ODDS_PROP_REGIONS.split(","))


def _event_id_for(home_code: Optional[str], away_code: Optional[str]) -> Optional[str]:
    """Map a (home, away) code pair to the the-odds-api event id via the cached
    bulk payload (already fetched for h2h/totals)."""
    if not (home_code and away_code):
        return None
    cached = read_cache(_ODDS_CACHE, _TTL_STALE_OK * 60)
    for e in (cached or {}).get("events", []):
        if resolve_team(e.get("home_team", "")) == home_code and \
           resolve_team(e.get("away_team", "")) == away_code:
            return e.get("id")
    return None


def _devig_scorer(bookmakers: List[dict]) -> Dict[str, tuple]:
    """De-vig anytime-scorer into {norm_name: (p_yes, n_2way_books)}.

    Prefer books quoting both Yes and No (clean 2-way; Pinnacle is the reference);
    consensus = median across them. For Yes-only books, haircut the implied Yes by
    the average per-player booksum measured from the two-sided books.
    """
    from .stats import _norm
    yes: Dict[str, Dict[str, float]] = defaultdict(dict)
    no: Dict[str, Dict[str, float]] = defaultdict(dict)
    for b in bookmakers:
        bk = b.get("key", "")
        for m in b.get("markets", []):
            if m.get("key") != "player_goal_scorer_anytime":
                continue
            for o in m.get("outcomes", []):
                nm = _norm(o.get("description") or "")
                price = o.get("price")
                if not nm or not price or price <= 1.0:
                    continue
                side = o.get("name", "").lower()
                if side == "yes":
                    yes[nm][bk] = price
                elif side == "no":
                    no[nm][bk] = price
    margins = [1.0 / yp + 1.0 / no[nm][bk]
               for nm in yes for bk, yp in yes[nm].items() if bk in no.get(nm, {})]
    avg_margin = (sum(margins) / len(margins)) if margins else 1.10
    out: Dict[str, tuple] = {}
    for nm, ybooks in yes.items():
        clean = [devig.two_way_prob(yp, no[nm][bk])
                 for bk, yp in ybooks.items() if bk in no.get(nm, {})]
        if clean:
            out[nm] = (float(median(clean)), len(clean))
        else:  # yes-only: consensus implied, proportional-devig by the measured booksum
            imp = median([1.0 / y for y in ybooks.values()])
            out[nm] = (float(min(0.97, imp / avg_margin)), 0)
    return out


def _devig_assist(bookmakers: List[dict]) -> Dict[str, float]:
    """De-vig player_assists into {norm_name: p_yes} (thin market; consensus implied)."""
    from .stats import _norm
    imp: Dict[str, List[float]] = defaultdict(list)
    for b in bookmakers:
        for m in b.get("markets", []):
            if m.get("key") != "player_assists":
                continue
            for o in m.get("outcomes", []):
                nm = _norm(o.get("description") or "")
                price = o.get("price")
                if nm and price and price > 1.0 and o.get("name", "").lower() in ("yes", "over"):
                    imp[nm].append(1.0 / price)
    return {nm: float(min(0.7, median(v) / 1.12)) for nm, v in imp.items()}


def _devig_btts(bookmakers: List[dict]) -> Optional[tuple]:
    """Match-level P(both teams score) from the btts Yes/No line, as (p, n_books)."""
    pairs = []
    for b in bookmakers:
        for m in b.get("markets", []):
            if m.get("key") != "btts":
                continue
            o = {x.get("name"): x.get("price") for x in m.get("outcomes", [])}
            y, n = o.get("Yes"), o.get("No")
            if y and n and y > 1.0 and n > 1.0:
                pairs.append(devig.two_way_prob(y, n))
    if not pairs:
        return None
    return (float(median(pairs)), len(pairs))


def _devig_sot(bookmakers: List[dict]) -> Dict[str, Dict[int, tuple]]:
    """Per-player shots-on-target into {norm_name: {n: (p_ge_n, n_books)}}.

    Books quote Over-only at points 0.5/1.5/2.5 (Over (n-0.5) == >=n SoT); we take
    the cross-book consensus implied prob and haircut by _SOT_VIG (one-sided margin).
    """
    from .stats import _norm
    over: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    for b in bookmakers:
        for m in b.get("markets", []):
            if m.get("key") != "player_shots_on_target":
                continue
            for o in m.get("outcomes", []):
                if o.get("name", "").lower() != "over":
                    continue
                nm = _norm(o.get("description") or "")
                pt, price = o.get("point"), o.get("price")
                if nm and pt is not None and price and price > 1.0:
                    over[nm][pt].append(price)
    out: Dict[str, Dict[int, tuple]] = {}
    for nm, pts in over.items():
        thr: Dict[str, tuple] = {}
        for pt, prices in pts.items():
            n = int(round(pt + 0.5))  # Over 0.5 -> >=1, Over 1.5 -> >=2
            imp = median([1.0 / p for p in prices])
            thr[str(n)] = (float(min(0.97, imp / _SOT_VIG)), len(prices))  # str: JSON-cache safe
        out[nm] = thr
    return out


def fetch_event_props(event_id: str, force: bool = False) -> Optional[dict]:
    """Fetch + de-vig per-event player props (scorer, assist). Cached PROP_TTL_MIN,
    budget-gated. Returns {fetched_ts, scorer:{norm:(p,nbooks)}, assist:{norm:p}}."""
    if not config.ODDS_API_KEY or not event_id:
        return None
    cache_key = f"oddsapi_props_{event_id}.json"
    cached = read_cache(cache_key, config.PROP_TTL_MIN * 60)
    if cached is not None and not force:
        return cached
    b = _load_budget()
    cost = _prop_cost()
    if not force and b["spent"] + cost > _daily_cap(b):
        return read_cache(cache_key, _TTL_STALE_OK * 60)
    try:
        data, headers = get_json_with_headers(
            f"{_BASE}/sports/{config.ODDS_API_SPORT}/events/{event_id}/odds",
            params={"apiKey": config.ODDS_API_KEY, "regions": config.ODDS_PROP_REGIONS,
                    "markets": _PROP_MARKETS, "oddsFormat": "decimal"})
    except Exception as exc:  # noqa: BLE001
        log.warning("prop fetch failed (%s): %s", event_id, exc)
        return read_cache(cache_key, _TTL_STALE_OK * 60)
    b["spent"] += cost
    rem = headers.get("x-requests-remaining")
    if rem is not None:
        try:
            b["remaining"] = float(rem)
        except ValueError:
            pass
    _save_budget(b)
    books = (data or {}).get("bookmakers", [])
    payload = {"fetched_ts": time.time(),
               "scorer": _devig_scorer(books), "assist": _devig_assist(books),
               "btts": _devig_btts(books), "sot": _devig_sot(books)}
    write_cache(cache_key, payload)
    log.info("fetched props: event %s, %d scorer / %d sot players, btts=%s, %d books, cost %d",
             event_id, len(payload["scorer"]), len(payload["sot"]),
             payload["btts"][0] if payload["btts"] else None, len(books), cost)
    return payload


def _match_player(table: dict, player: str):
    """Resolve a contest player name against a de-vigged prop table (norm + last-name)."""
    from .stats import _norm, _last
    nm = _norm(player)
    if nm in table:
        return table[nm]
    ln = _last(player)
    for k, v in table.items():
        if ln and k.split()[-1:] == [ln]:
            return v
    return None


def scorer_market(home_code, away_code, player) -> Optional[tuple]:
    """Market P(anytime scorer) for a player as (p, n_2way_books), or None."""
    eid = _event_id_for(home_code, away_code)
    if not eid:
        return None
    props = fetch_event_props(eid)
    if not props:
        return None
    return _match_player(props.get("scorer", {}), player)


def assist_market(home_code, away_code, player) -> Optional[float]:
    """Market P(assist) for a player, or None."""
    eid = _event_id_for(home_code, away_code)
    if not eid:
        return None
    props = fetch_event_props(eid)
    if not props:
        return None
    return _match_player(props.get("assist", {}), player)


def btts_market(home_code, away_code) -> Optional[tuple]:
    """Market P(both teams score) as (p, n_books), or None."""
    eid = _event_id_for(home_code, away_code)
    if not eid:
        return None
    props = fetch_event_props(eid)
    return props.get("btts") if props else None


def sot_market(home_code, away_code, player, n: int = 1) -> Optional[tuple]:
    """Market P(player has >=n shots on target) as (p, n_books), or None.

    Combines the the-odds-api consensus with Bet365 (BetsAPI) as an extra book,
    book-count weighted. This firms up the thin 1-book lines — esp. 2+ SoT, where
    a single Odds-API book and Bet365 often disagree — into a 2-book consensus that
    clears the engine's >=2-book trust gate. Bet365 is fully guarded: any failure
    leaves the Odds-API-only result unchanged.
    """
    nn = int(n or 1)
    oa = None
    eid = _event_id_for(home_code, away_code)
    if eid:
        props = fetch_event_props(eid)
        if props:
            tbl = _match_player(props.get("sot", {}), player)
            oa = tbl.get(str(nn)) if tbl else None
    from . import bet365  # lazy import: avoids any import-time coupling
    b365 = bet365.sot_prob(home_code, away_code, player, nn)
    if oa is not None and b365 is not None:
        p_oa, n_oa = oa
        return ((p_oa * n_oa + b365) / (n_oa + 1), n_oa + 1)
    if oa is not None:
        return oa
    if b365 is not None:
        return (b365, 1)  # Bet365-only: 1 book -> engine still blends with the model
    return None
