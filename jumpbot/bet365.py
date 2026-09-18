"""BetsAPI Bet365 prematch markets — a 2nd book / market anchor for our model-only
seams: player shots-on-target (2nd book vs the-odds-api) and total corners (a sharp
band line where the cup otherwise has no public line).

Read-only and heavily guarded: any failure (no token, network, rate-limit, missing
market, unresolved fixture/player) returns None/{} so callers fall back to the
existing path — Bet365 can only ADD signal, never break pricing.

Flow: /v1/bet365/upcoming -> map each "World Cup 2026" fixture to its Bet365 FI
(cached 6h; fixtures are stable); /v3/bet365/prematch?FI -> cache the raw object 30m
and parse markets on demand.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from scipy.stats import poisson

from . import config, devig
from .httputil import get_json, read_cache, write_cache
from .logging_setup import get_logger
from .mapping import resolve_team
from .stats import _norm

log = get_logger("jumpbot.bet365")

_BASE = "https://api.b365api.com"
_FI_CACHE = "bet365_fi_map.json"   # {home-away code pair: FI}; fixtures stable -> 6h TTL
_FI_TTL = 6 * 3600
_PROP_TTL = 30 * 60                # per-fixture prematch tree cached 30m
_STALE_OK = 36 * 3600             # serve a stale cache rather than nothing
_SOT_VIG = 1.06                    # match odds.py's one-sided haircut so books are comparable


def enabled() -> bool:
    return bool(config.BETSAPI_TOKEN)


def _get(path: str, params: dict) -> Optional[dict]:
    if not enabled():
        return None
    p = dict(params)
    p["token"] = config.BETSAPI_TOKEN
    try:
        d = get_json(f"{_BASE}/{path}", params=p)
    except Exception as exc:  # noqa: BLE001 - never let Bet365 break pricing
        log.warning("bet365 %s failed: %s", path, exc)
        return None
    if not isinstance(d, dict) or d.get("success") != 1:
        return None
    return d


def _fi_map(force: bool = False) -> Dict[str, int]:
    """{f'{home}-{away}': FI} for every World Cup fixture in bet365/upcoming."""
    cached = read_cache(_FI_CACHE, _FI_TTL)
    if cached is not None and not force:
        return cached
    out: Dict[str, int] = {}
    for page in range(1, 9):
        d = _get("v1/bet365/upcoming", {"sport_id": 1, "page": page})
        if not d:
            break
        res = d.get("results", []) or []
        for ev in res:
            lg = ((ev.get("league") or {}).get("name") or "").lower()
            if "world cup" not in lg:  # skip esoccer / club noise
                continue
            h = resolve_team((ev.get("home") or {}).get("name", ""))
            a = resolve_team((ev.get("away") or {}).get("name", ""))
            fi = ev.get("id")
            if h and a and fi:
                try:
                    out[f"{h}-{a}"] = int(fi)
                except (TypeError, ValueError):
                    pass
        if not res:
            break
    if out:
        write_cache(_FI_CACHE, out)
        return out
    return cached or {}


def fi_for(home_code, away_code) -> Optional[int]:
    if not (home_code and away_code):
        return None
    m = _fi_map()
    return m.get(f"{home_code}-{away_code}") or m.get(f"{away_code}-{home_code}")


def _prematch_obj(fi: int, force: bool = False) -> Optional[dict]:
    """Fetch + cache (30m) the raw Bet365 prematch results[0] object."""
    ck = f"bet365_pm_{fi}.json"
    cached = read_cache(ck, _PROP_TTL)
    if cached is not None and not force:
        return cached
    d = _get("v3/bet365/prematch", {"FI": fi})
    if not d:
        return read_cache(ck, _STALE_OK)
    try:
        obj = d["results"][0]
    except (KeyError, IndexError, TypeError):
        return read_cache(ck, _STALE_OK)
    if isinstance(obj, dict):
        write_cache(ck, obj)
        return obj
    return read_cache(ck, _STALE_OK)


def _group_odds(obj: dict, group: str, sub: str) -> List[dict]:
    try:
        return ((obj.get(group) or {}).get("sp") or {}).get(sub, {}).get("odds") or []
    except (AttributeError, TypeError):
        return []


# --- player shots on target ------------------------------------------------
def _sot_ladder(fi: int) -> Dict[str, Dict[str, float]]:
    """{norm_name: {str(n): p_over_devigged}} from player_shots_on_target."""
    obj = _prematch_obj(fi)
    if not obj:
        return {}
    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    for o in _group_odds(obj, "shots", "player_shots_on_target"):
        m = re.match(r"(\d+)\+", o.get("header", "") or "")
        nm = _norm(o.get("name", "") or "")
        od = o.get("odds")
        if not (m and nm and od):
            continue
        try:
            imp = 1.0 / float(od)
        except (TypeError, ValueError):
            continue
        out[nm][str(int(m.group(1)))] = min(0.97, imp / _SOT_VIG)  # str key: JSON-cache safe
    return {k: v for k, v in out.items()}


def sot_prob(home_code, away_code, player, n: int = 1) -> Optional[float]:
    """Bet365 de-vigged P(player has >= n SoT), or None if unavailable."""
    fi = fi_for(home_code, away_code)
    if not fi:
        return None
    ladder = _sot_ladder(fi)
    if not ladder:
        return None
    key = _norm(player)
    rec = ladder.get(key)
    if rec is None:  # last-name fallback, mirrors stats._player_rec
        parts = key.split()
        ln = parts[-1] if parts else ""
        for k, v in ladder.items():
            kp = k.split()
            if ln and kp and kp[-1] == ln:
                rec = v
                break
    if not rec:
        return None
    return rec.get(str(int(n or 1)))


# --- total corners (banded distribution) -----------------------------------
def _corner_bands(fi: int) -> List[Tuple[int, float]]:
    """[(lower_bound, implied_prob)] from the total_corners band market."""
    obj = _prematch_obj(fi)
    if not obj:
        return []
    bands: List[Tuple[int, float]] = []
    for o in _group_odds(obj, "corners", "total_corners"):
        nm = (o.get("name") or "").strip().lower()
        od = o.get("odds")
        try:
            imp = 1.0 / float(od)
        except (TypeError, ValueError):
            continue
        mu = re.match(r"under\s+(\d+)", nm)
        mo = re.match(r"over\s+(\d+)", nm)
        mb = re.match(r"(\d+)\s*-\s*(\d+)", nm)
        if mu:
            lo = 0
        elif mo:
            lo = int(mo.group(1)) + 1
        elif mb:
            lo = int(mb.group(1))
        else:
            continue
        bands.append((lo, imp))
    return bands


def corners_total_prob(home_code, away_code, thr: int) -> Optional[float]:
    """De-vigged P(total corners >= thr) from Bet365's band market, or None.

    Only returns when thr aligns with a band lower-bound (a clean cut); otherwise
    None so the caller keeps the model (no mid-band interpolation guesswork)."""
    fi = fi_for(home_code, away_code)
    if not fi:
        return None
    bands = _corner_bands(fi)
    if not bands:
        return None
    total = sum(b[1] for b in bands)
    los = {b[0] for b in bands}
    if total <= 0 or int(thr) not in los:
        return None
    p = sum(imp for lo, imp in bands if lo >= int(thr)) / total
    return min(0.97, max(0.03, p))


def corners_market_lambda(home_code, away_code) -> Optional[float]:
    """Market-implied TOTAL-corners lambda from Bet365's band market — the median of the
    per-edge Poisson inversions over the interior bands (skips the extreme tails where the
    inversion is unstable). Lets team_corners price off the sharp total instead of raw
    for-rates. None when no posted band market."""
    fi = fi_for(home_code, away_code)
    if not fi:
        return None
    bands = _corner_bands(fi)
    total = sum(b[1] for b in bands)
    if total <= 0:
        return None
    lams = []
    for lo in sorted({b[0] for b in bands if b[0] > 0}):
        p = sum(imp for b_lo, imp in bands if b_lo >= lo) / total
        if 0.12 <= p <= 0.88:   # interior bands only — tail inversions are noisy
            lams.append(_invert_poisson_tail(lo, p))
    if not lams:
        return None
    lams.sort()
    return lams[len(lams) // 2]


# --- total cards (anchor the Poisson card-rate to the market line) ----------
def _invert_poisson_tail(thr: int, p_ge: float, lo: float = 0.05, hi: float = 12.0) -> float:
    """lambda such that P(X>=thr)=p_ge for X~Poisson(lambda) (sf is monotone in lambda)."""
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if float(poisson.sf(thr - 1, mid)) < p_ge:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _card_line(fi: int) -> Optional[Tuple[float, int]]:
    """(de-vigged P(over), over_threshold) from number_of_cards_in_match, or None."""
    obj = _prematch_obj(fi)
    if not obj:
        return None
    over = under = None
    for o in _group_odds(obj, "cards_fouls", "number_of_cards_in_match"):
        txt = f"{o.get('header', '')} {o.get('name', '')}".strip().lower()
        m = re.search(r"(\d+(?:\.\d+)?)", txt)
        try:
            od = float(o.get("odds"))
        except (TypeError, ValueError):
            continue
        if not m or od <= 1.0:
            continue
        ln = float(m.group(1))
        if "over" in txt:
            over = (od, ln)
        elif "under" in txt:
            under = (od, ln)
    if not (over and under and over[1] == under[1]):
        return None
    p_over = devig.two_way_prob(over[0], under[0])  # P(> line) == P(>= ceil(line))
    return (float(p_over), int(over[1] + 0.5))


def cards_market_lambda(home_code, away_code) -> Optional[float]:
    """Market-implied Poisson card-rate from Bet365's total-cards line, or None.

    Anchors the *level* (the model under-prices KO-game card intensity) while the
    Poisson shape re-prices whatever threshold the cup asks for."""
    fi = fi_for(home_code, away_code)
    if not fi:
        return None
    cl = _card_line(fi)
    if not cl:
        return None
    p_over, line_thr = cl
    if not (0.02 < p_over < 0.98) or line_thr < 1:
        return None
    return _invert_poisson_tail(line_thr, p_over)


# --- player to score or assist (2nd book for player_involve) ----------------
_INV_VIG = 1.12  # one-sided "to score or assist" haircut (calibrated vs the-odds-api involve)


def _score_or_assist(fi: int) -> Dict[str, float]:
    """{norm_name: p_devigged} from the 'Score or Assist' side of the market."""
    obj = _prematch_obj(fi)
    if not obj:
        return {}
    out: Dict[str, float] = {}
    for o in _group_odds(obj, "scorers", "player_to_score_or_assist"):
        if "or assist" not in (o.get("header", "") or "").lower():
            continue
        nm = _norm(o.get("name", "") or "")
        try:
            imp = 1.0 / float(o.get("odds"))
        except (TypeError, ValueError):
            continue
        if nm:
            out[nm] = min(0.95, imp / _INV_VIG)
    return out


def involve_prob(home_code, away_code, player) -> Optional[float]:
    """Bet365 de-vigged P(player scores or assists), or None."""
    fi = fi_for(home_code, away_code)
    if not fi:
        return None
    tbl = _score_or_assist(fi)
    if not tbl:
        return None
    key = _norm(player)
    p = tbl.get(key)
    if p is None:
        parts = key.split()
        ln = parts[-1] if parts else ""
        for k, v in tbl.items():
            kp = k.split()
            if ln and kp and kp[-1] == ln:
                p = v
                break
    return p


# --- penalty awarded in match (market anchor for pen_only / pen_or_red) -----
def penalty_prob(home_code, away_code) -> Optional[float]:
    """De-vigged P(a penalty is awarded in the match), or None if unposted."""
    fi = fi_for(home_code, away_code)
    if not fi:
        return None
    obj = _prematch_obj(fi)
    if not obj:
        return None
    yes = no = None
    for o in _group_odds(obj, "other", "a_penalty_in_the_match"):
        txt = f"{o.get('header', '')} {o.get('name', '')}".strip().lower()
        try:
            od = float(o.get("odds"))
        except (TypeError, ValueError):
            continue
        if od <= 1.0:
            continue
        if "yes" in txt:
            yes = od
        elif "no" in txt:
            no = od
    if yes and no:
        return float(devig.two_way_prob(yes, no))
    return None


def both_teams_card_prob(home_code, away_code) -> Optional[float]:
    """De-vigged P(both teams receive >=1 card) from the 'a Card' Yes/No pair."""
    fi = fi_for(home_code, away_code)
    if not fi:
        return None
    obj = _prematch_obj(fi)
    if not obj:
        return None
    yes = no = None
    for o in _group_odds(obj, "cards_fouls", "both_teams_to_receive_cards"):
        if (o.get("name", "") or "").strip().lower() != "a card":
            continue  # the ">=1 card each" line — not '2+ Cards' or 'a Red Card'
        try:
            od = float(o.get("odds"))
        except (TypeError, ValueError):
            continue
        if od <= 1.0:
            continue
        hdr = (o.get("header", "") or "").strip().lower()
        if hdr == "yes":
            yes = od
        elif hdr == "no":
            no = od
    if yes and no:
        return float(devig.two_way_prob(yes, no))
    return None
