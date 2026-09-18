"""Pre-lock crowd-consensus guardrail (the game-theory layer).

The web match-stats endpoint exposes the live field average (prediction_average)
per market BEFORE lock — data the bot API hides. Measured on 174 settled markets
(2026-07-01): when we diverge >15pp from the field on a kind with no measured
positive RBP, the field is nearly always closer to truth ("thin-data confident
fades LOSE"). Pulling such prices halfway to the field would have added +107 RBP,
positive on EVERY affected kind (corners +23, offsides +22, fouls/cards compares
+27, hydration +27, player_scorer +8), and +86 even with no kind exemptions —
robust, not hindsight-fit. Kinds with measured positive total RBP are exempt:
their divergence IS the edge (player_sot fades, card fades, KO-upset line-sitting).

Fail-safe by design: token expiry / HTTP error / unknown match id -> no-op.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

from . import config, state
from .logging_setup import get_logger

log = get_logger("jumpbot.crowd_guard")

_TTL_OK = 600.0    # field averages move slowly; refresh a match every 10 min
_TTL_ERR = 180.0   # after a fetch failure, back off for 3 min
_EXEMPT_TTL = 3600.0

_match_cache: Dict[str, Tuple[float, Dict[str, float]]] = {}
_exempt_cache: Tuple[float, frozenset] = (0.0, frozenset())
_consec_failures = 0  # >=3 escalates to WARNING: guard + risk cap are blind

# Kinds whose market-tier price is a DIRECTLY-QUOTED book line (the book prices this
# exact question). Never pull those toward a hedging field — the line is truth by
# tier-1 doctrine (first live miss: COL/GHA team_advance line .84 pulled to .77 vs a
# field at .69 that was just hedging to 0.5). Derived market-tier prices (corner-λ →
# threshold transforms, split shares) stay guardable — they were the −42 corners leak.
_DIRECT_LINE_KINDS = frozenset({
    "win_match", "team_advance", "total_over", "total_under", "btts",
    "player_scorer", "player_involve", "win_margin", "match_tie",
    "extra_time",  # == the de-vigged draw quote; shootout stays guardable
                   # (its x0.5 ET factor is a model assumption, not a line)
})

# Kinds where the FIELD is the best price we have (top-8 study 2026-07-01: the #1-8
# ranked players sit AT field on these; realized rates agree — pens/reds hit 3/7 in
# KOs vs our .21 fades, sub scorers hit 2/3 vs our .20 fades). Our model has proven
# NEGATIVE RBP on both. Sit 75% of the way to the field; keep a 25% model voice for
# when the field is thin. Counterfactual on settled: +58.2 RBP.
# 2026-07-02: + card-OCCURRENCE kinds (any_card_half, both_teams_card): n=3 settled,
# all negative (-27 total; USA H1-card .71 vs field .61 came NO), top-8 sat AT field
# and beat us +14/mkt on the class — ref-dependent occurrence noise our lambda-split
# can't model. Card COUNT kinds (total_cards/team_cards) are NOT here — those fades
# are our #1 earner.
_FIELD_SIT_KINDS = frozenset({"pen_or_red", "sub_scorer", "any_card_half", "both_teams_card"})
_FIELD_SIT_PULL = 0.75


def _crowd_stats():
    try:
        import crowd_stats  # repo-root module; on sys.path when run from repo root
        return crowd_stats
    except Exception as exc:  # noqa: BLE001
        log.info("crowd_stats unavailable (%s) -> guard inactive", exc)
        return None


def _web_id_for(match_name: str) -> Optional[str]:
    """Map 'BEL vs SEN' to the web match id via the crowd_stats tag list."""
    cs = _crowd_stats()
    if cs is None or " vs " not in (match_name or ""):
        return None
    h, a = [t.strip() for t in match_name.split(" vs ")[:2]]
    for tag, mid in cs.MATCHES:
        parts = tag.split("_")
        if len(parts) >= 2 and parts[0] == h and parts[1] == a:
            return mid
    return None


def _crowd_for_match(match_name: str, match_id: Optional[str] = None) -> Dict[str, float]:
    """{market_id: field average 0-1} for one match, cached. {} on any failure.

    match_id (from the bot API snapshot — web id == bot id) makes this fully
    self-discovering; the crowd_stats tag list is only a fallback."""
    now = time.time()
    key = match_id or match_name
    hit = _match_cache.get(key)
    if hit and now - hit[0] < (_TTL_OK if hit[1] else _TTL_ERR):
        return hit[1]
    out: Dict[str, float] = {}
    wid = match_id or _web_id_for(match_name)
    if wid:
        cs = _crowd_stats()
        try:
            d = cs.match_stats(wid)
            for m in d.get("markets") or []:
                mid, p = m.get("id"), m.get("prediction_average")
                if mid is None or p is None:
                    continue
                p = float(p)
                out[mid] = p / 100.0 if p > 1.0 else p
        except Exception as exc:  # noqa: BLE001 - fragile web token; never block pricing
            global _consec_failures
            _consec_failures += 1
            if _consec_failures >= 3:
                log.warning("crowd fetch failed %dx in a row (%s) — guard AND risk cap "
                            "are BLIND; refresh data/sp_web_token.txt + sp_web_cookie.txt",
                            _consec_failures, exc)
            else:
                log.info("crowd fetch failed for %s: %s (guard no-op)", match_name, exc)
    if out:
        globals()["_consec_failures"] = 0
    _match_cache[key] = (now, out)
    return out


def _exempt_kinds() -> frozenset:
    """Kinds with measured positive total RBP over a real sample — their divergence
    is the edge. The n>=4 floor (matches NOVELTY_ANCHOR_MIN_N) exists because one
    lucky game is not proof: total_shots was exempt on n=1 (+23 from ENG/COD) and
    rode an unguarded 0.84-vs-field-0.57 into -70 RBP (USA/BIH 2026-07-01). LOO
    counterfactual on 200 settled markets: the floor is worth +51 RBP."""
    global _exempt_cache
    ts, kinds = _exempt_cache
    now = time.time()
    if now - ts < _EXEMPT_TTL:
        return kinds
    try:
        with state._conn() as c:  # noqa: SLF001 - same package
            rows = c.execute("SELECT kind, SUM(market_rbp) AS s, COUNT(*) AS n "
                             "FROM crowd_results GROUP BY kind").fetchall()
        kinds = frozenset(r["kind"] for r in rows if (r["s"] or 0) > 0 and r["n"] >= 4)
        _exempt_cache = (now, kinds)
    except Exception as exc:  # noqa: BLE001 - keep last good set, retry in 5 min
        log.info("exempt-kinds query failed: %s", exc)
        _exempt_cache = (now - _EXEMPT_TTL + 300.0, kinds)
    return kinds


def apply(kind: str, market_id: str, match_name: str, p: float,
          tier: Optional[str] = None, threshold: Optional[int] = None,
          match_id: Optional[str] = None,
          fav_volume: bool = False) -> Tuple[float, Optional[str]]:
    """Pull an over-divergent no-edge price partway to the field. (price, tag|None)."""
    if not config.CROWD_GUARD_ENABLED:
        return p, None
    if tier == "market" and kind in _DIRECT_LINE_KINDS and kind != "player_scorer":
        return p, None  # a directly-quoted sharp line is truth; don't chase the field
    exempt = kind in _exempt_kinds()
    # Carve-out: MODEL-ONLY (no book line) player-SoT TAIL thresholds (>=2). The thin
    # this-WC EB rate breaks there (Semenyo 270min/0-SoT -> 2+ priced 0.02 vs field
    # 0.33; Bellingham-class). The kind's winning fades all sit 8-14pp below the
    # field — inside the gap, untouched — so the edge survives; only breakage is caught.
    if kind == "player_sot" and (threshold or 1) >= 2 and tier != "market":
        exempt = False
    crowd = _crowd_for_match(match_name, match_id).get(market_id)
    if crowd is None:
        return p, None
    newp, tag = p, None
    # player_scorer is DIRECTIONAL (measured on 20 settled, 2026-07-05): every
    # above-field scorer price lost (Larin .45 vs field .25 -> -23.7, L.Diaz -20.0
    # — the "this-WC-hot rotation striker" trap), every below-field name fade won
    # (+10..+15 each). Cap at the field when above by >2pp (counterfactual +28.8
    # RBP / 20 markets); below-field prices, line-backed or not, ARE the edge —
    # leave them (scorer line prices skip the direct-line early-return for this).
    if kind == "player_scorer":
        if p > crowd + 0.02:
            newp, tag = crowd, f"scorercap(field{crowd:.2f})"
            log.info("crowd guard %s %s: %.2f -> %.2f (field %.2f)",
                     match_name, kind, p, newp, crowd)
            return newp, tag
        return p, None
    # Extreme-favorite volume markets (team SoT/corners, favorite-star SoT) run HOT:
    # the field beat our under-fades 6/9 (n=9 across ENG/GER/ESP games; +104.5 RBP
    # counterfactual). Pull 0.75 to field when we price BELOW it — overrides kind
    # exemptions AND line prices (favorite-star lines price minutes risk the game
    # script negates: Bellingham .36-line, Yamal .37-line both settled YES with the
    # field at .40/.51). Prices ABOVE field (Messi-class, career-confirmed volume)
    # are left alone.
    if fav_volume and p < crowd:
        newp = p + _FIELD_SIT_PULL * (crowd - p)
        tag = f"favvol(field{crowd:.2f})"
    elif kind in _FIELD_SIT_KINDS:
        newp = p + _FIELD_SIT_PULL * (crowd - p)
        tag = f"fieldsit(field{crowd:.2f})"
    elif not exempt and abs(p - crowd) > config.CROWD_GUARD_GAP:
        newp = p + config.CROWD_GUARD_PULL * (crowd - p)
        tag = f"guard(field{crowd:.2f})"
    # Rank-protection cap (OFF in go-for-the-win regime; RISK_CAP=1 re-enables):
    # bound the worst-case single-market RBP loss vs the field. Applies to EVERY
    # kind, proven edges included; direct book lines already returned above.
    # One-pass gmax is within ~1% of the fixed point — good enough for a risk bound.
    if config.RISK_CAP_ENABLED:
        term = max(newp + crowd, 2.0 - newp - crowd)
        if abs(newp - crowd) * term * 100.0 * config.RISK_WEIGHT > config.MAX_MKT_LOSS_RBP:
            gmax = config.MAX_MKT_LOSS_RBP / (config.RISK_WEIGHT * 100.0 * term)
            newp = crowd + (gmax if newp > crowd else -gmax)
            tag = f"riskcap(field{crowd:.2f})"
    if tag is None or abs(newp - p) < 1e-6:
        return p, None
    log.info("crowd %s %s %s: %.2f -> %.2f (field %.2f)",
             "cap" if tag.startswith("riskcap") else "guard", match_name, kind, p, newp, crowd)
    return newp, tag
