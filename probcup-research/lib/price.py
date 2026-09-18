"""Live-odds pricing support: parse the WC odds cache, recover goal lambdas from the
de-vigged market, and price the contest's prop markets with team-specific models.

Read-only: consumes the live bot's odds cache snapshot (data/cache/oddsapi_wc.json)
and never touches the contest API.

Key idea: for lined markets (1X2, totals) the de-vigged market is the benchmark; for
no-line props we map the market-implied goal lambdas to shots-on-target / offsides /
etc. so the rate is team- and opponent-specific instead of a flat base rate.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import poisson, nbinom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import devig

# contest abbreviation -> Odds-API full team name (from matched fixture ordering)
ABBR2FULL = {
    "POR": "Portugal", "UZB": "Uzbekistan", "ENG": "England", "GHA": "Ghana",
    "PAN": "Panama", "CRO": "Croatia", "COL": "Colombia", "COD": "DR Congo",
    "BIH": "Bosnia & Herzegovina", "QAT": "Qatar", "SUI": "Switzerland", "CAN": "Canada",
    "MAR": "Morocco", "Haiti": "Haiti", "SCO": "Scotland", "BRA": "Brazil",
    "RSA": "South Africa", "KOR": "South Korea", "CZE": "Czech Republic", "MEX": "Mexico",
    "Curacao": "Curaçao", "CIV": "Ivory Coast", "ECU": "Ecuador", "GER": "Germany",
    "JPN": "Japan", "SWE": "Sweden", "TUN": "Tunisia", "NED": "Netherlands",
    "PAR": "Paraguay", "AUS": "Australia", "TUR": "Turkey", "USA": "USA",
    "SEN": "Senegal", "IRQ": "Iraq", "NOR": "Norway", "FRA": "France",
    "URU": "Uruguay", "ESP": "Spain", "CPV": "Cape Verde", "KSA": "Saudi Arabia",
    "New Zealand": "New Zealand", "BEL": "Belgium", "EGY": "Egypt", "IRN": "Iran",
    "ALG": "Algeria", "AUT": "Austria", "JOR": "Jordan", "ARG": "Argentina",
}
# normalize question-text team variants -> canonical full name
QNAME_FIX = {
    "czechia": "Czech Republic", "türkiye": "Turkey", "turkiye": "Turkey",
    "united states": "USA", "bosnia and herzegovina": "Bosnia & Herzegovina",
    "south korea": "South Korea", "south africa": "South Africa",
    "saudi arabia": "Saudi Arabia", "ivory coast": "Ivory Coast",
    "cape verde": "Cape Verde", "new zealand": "New Zealand", "dr congo": "DR Congo",
    "curaçao": "Curaçao", "curacao": "Curaçao",
}

# Validated SoT map from candidate2_sot_vs_floor backtest (out-of-sample calibrated):
#   E[team SoT | market goal lambda] = SOT_B0 + SOT_B1 * lambda ; NegBin size r
SOT_B0 = 1.284
SOT_B1 = 2.197
SOT_NB_R = 33.78         # NegBin size fit on the panel (out-of-sample validated)
HALF2_GOAL_SHARE = 0.55  # 2nd-half goal share (matches floor)
HALF2_SOT_SHARE = 0.52
OFFSIDE_BASE = 1.9       # floor static base
CARDS_BASE = 3.9         # total cards per match (~ club avg; flagged)


def parse_cache(path):
    raw = json.load(open(path))
    out = {}
    for ev in raw["events"]:
        home, away = ev["home_team"], ev["away_team"]
        books = ev.get("bookmakers", [])
        # collect h2h triples (home, draw, away) per book
        h2h_home, h2h_draw, h2h_away = [], [], []
        pin_h2h = None
        over25, under25 = [], []
        pin_ou = None
        for b in books:
            for m in b.get("markets", []):
                if m["key"] == "h2h":
                    px = {o["name"]: o["price"] for o in m["outcomes"]}
                    if home in px and away in px and "Draw" in px:
                        h2h_home.append(px[home]); h2h_draw.append(px["Draw"]); h2h_away.append(px[away])
                        if b["key"] == "pinnacle":
                            pin_h2h = [px[home], px["Draw"], px[away]]
                elif m["key"] == "totals":
                    o25 = [o for o in m["outcomes"] if abs(o.get("point", 0) - 2.5) < 1e-6]
                    pv = {o["name"]: o["price"] for o in o25}
                    if "Over" in pv and "Under" in pv:
                        over25.append(pv["Over"]); under25.append(pv["Under"])
                        if b["key"] == "pinnacle":
                            pin_ou = [pv["Over"], pv["Under"]]
        if not h2h_home:
            continue
        cons_h2h = [float(np.median(h2h_home)), float(np.median(h2h_draw)), float(np.median(h2h_away))]
        cons_ou = [float(np.median(over25)), float(np.median(under25))] if over25 else None
        out[(home, away)] = {
            "home": home, "away": away, "commence": ev.get("commence_time"),
            "cons_h2h": cons_h2h, "pin_h2h": pin_h2h,
            "cons_ou": cons_ou, "pin_ou": pin_ou, "n_books": len(h2h_home),
        }
    return out


def devig_1x2(odds_triple, source_method):
    """source_method like ('pin','power') or ('cons','shin'). Returns (pH,pD,pA)."""
    p = devig.devig(odds_triple, method=source_method)
    return float(p[0]), float(p[1]), float(p[2])


def _score_matrix(lh, la, rho=-0.05, maxg=12):
    ph = poisson.pmf(np.arange(maxg), lh); pa = poisson.pmf(np.arange(maxg), la)
    M = np.outer(ph, pa)
    M[0, 0] *= 1 - lh * la * rho; M[1, 1] *= 1 - rho
    M[0, 1] *= 1 + lh * rho; M[1, 0] *= 1 + la * rho
    return M / M.sum()


def probs_from_lambdas(lh, la):
    M = _score_matrix(lh, la)
    tot = np.add.outer(np.arange(M.shape[0]), np.arange(M.shape[1]))
    return {
        "pH": float(np.tril(M, -1).sum()), "pD": float(np.trace(M)), "pA": float(np.triu(M, 1).sum()),
        "p_over25": float(M[tot >= 3].sum()),
    }


def solve_lambdas(pH, pA, p_over25=None):
    """Recover (lh,la) from de-vigged 1X2 (+ optional over2.5)."""
    sup = np.log(max(pH, 1e-3) / max(pA, 1e-3))
    x0 = [max(0.2, 1.4 + 0.5 * sup), max(0.2, 1.4 - 0.5 * sup)]

    def resid(x):
        lh, la = max(x[0], 1e-3), max(x[1], 1e-3)
        pr = probs_from_lambdas(lh, la)
        r = [pr["pH"] - pH, pr["pA"] - pA]
        if p_over25 is not None:
            r.append(pr["p_over25"] - p_over25)
        return r

    sol = least_squares(resid, x0, bounds=([0.05, 0.05], [6, 6]))
    return float(sol.x[0]), float(sol.x[1])


# ---- prop pricers -------------------------------------------------------
def sot_lambda(goal_lambda):
    """Validated linear map E[SoT] = b0 + b1*goal_lambda (full match)."""
    return SOT_B0 + SOT_B1 * goal_lambda


def p_sot_ge(goal_lambda, n, half2=False):
    lam = sot_lambda(goal_lambda) * (HALF2_SOT_SHARE if half2 else 1.0)
    p = SOT_NB_R / (SOT_NB_R + lam)
    return float(nbinom.sf(n - 1, SOT_NB_R, p))


def p_a_more_sot_than_b(lam_a_goals, lam_b_goals, half2=False):
    """P(A shots-on-target > B), two independent NegBins."""
    la = sot_lambda(lam_a_goals) * (HALF2_SOT_SHARE if half2 else 1.0)
    lb = sot_lambda(lam_b_goals) * (HALF2_SOT_SHARE if half2 else 1.0)
    ks = np.arange(0, 30)
    pa = nbinom.pmf(ks, SOT_NB_R, SOT_NB_R / (SOT_NB_R + la))
    pb = nbinom.pmf(ks, SOT_NB_R, SOT_NB_R / (SOT_NB_R + lb))
    # P(A>B) = sum_a pa[a] * P(B<a)
    cb = np.cumsum(pb)
    pgt = sum(pa[a] * (cb[a - 1] if a >= 1 else 0.0) for a in range(len(ks)))
    return float(pgt)


def static_sot_floor(n, shrink=0.70, base=4.3):
    p = float(poisson.sf(n - 1, base))
    return 0.5 + shrink * (p - 0.5)
