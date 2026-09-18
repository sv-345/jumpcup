#!/usr/bin/env python3
"""Price every OPEN contest market with each candidate, next to the live floor's
own price and the raw market line, and log the divergences. READ-ONLY: consumes
open_markets.json (pulled read-only) + the live bot's odds cache snapshot. Writes
only to probcup-research/. NEVER submits anything.

Output: reports/open_market_divergences.{json,csv,md} and shadow/forecasts_<ts>.jsonl
(the shadow scorer reads the latter as it resolves over the contest).

Confidence tiers:
  HIGH  - candidate has an out-of-sample backtested edge (win_match de-vig; team SoT)
  MED   - model-anchored to the market (goal-derived); expected ~no edge vs floor
  LOW   - not backtested / needs player-or-lineup data (offsides, discipline, players, corners)
"""
import os, sys, re, json, glob
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "lib"))
import devig, price

CACHE = os.path.join(HERE, "..", "data", "cache", "oddsapi_wc.json")
OPEN = os.path.join(HERE, "data", "open_markets.json")

ALIASES = {
    "Czech Republic": ["czech republic", "czechia", "czech"],
    "Turkey": ["türkiye", "turkiye", "turkey"],
    "USA": ["united states", "usa"],
    "Bosnia & Herzegovina": ["bosnia and herzegovina", "bosnia & herzegovina", "bosnia"],
    "South Korea": ["south korea", "korea republic"],
    "South Africa": ["south africa"],
    "DR Congo": ["dr congo", "democratic republic", "congo"],
    "Ivory Coast": ["ivory coast", "côte", "cote d'ivoire"],
    "Cape Verde": ["cape verde"],
    "Saudi Arabia": ["saudi arabia", "saudi"],
    "New Zealand": ["new zealand"],
    "Curaçao": ["curaçao", "curacao"],
}


def aliases(full):
    return ALIASES.get(full, [full.lower()])


def which_side(q, home, away):
    ql = q.lower()
    h = any(a in ql for a in aliases(home))
    a = any(al in ql for al in aliases(away))
    if "both teams" in ql:
        return "both", None
    if h and a:  # comparison: first-mentioned is subject
        ph = min((ql.find(x) for x in aliases(home) if x in ql), default=99)
        pa = min((ql.find(x) for x in aliases(away) if x in ql), default=99)
        return ("home", "away") if ph < pa else ("away", "home")
    if h:
        return "home", "away"
    if a:
        return "away", "home"
    return None, None


def num(q):
    m = re.search(r"(\d+)\s+or\s+(?:more|fewer)|at least (\d+)|(\d+) or more", q)
    if m:
        return int(next(g for g in m.groups() if g))
    m2 = re.search(r"\b(\d+)\b", q)
    return int(m2.group(1)) if m2 else None


def classify(q):
    """Return (kind, half, threshold-needed?) — coarse routing."""
    ql = q.lower()
    is_2h = "second half" in ql or "2nd half" in ql
    is_1h = "first half" in ql or ("halftime" in ql and "tied" not in ql and "winning" not in ql)
    half = 2 if is_2h else (1 if is_1h else None)
    if "win the match" in ql:
        return "win_match", None
    if "halftime, will the match be tied" in ql:
        return "ht_tie", None
    if "halftime, will" in ql and "winning" in ql:
        return "ht_lead", None
    if "shot on target" in ql or "shots on target" in ql:
        if "both teams" in ql:
            return "both_teams_sot", half
        if "more shots on target than" in ql:
            return "sot_compare", half
        if "total shots on target" in ql or "more total shots" in ql:
            return "total_sot", half
        # player template uses "at least N shot on target" (singular); team uses "N or more shots"
        if "at least" in ql and "shots on target" not in ql:
            return ("player_sot_half" if is_2h else "player_sot"), half
        return "team_sot", half
    if "caught offside" in ql:
        return "team_offsides", None
    if "penalty kick be awarded or a red card" in ql:
        return "pen_or_red", None
    if "penalty kick be awarded" in ql:
        return "pen", None
    if "cards shown" in ql or "receive more cards" in ql or "receive at least" in ql:
        return "cards", half
    if "commit more fouls" in ql:
        return "fouls_compare", None
    if "corner" in ql:
        return "corners", half
    if "score or assist" in ql:
        return "player_involve", None
    if "score a goal" in ql and ("excluding own goals" in ql):
        return "player_scorer", None
    if "both teams score and" in ql:
        return "btts_and_total", None
    if "or fewer total goals" in ql:
        return "total_under", None
    if "or more total goals" in ql:
        return ("half_total_over" if is_2h else "total_over"), half
    if "more total goals than the first half" in ql or "more goals than the first half" in ql:
        return "half_vs_half_goals", None
    if "score more goals than" in ql:
        return "half_goals_compare", half
    if "first goal of the second half" in ql or "first goal of the game" in ql:
        return "first_goal", half
    if "score in the second half" in ql or "score in the first half" in ql:
        return "team_score_half", half
    if "score at least" in ql and "goal" in ql:
        return "team_score", None
    return "other", half


def floor_attack(lam, avg):
    return float(np.clip(lam / max(avg, 0.4), 0.6, 1.6))


def main():
    cache = price.parse_cache(CACHE)
    markets = json.load(open(OPEN))
    # build match name (abbrev) -> full team pair, via the cache ordering
    abbr_pairs = {}
    for m in markets:
        name = m["match"]["name"]
        if " vs " in name:
            a, b = name.split(" vs ")
            abbr_pairs[name] = (price.ABBR2FULL.get(a.strip(), a.strip()),
                                price.ABBR2FULL.get(b.strip(), b.strip()))

    # per-match precomputed odds params
    mparams = {}
    for name, (hf, af) in abbr_pairs.items():
        c = cache.get((hf, af))
        if not c:
            continue
        # candidate1 source: pinnacle+power; floor: consensus-median+shin
        cand_h2h = price.devig_1x2(c["pin_h2h"], "power") if c["pin_h2h"] else price.devig_1x2(c["cons_h2h"], "power")
        floor_h2h = price.devig_1x2(c["cons_h2h"], "shin")
        pov = None
        if c["pin_ou"]:
            pov = devig.two_way(c["pin_ou"][0], c["pin_ou"][1], "power")
        elif c["cons_ou"]:
            pov = devig.two_way(c["cons_ou"][0], c["cons_ou"][1], "power")
        lh, la = price.solve_lambdas(cand_h2h[0], cand_h2h[2], pov)
        mparams[name] = dict(home=hf, away=af, cand_h2h=cand_h2h, floor_h2h=floor_h2h,
                             lam_h=lh, lam_a=la, avg=(lh + la) / 2, pov=pov, raw_h2h=c["cons_h2h"])

    rows = []
    for m in markets:
        name = m["match"]["name"]
        mp = mparams.get(name)
        q = m["question"]
        kind, half = classify(q)
        rec = {"market_id": m["id"], "match": name, "question": q, "kind": kind,
               "closing_time": m["match"]["closing_time"], "candidate": "", "tier": "",
               "raw_line": None, "floor": None, "cand": None, "divergence": None,
               "floor_actual": None, "div_vs_floor_actual": None, "note": ""}
        if mp is None:
            rec["note"] = "no odds match"; rows.append(rec); continue
        hf, af, lh, la, avg = mp["home"], mp["away"], mp["lam_h"], mp["lam_a"], mp["avg"]
        side, other = which_side(q, hf, af)

        if kind == "win_match":
            si = 0 if side == "home" else 2
            rec.update(candidate="C1 de-vig (Pinnacle+power)", tier="HIGH",
                       raw_line=round(1.0 / [mp["raw_h2h"][0], None, mp["raw_h2h"][2]][si], 4) if False else None,
                       floor=round(mp["floor_h2h"][si], 4), cand=round(mp["cand_h2h"][si], 4))
            rec["raw_line"] = round({"home": 1.0 / mp["raw_h2h"][0], "away": 1.0 / mp["raw_h2h"][2]}[side], 4)
            rec["divergence"] = round(rec["cand"] - rec["floor"], 4)
            rec["note"] = "Pinnacle+power vs consensus+shin"
        elif kind in ("team_sot", "team_sot_half"):
            n = num(q) or 4
            lam_team = lh if side == "home" else la
            h2 = (kind == "team_sot_half") or half == 2
            cand = price.p_sot_ge(lam_team, n, half2=h2)
            fmean = (4.3 if not h2 else 2.2) * floor_attack(lam_team, avg)
            from scipy.stats import poisson
            floor = float(poisson.sf(n - 1, fmean))
            rec.update(candidate="C2 SoT model (team NegBin)", tier="HIGH",
                       floor=round(floor, 4), cand=round(cand, 4), divergence=round(cand - floor, 4),
                       note=f"{('home' if side=='home' else 'away')}={hf if side=='home' else af} lam_goals={lam_team:.2f} -> SoT mean {price.sot_lambda(lam_team)*(0.52 if h2 else 1):.2f} vs floor {fmean:.2f}")
        elif kind == "sot_compare":
            h2 = half == 2
            la_g = lh if side == "home" else la
            lb_g = la if side == "home" else lh
            cand = price.p_a_more_sot_than_b(la_g, lb_g, half2=h2)
            # floor: prob_a_more_than_b with clamped attack base 2.2(2h)/4.3
            base = 2.2 if h2 else 4.3
            ma, mb = base * floor_attack(la_g, avg), base * floor_attack(lb_g, avg)
            floor = floor_more_than(ma, mb)
            rec.update(candidate="C2 SoT model (compare)", tier="HIGH",
                       floor=round(floor, 4), cand=round(cand, 4), divergence=round(cand - floor, 4),
                       note=f"{(hf if side=='home' else af)} vs {(af if side=='home' else hf)}")
        elif kind == "both_teams_sot":
            n = num(q) or 1
            h2 = half == 2
            cand = price.p_sot_ge(lh, n, half2=h2) * price.p_sot_ge(la, n, half2=h2)
            from scipy.stats import poisson
            base = 2.2 if h2 else 4.3
            floor = float(poisson.sf(n - 1, base * floor_attack(lh, avg)) * poisson.sf(n - 1, base * floor_attack(la, avg)))
            rec.update(candidate="C2 SoT model (both teams)", tier="HIGH",
                       floor=round(floor, 4), cand=round(cand, 4), divergence=round(cand - floor, 4))
        elif kind == "team_offsides":
            n = num(q) or 2
            from scipy.stats import poisson
            lam_team = lh if side == "home" else la
            # team-adjusted: offsides scale with attacking volume (dampened); NOT backtested
            cand_mean = price.OFFSIDE_BASE * (floor_attack(lam_team, avg) ** 0.7)
            cand = float(poisson.sf(n - 1, cand_mean))
            floor = float(poisson.sf(n - 1, price.OFFSIDE_BASE))
            rec.update(candidate="C2 offsides (team-adj, UNVALIDATED)", tier="LOW",
                       floor=round(floor, 4), cand=round(cand, 4), divergence=round(cand - floor, 4),
                       note="offsides not in backtest data; team-strength heuristic only")
        elif kind in ("total_over", "total_under", "half_total_over", "btts_and_total",
                      "team_score", "team_score_half", "half_goals_compare", "half_vs_half_goals",
                      "ht_tie", "ht_lead", "first_goal"):
            rec.update(candidate="C2 goals (market-anchored)", tier="MED",
                       note="goal-derived: model anchored to market -> ~no edge vs floor (totals backtest: market unbeatable)")
        elif kind in ("pen_or_red", "pen", "cards", "fouls_compare"):
            rec.update(candidate="C2 discipline (base rate)", tier="LOW",
                       note="discipline ~roughly calibrated; floor red-rate slightly high (club data). minor")
        elif kind in ("player_sot", "player_sot_half"):
            # team-scaled heuristic: featured attacker gets ~22% of team SoT. LOW (no lineup data)
            h2 = kind == "player_sot_half"
            # player's team unknown from text -> use the stronger-attacking side as a proxy upper, but
            # default to match-average team since we cannot resolve the player's team reliably
            lam_team = (lh + la) / 2.0
            lam_player = 0.22 * price.sot_lambda(lam_team) * (0.52 if h2 else 1.0)
            cand = float(1 - np.exp(-lam_player))
            floor = 0.45 if h2 else 0.55
            rec.update(candidate="C2 player SoT (heuristic)", tier="LOW",
                       floor=round(floor, 4), cand=round(cand, 4), divergence=round(cand - floor, 4),
                       note="player team unresolved from text; team-scaled heuristic. needs lineup data (API-Football)")
        elif kind in ("player_scorer", "player_involve"):
            lam_team = (lh + la) / 2.0
            share = 0.28 if kind == "player_involve" else 0.22
            cand = float(1 - np.exp(-share * lam_team))
            rec.update(candidate="C2 player (heuristic)", tier="LOW",
                       cand=round(cand, 4),
                       note="needs lineup/role data (API-Football). team-scaled heuristic only")
        elif kind == "total_sot":
            n = num(q) or 8
            from scipy.stats import poisson
            h2 = half == 2
            cand_mean = (price.sot_lambda(lh) + price.sot_lambda(la)) * (0.52 if h2 else 1.0)
            cand = float(poisson.sf(n - 1, cand_mean))  # approx; total SoT ~ Poisson(sum)
            base = 4.4 if h2 else 8.6
            floor = float(poisson.sf(n - 1, base))
            rec.update(candidate="C2 total SoT", tier="MED",
                       floor=round(floor, 4), cand=round(cand, 4), divergence=round(cand - floor, 4),
                       note=f"total SoT mean cand {cand_mean:.1f} vs floor {base}")
        elif kind == "corners":
            rec.update(candidate="C2 corners (not built)", tier="LOW",
                       note="corners modelable from data (HC/AC) but not built this run")
        else:
            rec.update(candidate="-", tier="LOW", note="unclassified/needs data")
        rows.append(rec)

    # join the floor bot's ACTUAL submitted prices (read-only from list_predictions)
    fp_path = os.path.join(HERE, "shadow", "floor_predictions.json")
    floor_actual = json.load(open(fp_path)) if os.path.exists(fp_path) else {}
    for r in rows:
        fa = floor_actual.get(r["market_id"])
        if fa is not None:
            r["floor_actual"] = round(fa / 100.0, 4)
            if r["cand"] is not None:
                r["div_vs_floor_actual"] = round(r["cand"] - r["floor_actual"], 4)

    # outputs
    json.dump(rows, open(os.path.join(HERE, "reports/open_market_divergences.json"), "w"), indent=1)
    import csv
    with open(os.path.join(HERE, "reports/open_market_divergences.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # markdown: highlight HIGH-confidence biggest divergences
    high = [r for r in rows if r["tier"] == "HIGH" and r["div_vs_floor_actual"] is not None]
    high.sort(key=lambda r: -abs(r["div_vs_floor_actual"]))
    lines = ["# Open-market divergences (candidate vs live floor's ACTUAL submitted price)",
             f"{len(rows)} open markets priced. Full table: open_market_divergences.csv",
             "div = candidate price - floor's actual submitted price (from read-only list_predictions)\n",
             "## Top HIGH-confidence divergences (backtested candidates: C1 win-devig, C2 SoT)\n",
             f"{'match':18s} {'kind':14s} {'cand':>6s} {'floorA':>6s} {'div':>7s}  question"]
    for r in high[:45]:
        lines.append(f"{r['match']:18s} {r['kind']:14s} {r['cand']:>6.3f} {r['floor_actual']:>6.3f} {r['div_vs_floor_actual']:>+7.3f}  {r['question'][:58]}")
    # tier counts
    from collections import Counter
    tc = Counter(r["tier"] for r in rows); kc = Counter(r["kind"] for r in rows)
    lines += ["", f"tier counts: {dict(tc)}", f"kind counts: {dict(kc)}"]
    open(os.path.join(HERE, "reports/open_market_divergences.md"), "w").write("\n".join(lines))
    print("\n".join(lines[:55]))
    print("\nwrote reports/open_market_divergences.{json,csv,md}")
    return rows


def floor_more_than(ma, mb):
    from scipy.stats import poisson
    ks = np.arange(0, 25)
    pa = poisson.pmf(ks, ma); pb = poisson.pmf(ks, mb)
    cb = np.cumsum(pb)
    return float(sum(pa[a] * (cb[a - 1] if a >= 1 else 0) for a in range(len(ks))))


if __name__ == "__main__":
    main()
