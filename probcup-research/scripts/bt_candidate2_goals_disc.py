#!/usr/bin/env python3
"""Candidate 2 controls:
 (A) Totals: does a Dixon-Coles goals model BEAT the de-vigged closing total line on
     O/U 2.5 out of sample? Expectation: NO (market is efficient) -> validates the
     floor's market-anchoring for goal markets; don't replace it with a pure model.
 (B) Discipline base rate: empirical P(>=1 red card) and the implied pen-or-red rate
     vs the floor's static numbers (pen 0.28, red 0.13 -> ~0.336). Calibration check.

Read-only on local CSVs. No contest API.
"""
import os, sys, glob, re, json
import numpy as np, pandas as pd
from scipy.stats import poisson
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "lib"))
import devig, scoring

TRAIN = {"1819", "1920", "2021", "2122", "2223"}; TEST = {"2324", "2425", "2526"}


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, "data/hist/*.csv"))):
        s = re.search(r"_(\d{4})\.csv", f)
        if not s:
            continue
        try:
            df = pd.read_csv(f, encoding="latin-1", on_bad_lines="skip")
        except Exception:
            continue
        df["__season"] = s.group(1)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def fit_goals(train, k=5.0):
    t = train.dropna(subset=["FTHG", "FTAG", "HomeTeam", "AwayTeam"]).copy()
    t["FTHG"] = pd.to_numeric(t.FTHG, errors="coerce"); t["FTAG"] = pd.to_numeric(t.FTAG, errors="coerce")
    t = t.dropna(subset=["FTHG", "FTAG"])
    lg_h, lg_a = t.FTHG.mean(), t.FTAG.mean()
    lg = (t.FTHG.sum() + t.FTAG.sum()) / (2 * len(t))
    teams = set(t.HomeTeam) | set(t.AwayTeam)
    att, dfn = {}, {}
    for tm in teams:
        gf = np.concatenate([t.loc[t.HomeTeam == tm, "FTHG"].values, t.loc[t.AwayTeam == tm, "FTAG"].values])
        ga = np.concatenate([t.loc[t.HomeTeam == tm, "FTAG"].values, t.loc[t.AwayTeam == tm, "FTHG"].values])
        att[tm] = (gf.sum() + k * lg) / ((len(gf) + k) * lg)
        dfn[tm] = (ga.sum() + k * lg) / ((len(ga) + k) * lg)
    return dict(lg_h=lg_h, lg_a=lg_a, att=att, dfn=dfn)


def p_over25(gm, home, away, rho=-0.05, maxg=12):
    lh = gm["lg_h"] * gm["att"].get(home, 1) * gm["dfn"].get(away, 1)
    la = gm["lg_a"] * gm["att"].get(away, 1) * gm["dfn"].get(home, 1)
    ph = poisson.pmf(np.arange(maxg), lh); pa = poisson.pmf(np.arange(maxg), la)
    M = np.outer(ph, pa)
    # Dixon-Coles low-score correction
    M[0, 0] *= 1 - lh * la * rho; M[1, 1] *= 1 - rho
    M[0, 1] *= 1 + lh * rho;      M[1, 0] *= 1 + la * rho
    M /= M.sum()
    tot_idx = np.add.outer(np.arange(maxg), np.arange(maxg))
    return float(M[tot_idx >= 3].sum())


def main():
    df = load()
    train = df[df["__season"].isin(TRAIN)]; test = df[df["__season"].isin(TEST)]
    gm = fit_goals(train)

    # (A) totals: model vs market on common test rows with Pinnacle closing total
    cols = ["PC>2.5", "PC<2.5"]
    tt = test.dropna(subset=["FTHG", "FTAG", "HomeTeam", "AwayTeam"] + cols).copy()
    tt["FTHG"] = pd.to_numeric(tt.FTHG, errors="coerce"); tt["FTAG"] = pd.to_numeric(tt.FTAG, errors="coerce")
    for c in cols:
        tt = tt[pd.to_numeric(tt[c], errors="coerce") > 1.0]
    tt = tt.dropna(subset=["FTHG", "FTAG"])
    y = ((tt.FTHG + tt.FTAG) > 2.5).astype(float).values
    p_mkt = np.array([devig.two_way(float(r["PC>2.5"]), float(r["PC<2.5"]), "power") for _, r in tt.iterrows()])
    p_mod = np.array([p_over25(gm, r.HomeTeam, r.AwayTeam) for _, r in tt.iterrows()])
    # 50/50 blend too
    p_blend = 0.5 * p_mkt + 0.5 * p_mod
    res = {}
    for name, p in [("market_close", p_mkt), ("goals_model", p_mod), ("blend_50_50", p_blend)]:
        br = scoring.brier_binary(p, y); ll = scoring.logloss_binary(p, y)
        res[name] = {"brier": float(br.mean()), "logloss": float(ll.mean())}
    bs = scoring.paired_bootstrap(scoring.brier_binary(p_mod, y), scoring.brier_binary(p_mkt, y), n_boot=5000)
    res["model_minus_market_brier"] = bs["mean_diff"]; res["p_model_not_better"] = bs["p_A_not_better"]
    bs2 = scoring.paired_bootstrap(scoring.brier_binary(p_blend, y), scoring.brier_binary(p_mkt, y), n_boot=5000)
    res["blend_minus_market_brier"] = bs2["mean_diff"]; res["p_blend_not_better"] = bs2["p_A_not_better"]
    res["n"] = int(len(tt))

    # (B) discipline
    test_d = test.dropna(subset=["HR", "AR"]).copy()
    test_d["HR"] = pd.to_numeric(test_d.HR, errors="coerce"); test_d["AR"] = pd.to_numeric(test_d.AR, errors="coerce")
    reds = (test_d.HR.fillna(0) + test_d.AR.fillna(0))
    p_red_ge1 = float((reds >= 1).mean())
    lam_red = float(reds.mean())
    disc = {"n": int(len(test_d)), "emp_P_ge1_red": p_red_ge1, "reds_per_match": lam_red,
            "floor_red_lambda_total": 0.26, "floor_implied_P_ge1_red": float(1 - np.exp(-0.26)),
            "floor_pen_or_red": float(1 - np.exp(-0.28) * np.exp(-0.13))}

    out = {"totals_model_vs_market": res, "discipline_calibration": disc}
    json.dump(out, open(os.path.join(HERE, "reports/candidate2_goals_disc.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
