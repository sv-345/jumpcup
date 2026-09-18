#!/usr/bin/env python3
"""Candidate 2 (FAITHFUL): team SoT model vs the LIVE FLOOR's actual SoT pricing.

The floor is NOT flat: team_sot mean = 4.3 * clamp(team_goal_lambda/avg_lambda, 0.6, 1.6),
Poisson. Both the floor and my model anchor to the SAME de-vigged market goal lambdas
(closing 1X2 + O/U 2.5). The differences under test:
  - floor CLAMPS the strength multiplier to [0.6,1.6]; my model does not (full lambda spread)
  - floor uses 4.3*ratio scaling + Poisson; mine uses a calibrated linear map + NegBin
We calibrate my map (intercept/slope + dispersion) on TRAIN seasons and evaluate on TEST
seasons, out of sample, then compare to the floor on Brier/logloss with paired bootstrap.

Read-only on local CSVs. No contest API.
"""
import os, sys, glob, re, json
import numpy as np, pandas as pd
from scipy.stats import poisson, nbinom
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "lib"))
import devig, scoring, price

TRAIN = {"1819", "1920", "2021", "2122"}; TEST = {"2223", "2324", "2425", "2526"}
THRESH = [3, 4, 5]
P1X2 = ["PSCH", "PSCD", "PSCA"]; POU = ["PC>2.5", "PC<2.5"]


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


def solve_lams(df):
    """Return arrays lam_home, lam_away from de-vigged closing 1X2 + O/U2.5 (power)."""
    lh = np.full(len(df), np.nan); la = np.full(len(df), np.nan)
    for i, (_, r) in enumerate(df.iterrows()):
        pH, pD, pA = devig.devig([r.PSCH, r.PSCD, r.PSCA], "power")
        pov = devig.two_way(r["PC>2.5"], r["PC<2.5"], "power")
        a, b = price.solve_lambdas(float(pH), float(pA), float(pov))
        lh[i] = a; la[i] = b
    return lh, la


def main():
    df = load()
    keep = ["FTHG", "FTAG", "HST", "AST", "HomeTeam", "AwayTeam", "__season"] + P1X2 + POU
    df = df[[c for c in keep if c in df.columns]].dropna(subset=P1X2 + POU + ["HST", "AST"])
    for c in P1X2 + POU:
        df = df[pd.to_numeric(df[c], errors="coerce") > 1.0]
    df = df.reset_index(drop=True)
    df["HST"] = pd.to_numeric(df.HST, errors="coerce"); df["AST"] = pd.to_numeric(df.AST, errors="coerce")
    df = df.dropna(subset=["HST", "AST"]).reset_index(drop=True)
    print(f"matches with closing 1X2+OU+SoT: {len(df)}")

    lh, la = solve_lams(df)
    df["lam_h"], df["lam_a"] = lh, la
    avg = (lh + la) / 2.0

    tr = df["__season"].isin(TRAIN); te = df["__season"].isin(TEST)
    # stack to team-games: (lam_team, lam_avg_match, sot, is_test)
    rec_lam = np.concatenate([df.lam_h.values, df.lam_a.values])
    rec_avg = np.concatenate([avg, avg])
    rec_sot = np.concatenate([df.HST.values, df.AST.values])
    rec_te = np.concatenate([te.values, te.values])
    rec_tr = np.concatenate([tr.values, tr.values])

    # calibrate MY map on train: SoT_mean = b0 + b1*lam_team ; dispersion r
    X = rec_lam[rec_tr]; Y = rec_sot[rec_tr]
    b1, b0 = np.polyfit(X, Y, 1)
    mu_tr = b0 + b1 * X
    var = np.mean((Y - mu_tr) ** 2); mbar = mu_tr.mean()
    r = max(2.0, mbar ** 2 / max(var - mbar, 1e-6)) if var > mbar else 50.0
    print(f"my map: SoT_mean = {b0:.3f} + {b1:.3f}*lam_goals ; NegBin r = {r:.2f}")

    # TEST set
    lam_t = rec_lam[rec_te]; avg_t = rec_avg[rec_te]; sot_t = rec_sot[rec_te]
    n_test = len(sot_t)
    my_mean = np.clip(b0 + b1 * lam_t, 0.3, None)
    floor_mult = np.clip(lam_t / np.clip(avg_t, 0.4, None), 0.6, 1.6)
    floor_mean = 4.3 * floor_mult

    out_lines = [f"# Candidate 2 (faithful): team SoT model vs LIVE FLOOR (clamped market-anchored)",
                 f"out-of-sample test team-games = {n_test} (seasons {sorted(TEST)})",
                 f"my map SoT_mean={b0:.3f}+{b1:.3f}*lam ; NegBin r={r:.2f}; floor=4.3*clamp(lam/avg,0.6,1.6) Poisson\n"]
    report = {"n_test": int(n_test), "map": {"b0": b0, "b1": b1, "r": r}, "by_threshold": {}}
    for n in THRESH:
        y = (sot_t >= n).astype(float)
        p_mine = nbinom.sf(n - 1, r, r / (r + my_mean))
        p_floor = poisson.sf(n - 1, floor_mean)
        br_m = scoring.brier_binary(p_mine, y); br_f = scoring.brier_binary(p_floor, y)
        ll_m = scoring.logloss_binary(p_mine, y); ll_f = scoring.logloss_binary(p_floor, y)
        bs = scoring.paired_bootstrap(br_m, br_f, n_boot=5000)
        # where does the edge concentrate? bucket by |floor_mult at clamp bound|
        clamped = (floor_mult <= 0.6001) | (floor_mult >= 1.5999)
        edge_clamped = float((br_f[clamped] - br_m[clamped]).mean()) if clamped.sum() else float("nan")
        edge_unclamped = float((br_f[~clamped] - br_m[~clamped]).mean()) if (~clamped).sum() else float("nan")
        report["by_threshold"][n] = {
            "emp_rate": float(y.mean()), "mine_brier": float(br_m.mean()), "floor_brier": float(br_f.mean()),
            "mine_logloss": float(ll_m.mean()), "floor_logloss": float(ll_f.mean()),
            "mine_minus_floor_brier": bs["mean_diff"], "p_mine_not_better": bs["p_A_not_better"],
            "frac_clamped": float(clamped.mean()), "edge_on_clamped": edge_clamped, "edge_on_unclamped": edge_unclamped}
        out_lines += [f"## SoT >= {n}  (emp {y.mean():.3f})",
                      f"  mine  Brier={br_m.mean():.5f} logloss={ll_m.mean():.5f}",
                      f"  floor Brier={br_f.mean():.5f} logloss={ll_f.mean():.5f}",
                      f"  mine-floor dBrier={bs['mean_diff']:+.5f}  p_mine_not_better={bs['p_A_not_better']:.3f}",
                      f"  clamp binds on {clamped.mean()*100:.1f}% of games; "
                      f"edge there={edge_clamped:+.5f} vs unclamped={edge_unclamped:+.5f}\n"]
    json.dump(report, open(os.path.join(HERE, "reports/candidate2_sot_vs_floor.json"), "w"), indent=1)
    open(os.path.join(HERE, "reports/candidate2_sot_vs_floor.md"), "w").write("\n".join(out_lines))
    print("\n".join(out_lines))


if __name__ == "__main__":
    main()
