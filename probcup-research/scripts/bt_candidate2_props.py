#!/usr/bin/env python3
"""Candidate 2 (prop sub-model): team shots-on-target vs the floor's STATIC base rate.

Out-of-sample by season: fit team SoT attack/defense ratings on 1819-2223, test on
2324/2425/2526. For each test team-game and threshold n in {3,4,5}, score
  P(team SoT >= n)  for:
    - my team model (Poisson)
    - my team model (NegBin, overdispersion fit)
    - floor static Poisson(4.3)
    - floor static Poisson(4.3) shrunk 0.70 toward 0.5 (the live floor's actual output)
against the realized outcome. Brier + logloss + calibration + paired-bootstrap signif.

Read-only on local CSVs. No contest API.
"""
import os, sys, glob, json, re
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "lib"))
import props, scoring

TRAIN = {"1819", "1920", "2021", "2122", "2223"}
TEST = {"2324", "2425", "2526"}
THRESH = [3, 4, 5]


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, "data/hist/*.csv"))):
        season = re.search(r"_(\d{4})\.csv", f)
        if not season:
            continue
        try:
            df = pd.read_csv(f, encoding="latin-1", on_bad_lines="skip")
        except Exception:
            continue
        df["__season"] = season.group(1)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def main():
    df = load()
    df = df.dropna(subset=["HST", "AST", "HomeTeam", "AwayTeam"])
    df["HST"] = pd.to_numeric(df["HST"], errors="coerce")
    df["AST"] = pd.to_numeric(df["AST"], errors="coerce")
    df = df.dropna(subset=["HST", "AST"])
    train = df[df["__season"].isin(TRAIN)]
    test = df[df["__season"].isin(TEST)]
    print(f"train team-games: {2*len(train)}, test team-games: {2*len(test)}")

    m_pois = props.SoTModel(dist="poisson").fit(train)
    m_nb = props.SoTModel(dist="nbinom").fit(train)
    print(f"fitted NegBin size r = {m_nb.r:.2f}, lg_home={m_nb.lg_home:.2f}, lg_away={m_nb.lg_away:.2f}")
    static = props.StaticBaseRate(base=4.3, shrink=1.0)
    static_shrunk = props.StaticBaseRate(base=4.3, shrink=0.70)

    # build test team-game records: (team, opp, home, sot)
    recs = []
    for _, r in test.iterrows():
        recs.append((r.HomeTeam, r.AwayTeam, True, r.HST))
        recs.append((r.AwayTeam, r.HomeTeam, False, r.AST))

    report = {"n_test_team_games": len(recs), "negbin_r": m_nb.r, "by_threshold": {}}
    out_lines = [f"# Candidate 2: team SoT model vs static base rate (out-of-sample {sorted(TEST)})",
                 f"test team-games = {len(recs)}, NegBin r = {m_nb.r:.2f}\n"]
    for n in THRESH:
        y = np.array([1.0 if sot >= n else 0.0 for *_, sot in recs])
        preds = {}
        for name, model in [("team_poisson", m_pois), ("team_nbinom", m_nb),
                            ("floor_static4.3", static), ("floor_static_shrunk0.7", static_shrunk)]:
            p = np.array([model.p_ge(t, o, h, n) for (t, o, h, _) in recs])
            preds[name] = p
        base = preds["floor_static_shrunk0.7"]
        thr = {"base_rate_outcome": float(y.mean())}
        out_lines.append(f"## SoT >= {n}  (empirical rate {y.mean():.3f}, n={len(y)})")
        out_lines.append(f"{'model':24s} {'brier':>9s} {'logloss':>9s} {'dVsFloor':>10s} {'p_notbetter':>12s}")
        for name, p in sorted(preds.items(), key=lambda kv: scoring.brier_binary(kv[1], y).mean()):
            br = scoring.brier_binary(p, y); ll = scoring.logloss_binary(p, y)
            bs = scoring.paired_bootstrap(br, scoring.brier_binary(base, y), n_boot=4000)
            thr[name] = {"brier": float(br.mean()), "logloss": float(ll.mean()),
                         "d_vs_floor": bs["mean_diff"], "p_not_better": bs["p_A_not_better"]}
            out_lines.append(f"{name:24s} {br.mean():>9.5f} {ll.mean():>9.5f} "
                             f"{bs['mean_diff']:>10.5f} {bs['p_A_not_better']:>12.3f}")
        # calibration of the best model (nbinom)
        cal = scoring.calibration(preds["team_nbinom"], y, bins=8)
        thr["nbinom_calibration"] = cal
        report["by_threshold"][n] = thr
        out_lines.append("")

    json.dump(report, open(os.path.join(HERE, "reports/candidate2_props.json"), "w"), indent=1)
    open(os.path.join(HERE, "reports/candidate2_props.md"), "w").write("\n".join(out_lines))
    print("\n".join(out_lines))


if __name__ == "__main__":
    main()
