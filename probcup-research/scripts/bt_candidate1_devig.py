#!/usr/bin/env python3
"""Candidate 1 backtest: which de-vig method + which odds source best predicts
outcomes on historical CLOSING lines (1X2 and Over/Under 2.5).

Compares the live floor's choices (consensus/avg + Shin/power) against the
alternatives it lacks (Pinnacle-only source; additive / odds-ratio methods),
out of sample, with paired-bootstrap significance. The de-vigged closing line is
the benchmark; we ask which transform/source of it scores best on Brier+logloss.

Read-only on local CSVs. No contest API, no live writes.
"""
import os, sys, json, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "lib"))
import devig, scoring

HIST = os.path.join(HERE, "data", "hist")
REP = os.path.join(HERE, "reports")
os.makedirs(REP, exist_ok=True)

# ---- 1X2 closing-odds sources: (label, [Home,Draw,Away] columns) ----
SRC_1X2 = {
    "Pinnacle_close":  ["PSCH", "PSCD", "PSCA"],   # sharp single book, closing
    "Avg_close":       ["AvgCH", "AvgCD", "AvgCA"],  # consensus mean ~ live bot median
    "Bet365_close":    ["B365CH", "B365CD", "B365CA"],
    "Max_close":       ["MaxCH", "MaxCD", "MaxCA"],
    "Pinnacle_open":   ["PSH", "PSD", "PSA"],       # for open-vs-close comparison
    "Avg_open":        ["AvgH", "AvgD", "AvgA"],
}
# ---- Over/Under 2.5 closing sources: (label, [Over,Under] columns) ----
SRC_OU = {
    "Pinnacle_close": ["PC>2.5", "PC<2.5"],
    "Avg_close":      ["AvgC>2.5", "AvgC<2.5"],
    "Bet365_close":   ["B365C>2.5", "B365C<2.5"],
    "Pinnacle_open":  ["P>2.5", "P<2.5"],
}
METHODS = ["multiplicative", "additive", "power", "shin", "log"]


def load_all():
    frames = []
    for f in sorted(glob.glob(os.path.join(HIST, "*.csv"))):
        try:
            df = pd.read_csv(f, encoding="latin-1", on_bad_lines="skip")
        except Exception:
            continue
        df["__file"] = os.path.basename(f)
        frames.append(df)
    big = pd.concat(frames, ignore_index=True)
    return big


def run_1x2(df):
    out = {}
    res_map = {"H": 0, "D": 1, "A": 2}
    valid_res = df["FTR"].isin(res_map) if "FTR" in df else pd.Series(False, index=df.index)
    for src, cols in SRC_1X2.items():
        if not all(c in df.columns for c in cols):
            continue
        sub = df[valid_res & df[cols].notna().all(axis=1)].copy()
        # drop nonsense odds
        for c in cols:
            sub = sub[pd.to_numeric(sub[c], errors="coerce") > 1.0]
        if len(sub) < 200:
            continue
        odds = sub[cols].apply(pd.to_numeric, errors="coerce").values  # (N,3)
        y = sub["FTR"].map(res_map).values.astype(int)
        for m in METHODS:
            P = np.array([devig.METHODS[m](o) for o in odds])
            br = scoring.brier_multi(P, y)
            ll = scoring.logloss_multi(P, y)
            out[(src, m)] = {"brier": br, "logloss": ll, "n": len(sub)}
    return out


def run_ou(df):
    out = {}
    if not ("FTHG" in df and "FTAG" in df):
        return out
    tot = pd.to_numeric(df["FTHG"], errors="coerce") + pd.to_numeric(df["FTAG"], errors="coerce")
    over = (tot > 2.5).astype(float)
    for src, cols in SRC_OU.items():
        if not all(c in df.columns for c in cols):
            continue
        mask = df[cols].notna().all(axis=1) & tot.notna()
        for c in cols:
            mask &= pd.to_numeric(df[c], errors="coerce") > 1.0
        sub_idx = df.index[mask]
        if len(sub_idx) < 200:
            continue
        oo = df.loc[sub_idx, cols].apply(pd.to_numeric, errors="coerce").values  # (N,2) [over,under]
        y = over.loc[sub_idx].values
        for m in METHODS:
            p_over = np.array([devig.METHODS[m](o)[0] for o in oo])
            br = scoring.brier_binary(p_over, y)
            ll = scoring.logloss_binary(p_over, y)
            out[(src, m)] = {"brier": br, "logloss": ll, "n": len(sub_idx)}
    return out


def table(out, baseline_key, title):
    rows = []
    base = out.get(baseline_key)
    for (src, m), d in sorted(out.items(), key=lambda kv: kv[1]["brier"].mean()):
        r = {"source": src, "method": m, "n": d["n"],
             "brier": float(d["brier"].mean()), "logloss": float(d["logloss"].mean())}
        if base is not None and d["brier"].shape == base["brier"].shape:
            bs = scoring.paired_bootstrap(d["brier"], base["brier"], n_boot=4000)
            r["d_vs_base"] = bs["mean_diff"]
            r["p_not_better"] = bs["p_A_not_better"]
        rows.append(r)
    return {"title": title, "baseline": list(baseline_key), "rows": rows}


def main():
    df = load_all()
    print(f"loaded {len(df)} rows from {df['__file'].nunique()} files")
    r1 = run_1x2(df)
    r2 = run_ou(df)
    # live-bot baseline: consensus(avg) closing + shin (3-way) / power (2-way)
    t1 = table(r1, ("Avg_close", "shin"), "1X2 closing-line de-vig (benchmark: Avg_close+shin = live-bot style)")
    t2 = table(r2, ("Avg_close", "power"), "Over/Under 2.5 closing-line de-vig (benchmark: Avg_close+power)")
    result = {"oneXtwo": t1, "over_under": t2,
              "rows_total": int(len(df))}
    json.dump(result, open(os.path.join(REP, "candidate1_devig.json"), "w"), indent=1)

    # readable
    lines = []
    for t in (t1, t2):
        lines.append(f"\n## {t['title']}")
        lines.append(f"baseline = {t['baseline']}  (lower Brier = better; d_vs_base<0 means better than baseline)")
        lines.append(f"{'source':16s} {'method':14s} {'n':>6s} {'brier':>9s} {'logloss':>9s} {'dBrier':>9s} {'p_notbetter':>11s}")
        for r in t["rows"]:
            lines.append(f"{r['source']:16s} {r['method']:14s} {r['n']:>6d} {r['brier']:>9.5f} {r['logloss']:>9.5f} "
                         f"{r.get('d_vs_base', float('nan')):>9.5f} {r.get('p_not_better', float('nan')):>11.3f}")
    open(os.path.join(REP, "candidate1_devig.md"), "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
