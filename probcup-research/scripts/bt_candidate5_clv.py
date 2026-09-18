#!/usr/bin/env python3
"""Candidate 5: CLV timing study. Does the CLOSING line beat the OPENING line at
predicting outcomes (lower Brier/logloss)? If yes, last-second resubmission near
market close captures real information -> the live bot's latency engineering /
lock-window resubmit earns its keep. We also bucket by line-move magnitude to show
where the closing edge concentrates.

Uses Pinnacle open (PSH/PSD/PSA, P>2.5/P<2.5) vs Pinnacle close (PSCH.., PC>2.5..).
Read-only on local CSVs. No contest API.
"""
import os, sys, json, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "lib"))
import devig, scoring

HIST = os.path.join(HERE, "data", "hist")
REP = os.path.join(HERE, "reports")


def load_all():
    frames = []
    for f in sorted(glob.glob(os.path.join(HIST, "*.csv"))):
        try:
            frames.append(pd.read_csv(f, encoding="latin-1", on_bad_lines="skip"))
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True)


def devig_rows(odds, method, two_way=False):
    if two_way:
        return np.array([devig.METHODS[method](o)[0] for o in odds])
    return np.array([devig.METHODS[method](o) for o in odds])


def main():
    df = load_all()
    res_map = {"H": 0, "D": 1, "A": 2}
    out = {"oneXtwo": {}, "over_under": {}}

    # ---- 1X2 open vs close (Pinnacle, shin) ----
    o_cols, c_cols = ["PSH", "PSD", "PSA"], ["PSCH", "PSCD", "PSCA"]
    m = df["FTR"].isin(res_map)
    for cols in (o_cols, c_cols):
        m &= df[cols].notna().all(axis=1)
        for c in cols:
            m &= pd.to_numeric(df[c], errors="coerce") > 1.0
    sub = df[m]
    y = sub["FTR"].map(res_map).values.astype(int)
    oo = sub[o_cols].apply(pd.to_numeric, errors="coerce").values
    cc = sub[c_cols].apply(pd.to_numeric, errors="coerce").values
    Po = devig_rows(oo, "shin"); Pc = devig_rows(cc, "shin")
    bro = scoring.brier_multi(Po, y); brc = scoring.brier_multi(Pc, y)
    llo = scoring.logloss_multi(Po, y); llc = scoring.logloss_multi(Pc, y)
    bs = scoring.paired_bootstrap(brc, bro, n_boot=5000)  # close - open ; <0 => close better
    out["oneXtwo"] = {
        "n": int(len(sub)),
        "open_brier": float(bro.mean()), "close_brier": float(brc.mean()),
        "open_logloss": float(llo.mean()), "close_logloss": float(llc.mean()),
        "close_minus_open_brier": bs["mean_diff"], "ci95": bs["ci95"],
        "p_close_not_better": bs["p_A_not_better"],
    }
    # move-magnitude buckets: |p_close(fav) - p_open(fav)| on the home prob
    move = np.abs(Pc[:, 0] - Po[:, 0])
    qs = np.quantile(move, [0.5, 0.8, 0.95])
    buckets = []
    edges = [(-1, qs[0], "small <50%"), (qs[0], qs[1], "med 50-80%"),
             (qs[1], qs[2], "large 80-95%"), (qs[2], 1.1, "huge >95%")]
    for lo, hi, lbl in edges:
        bm = (move > lo) & (move <= hi)
        if bm.sum() < 30:
            continue
        buckets.append({"bucket": lbl, "n": int(bm.sum()),
                        "open_brier": float(bro[bm].mean()),
                        "close_brier": float(brc[bm].mean()),
                        "edge_close": float((bro[bm] - brc[bm]).mean())})
    out["oneXtwo"]["move_buckets"] = buckets

    # ---- O/U 2.5 open vs close (Pinnacle, power) ----
    oc, cc2 = ["P>2.5", "P<2.5"], ["PC>2.5", "PC<2.5"]
    tot = pd.to_numeric(df["FTHG"], errors="coerce") + pd.to_numeric(df["FTAG"], errors="coerce")
    over = (tot > 2.5).astype(float)
    m2 = tot.notna()
    for cols in (oc, cc2):
        m2 &= df[cols].notna().all(axis=1)
        for c in cols:
            m2 &= pd.to_numeric(df[c], errors="coerce") > 1.0
    sub2 = df[m2]
    yo = over[m2].values
    oo2 = sub2[oc].apply(pd.to_numeric, errors="coerce").values
    cc2v = sub2[cc2].apply(pd.to_numeric, errors="coerce").values
    po = devig_rows(oo2, "power", two_way=True); pc = devig_rows(cc2v, "power", two_way=True)
    bro2 = scoring.brier_binary(po, yo); brc2 = scoring.brier_binary(pc, yo)
    bs2 = scoring.paired_bootstrap(brc2, bro2, n_boot=5000)
    out["over_under"] = {
        "n": int(len(sub2)),
        "open_brier": float(bro2.mean()), "close_brier": float(brc2.mean()),
        "close_minus_open_brier": bs2["mean_diff"], "ci95": bs2["ci95"],
        "p_close_not_better": bs2["p_A_not_better"],
    }
    json.dump(out, open(os.path.join(REP, "candidate5_clv.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
