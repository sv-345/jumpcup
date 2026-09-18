#!/usr/bin/env python3
"""Candidate-1 refinement: clean PAIRED tests on the common subset (matches that
have BOTH Pinnacle-closing and consensus-Avg-closing) so source and method effects
are isolated with significance. Answers the promotable question directly:
  Does switching the live floor from {Avg/consensus + Shin} to {Pinnacle + power}
  reduce Brier out of sample, and is it significant?
"""
import os, sys, glob
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "lib"))
import devig, scoring

df = pd.concat([pd.read_csv(f, encoding="latin-1", on_bad_lines="skip")
                for f in sorted(glob.glob(os.path.join(HERE, "data/hist/*.csv")))],
               ignore_index=True)
res_map = {"H": 0, "D": 1, "A": 2}
P = ["PSCH", "PSCD", "PSCA"]; A = ["AvgCH", "AvgCD", "AvgCA"]
m = df["FTR"].isin(res_map)
for cols in (P, A):
    m &= df[cols].notna().all(axis=1)
    for c in cols:
        m &= pd.to_numeric(df[c], errors="coerce") > 1.0
sub = df[m]
y = sub["FTR"].map(res_map).values.astype(int)
pin = sub[P].apply(pd.to_numeric, errors="coerce").values
avg = sub[A].apply(pd.to_numeric, errors="coerce").values
print(f"common paired subset n = {len(sub)}")

def brier(odds, method):
    return scoring.brier_multi(np.array([devig.METHODS[method](o) for o in odds]), y)

configs = {
    "LIVE: Avg+shin": brier(avg, "shin"),
    "Avg+power": brier(avg, "power"),
    "Pinnacle+shin": brier(pin, "shin"),
    "PROMOTE?: Pinnacle+power": brier(pin, "power"),
    "Pinnacle+multiplicative": brier(pin, "multiplicative"),
}
base = configs["LIVE: Avg+shin"]
print(f"\n{'config':28s} {'brier':>9s} {'dVsLive':>10s} {'p_notbetter':>12s}")
for name, br in sorted(configs.items(), key=lambda kv: kv[1].mean()):
    bs = scoring.paired_bootstrap(br, base, n_boot=8000)
    print(f"{name:28s} {br.mean():>9.5f} {bs['mean_diff']:>10.5f} {bs['p_A_not_better']:>12.3f}")
# isolate the source effect: Pinnacle+power vs Avg+power (same method)
bs_src = scoring.paired_bootstrap(configs["PROMOTE?: Pinnacle+power"], configs["Avg+power"], n_boot=8000)
print(f"\nSOURCE effect (Pinnacle+power - Avg+power): dBrier={bs_src['mean_diff']:.5f} "
      f"CI={[round(x,5) for x in bs_src['ci95']]} p_notbetter={bs_src['p_A_not_better']:.3f}")
