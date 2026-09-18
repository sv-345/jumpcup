#!/usr/bin/env python3
"""Shadow scorer — STEP 2 (score). Join the locked candidate forecasts (ledger.jsonl)
to RESOLUTIONS and score each candidate's Brier against the live floor's realized Brier.

Resolution source (read-only, no submits): the live floor bot's settled predictions from
the SportsPredict `list_results` endpoint. Because the floor shares the contest account,
its settled predictions reveal each market's outcome AND the floor's realized Brier — the
gold benchmark. An agent must dump that read-only response to shadow/results_dump.json:
each item like {"market_id","probability","brier_score", ...}. We recover the binary
outcome y from the floor's (probability, brier_score), then score OUR candidate forecast.

Re-runnable and idempotent: recomputes the scoreboard from the current ledger + dump.
Accumulates a history line each run so the operator can watch the edge converge.
"""
import os, sys, json, glob, datetime
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SH = os.path.join(HERE, "shadow")
LEDGER = os.path.join(SH, "ledger.jsonl")
DUMP = os.path.join(SH, "results_dump.json")


def recover_outcome(prob_pct, brier):
    """y in {0,1} from floor's submitted prob (0-100) and realized Brier=(p-y)^2."""
    p = prob_pct / 100.0
    return 1 if abs((1 - p) ** 2 - brier) <= abs(p ** 2 - brier) else 0


def latest_forecasts():
    """Last logged forecast per market_id (most recent ts wins)."""
    best = {}
    if not os.path.exists(LEDGER):
        return best
    for line in open(LEDGER):
        r = json.loads(line)
        mid = r["market_id"]
        if mid not in best or r["ts"] > best[mid]["ts"]:
            best[mid] = r
    return best


def main():
    fc = latest_forecasts()
    results = json.load(open(DUMP)) if os.path.exists(DUMP) else []
    settled = [r for r in results if r.get("brier_score") is not None]
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

    per = {"by_tier": {}, "by_kind": {}, "overall": {}}
    cand_scores, floor_scores, kinds, tiers = [], [], [], []
    matched = 0
    for r in settled:
        mid = r["market_id"]
        if mid not in fc:
            continue
        y = recover_outcome(r["probability"], r["brier_score"])
        f = fc[mid]
        if f.get("cand") is None:
            continue
        cb = (f["cand"] - y) ** 2
        fb = r["brier_score"]  # floor's realized Brier (ground truth)
        cand_scores.append(cb); floor_scores.append(fb)
        kinds.append(f["kind"]); tiers.append(f["tier"]); matched += 1

    def agg(cs, fs):
        cs, fs = np.array(cs), np.array(fs)
        d = {"n": int(len(cs)),
             "cand_brier": float(cs.mean()) if len(cs) else None,
             "floor_brier": float(fs.mean()) if len(fs) else None}
        if len(cs):
            d["cand_minus_floor"] = float((cs - fs).mean())
            if len(cs) > 30:
                rng = np.random.default_rng(0)
                idx = rng.integers(0, len(cs), size=(4000, len(cs)))
                boot = (cs - fs)[idx].mean(axis=1)
                d["p_cand_not_better"] = float(np.mean(boot >= 0))
        return d

    cand_scores_a = np.array(cand_scores); floor_scores_a = np.array(floor_scores)
    kinds_a = np.array(kinds); tiers_a = np.array(tiers)
    per["overall"] = agg(cand_scores, floor_scores)
    for t in set(tiers):
        m = tiers_a == t
        per["by_tier"][t] = agg(cand_scores_a[m], floor_scores_a[m])
    for k in set(kinds):
        m = kinds_a == k
        per["by_kind"][k] = agg(cand_scores_a[m], floor_scores_a[m])

    out = {"scored_at": ts, "settled_markets_in_dump": len(settled),
           "matched_to_ledger": matched, "results": per}
    json.dump(out, open(os.path.join(SH, f"scores_{ts[:13].replace(':','')}.json"), "w"), indent=1)

    # human scoreboard
    L = [f"# Shadow scoreboard (updated {ts})",
         f"settled markets in dump: {len(settled)} | matched to ledger: {matched}",
         "Scores each candidate's Brier vs the floor's REALIZED Brier (lower = better).",
         "Negative cand_minus_floor => candidate beat the live floor.\n"]
    o = per["overall"]
    if o.get("n"):
        L.append(f"OVERALL  n={o['n']}  cand={o['cand_brier']:.4f}  floor={o['floor_brier']:.4f}  "
                 f"diff={o.get('cand_minus_floor',float('nan')):+.4f}  p_not_better={o.get('p_cand_not_better','-')}")
        L.append("\nBy tier:")
        for t, d in sorted(per["by_tier"].items()):
            if d.get("n"):
                L.append(f"  {t:5s} n={d['n']:4d} cand={d['cand_brier']:.4f} floor={d['floor_brier']:.4f} "
                         f"diff={d.get('cand_minus_floor',0):+.4f}")
        L.append("\nBy kind:")
        for k, d in sorted(per["by_kind"].items(), key=lambda kv: kv[1].get('cand_minus_floor', 0)):
            if d.get("n"):
                L.append(f"  {k:18s} n={d['n']:4d} cand={d['cand_brier']:.4f} floor={d['floor_brier']:.4f} "
                         f"diff={d.get('cand_minus_floor',0):+.4f}")
    else:
        L.append("No settled markets matched yet. Re-run after resolutions accrue:")
        L.append("  1) an agent calls list_results (read-only) -> save JSON to shadow/results_dump.json")
        L.append("  2) python scripts/shadow_score.py")
    open(os.path.join(SH, "scoreboard.md"), "w").write("\n".join(L))
    # history line
    with open(os.path.join(SH, "history.csv"), "a") as h:
        h.write(f"{ts},{matched},{o.get('cand_brier','')},{o.get('floor_brier','')},{o.get('cand_minus_floor','')}\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
