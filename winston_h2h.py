"""Head-to-head capture: our vs winston weng per-market predictions on settled KO games.

For every settled KO match (15-market boards), pulls web match-stats for us and
winston, joins to state.db results for kind+outcome, writes data/winston_h2h.json
and prints per-kind + per-game aggregates.

    .venv/bin/python winston_h2h.py            # fetch + analyze
    .venv/bin/python winston_h2h.py --report   # analyze from cached file
"""
import json
import os
import sqlite3
import sys
import time

import crowd_stats as cs

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data", "winston_h2h.json")
WINSTON = "2e9d0d26-cb3f-414e-a424-472a12a55691"


def settled_ko_matches():
    """(tag, match_id) for settled KO boards, from crowd_stats.MATCHES tags."""
    return [(t, m) for t, m in cs.MATCHES if t.endswith("_KO")]


def fetch():
    db = sqlite3.connect(os.path.join(ROOT, "data", "state.db"))
    db.row_factory = sqlite3.Row
    kinds = {r["market_id"]: (r["kind"], r["outcome"]) for r in
             db.execute("SELECT market_id, kind, outcome FROM results")}
    rows = []
    for tag, mid in settled_ko_matches():
        try:
            ours = cs.match_stats(mid)
            time.sleep(0.6)
            wins = cs._post("match-stats", {"matchId": mid, "lobbyId": cs.LOBBY,
                                            "targetUserId": WINSTON})
            time.sleep(0.6)
        except Exception as exc:  # noqa: BLE001
            print(f"{tag}: fetch failed ({exc})")
            continue
        if not (ours.get("markets") and wins.get("markets")):
            print(f"{tag}: no markets (unsettled?)")
            continue
        wmap = {m.get("id") or m.get("question"): m for m in wins["markets"]}
        n = 0
        for m in ours["markets"]:
            if m.get("status") != "settled":
                continue  # pre-lock current_value is not an outcome
            key = m.get("id") or m.get("question")
            w = wmap.get(key)
            if not w or m.get("prediction") is None or w.get("prediction") is None:
                continue
            k, o = kinds.get(m.get("id"), ("?", None))
            if o is None and m.get("current_value") is not None:
                o = 1 if m["current_value"] >= 50 else 0
            if o is None:
                continue
            rows.append({
                "tag": tag, "market_id": m.get("id"), "q": m.get("question"),
                "kind": k, "outcome": o,
                "our": m["prediction"], "win": w["prediction"],
                "field": round(m.get("prediction_average") or 0, 1),
                "our_mrbp": m.get("relative_brier_score"),
                "win_mrbp": w.get("relative_brier_score"),
            })
            n += 1
        print(f"{tag:<12} {n} joined markets  (our RBP {ours.get('relative_brier_score')!s:>7}"
              f"  win {wins.get('relative_brier_score')!s:>7})")
    json.dump(rows, open(OUT, "w"), indent=0)
    print(f"\n{len(rows)} rows -> {OUT}")
    return rows


def brier(p, o):
    return (p / 100 - o) ** 2


def report(rows):
    per_kind = {}
    for r in rows:
        d = per_kind.setdefault(r["kind"], {"n": 0, "our": 0.0, "win": 0.0, "field": 0.0,
                                            "our_dev": 0.0, "win_dev": 0.0})
        d["n"] += 1
        d["our"] += brier(r["our"], r["outcome"])
        d["win"] += brier(r["win"], r["outcome"])
        d["field"] += brier(r["field"], r["outcome"])
        d["our_dev"] += abs(r["our"] - r["field"])
        d["win_dev"] += abs(r["win"] - r["field"])
    tot = {"n": 0, "our": 0.0, "win": 0.0, "field": 0.0}
    print(f"\n{'kind':<24}{'n':>4}{'ours':>8}{'winston':>9}{'field':>8}"
          f"{'w-edge':>8}{'|dev| us/win':>14}")
    for k, d in sorted(per_kind.items(), key=lambda kv: (kv[1]["our"] - kv[1]["win"]) / kv[1]["n"],
                       reverse=True):
        n = d["n"]
        tot["n"] += n
        for f in ("our", "win", "field"):
            tot[f] += d[f]
        print(f"{k:<24}{n:>4}{d['our']/n:>8.3f}{d['win']/n:>9.3f}{d['field']/n:>8.3f}"
              f"{(d['our']-d['win'])/n*100:>+8.1f}{d['our_dev']/n:>7.1f}/{d['win_dev']/n:<5.1f}")
    n = tot["n"]
    print(f"{'TOTAL':<24}{n:>4}{tot['our']/n:>8.3f}{tot['win']/n:>9.3f}{tot['field']/n:>8.3f}"
          f"{(tot['our']-tot['win'])/n*100:>+8.1f}")
    print("\nBIGGEST per-market losses to winston (our_brier - win_brier):")
    worst = sorted(rows, key=lambda r: brier(r["win"], r["outcome"]) - brier(r["our"], r["outcome"]))[:15]
    for r in worst:
        gap = (brier(r["our"], r["outcome"]) - brier(r["win"], r["outcome"])) * 100
        print(f"{r['tag']:<12}{r['kind']:<20} us={r['our']:>3} win={r['win']:>3} "
              f"field={r['field']:>5} out={r['outcome']} loss={gap:+.1f}  {r['q'][:55]}")
    print("\nBIGGEST per-market wins over winston:")
    for r in sorted(rows, key=lambda r: brier(r["win"], r["outcome"]) - brier(r["our"], r["outcome"]))[-15:][::-1]:
        gap = (brier(r["our"], r["outcome"]) - brier(r["win"], r["outcome"])) * 100
        print(f"{r['tag']:<12}{r['kind']:<20} us={r['our']:>3} win={r['win']:>3} "
              f"field={r['field']:>5} out={r['outcome']} loss={gap:+.1f}  {r['q'][:55]}")


if __name__ == "__main__":
    if "--report" in sys.argv and os.path.exists(OUT):
        report(json.load(open(OUT)))
    else:
        report(fetch())
