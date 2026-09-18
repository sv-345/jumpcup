#!/usr/bin/env python3
"""Shadow scorer — STEP 1 (log). Snapshot every candidate's current forecast per open
market into an append-only ledger, time-stamped, BEFORE the market resolves. This is
what makes later scoring a legitimate out-of-sample shadow test. Never submits anything.

Run this whenever you refresh prices (re-run price_open_markets.py first). The scorer
(shadow_score.py) later joins these locked forecasts to resolutions.
"""
import os, sys, json, datetime
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIV = os.path.join(HERE, "reports", "open_market_divergences.json")
LEDGER = os.path.join(HERE, "shadow", "ledger.jsonl")
os.makedirs(os.path.join(HERE, "shadow"), exist_ok=True)


def main():
    rows = json.load(open(DIV))
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    odds_ts = None
    cache = os.path.join(HERE, "..", "data", "cache", "oddsapi_wc.json")
    if os.path.exists(cache):
        odds_ts = json.load(open(cache)).get("fetched_ts")
    n = 0
    with open(LEDGER, "a") as f:
        for r in rows:
            if r.get("cand") is None:
                continue  # only log markets we actually price
            rec = {"ts": ts, "odds_ts": odds_ts, "market_id": r["market_id"],
                   "match": r["match"], "question": r["question"], "kind": r["kind"],
                   "tier": r["tier"], "candidate": r["candidate"],
                   "cand": r["cand"], "floor_replica": r["floor"],
                   "floor_actual": r["floor_actual"], "closing_time": r["closing_time"]}
            f.write(json.dumps(rec) + "\n")
            n += 1
    print(f"logged {n} candidate forecasts at {ts} -> {LEDGER}")


if __name__ == "__main__":
    main()
