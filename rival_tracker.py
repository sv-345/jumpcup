"""Track our per-match RBP against named rivals + live leaderboard position.

For every match in crowd_stats.MATCHES, pulls web match-stats for us and each
rival (targetUserId works for ANY user), plus the full-field leaderboard, and
appends a snapshot to data/rival_log.json. Prints the gap report.

    .venv/bin/python rival_tracker.py            # fetch + report
    .venv/bin/python rival_tracker.py --report   # report from last snapshot only

Run after each lock/settlement (or via cron). Needs a live web token
(sp_login.py to refresh). Built 2026-07-03 for the beat-winston campaign.
"""
import json
import os
import sys
import time
import urllib.request

import crowd_stats as cs

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(ROOT, "data", "rival_log.json")
EVENT = "aa5572ec-5930-4d99-b06b-f8966333d172"
US = cs.US
RIVALS = {
    "winston weng": "2e9d0d26-cb3f-414e-a424-472a12a55691",
    "dtz2104": "02928b5b-f399-4a56-bc58-58f1910d4e2c",
    "Jholm": "e1344284-aec9-4682-b471-1f06ac7ded2b",
}


def leaderboard():
    req = urllib.request.Request(
        "https://api.sportspredict.com/api/probability/leaderboard",
        data=json.dumps({"lobbyId": cs.LOBBY, "eventId": EVENT,
                         "type": "all", "sortDirection": "desc"}).encode(),
        method="POST",
        headers={"authorization": "Bearer " + cs._bearer(), "cookie": "jwt=" + cs._cookie(),
                 "content-type": "application/json", "accept": "application/json",
                 "origin": "https://play.sportspredict.com",
                 "referer": "https://play.sportspredict.com/"})
    return json.load(urllib.request.urlopen(req, timeout=30))["items"]


def snapshot():
    snap = {"ts": time.time(), "matches": {}, "standings": {}}
    for name, mid in cs.MATCHES:
        row = {}
        for label, uid in [("us", US)] + list(RIVALS.items()):
            try:
                d = cs.match_stats(mid) if uid == US else cs._post(
                    "match-stats", {"matchId": mid, "lobbyId": cs.LOBBY, "targetUserId": uid})
            except Exception:
                continue
            if d.get("relative_brier_score") is not None:
                row[label] = round(d["relative_brier_score"], 1)
        if row:
            snap["matches"][name] = row
    try:
        items = leaderboard()
        wanted = {US: "us", **{v: k for k, v in RIVALS.items()}}
        for x in items:
            if x["user_id"] in wanted:
                snap["standings"][wanted[x["user_id"]]] = {
                    "rank": x["rank"], "score": round(x["score"], 1)}
        for bar in (10, 50, 100):
            if len(items) >= bar:
                snap["standings"][f"top{bar}_bar"] = round(items[bar - 1]["score"], 1)
    except Exception as exc:  # noqa: BLE001 - leaderboard is best-effort
        print(f"leaderboard fetch failed: {exc}")
    hist = []
    if os.path.exists(LOG):
        hist = json.load(open(LOG))
    hist.append(snap)
    json.dump(hist, open(LOG, "w"), indent=0)
    return snap


def report(snap):
    st = snap.get("standings", {})
    us, w = st.get("us"), st.get("winston weng")
    if us and w:
        print(f"us: rank {us['rank']} @ {us['score']}  |  winston: rank {w['rank']} "
              f"@ {w['score']}  |  gap {w['score'] - us['score']:+.1f}")
    for bar in (100, 50, 10):
        if f"top{bar}_bar" in st and us:
            print(f"  top-{bar} bar: {st[f'top{bar}_bar']}  ({st[f'top{bar}_bar'] - us['score']:+.1f} from us)")
    print(f"\n{'match':<12}{'us':>8}{'winston':>9}{'edge':>7}   (per-match RBP)")
    tot_us = tot_w = 0
    for name, row in snap.get("matches", {}).items():
        u, wn = row.get("us"), row.get("winston weng")
        if u is None or wn is None:
            continue
        tot_us += u
        tot_w += wn
        print(f"{name:<12}{u:>8.1f}{wn:>9.1f}{u - wn:>+7.1f}")
    print(f"{'TOTAL':<12}{tot_us:>8.1f}{tot_w:>9.1f}{tot_us - tot_w:>+7.1f}")


if __name__ == "__main__":
    if "--report" in sys.argv and os.path.exists(LOG):
        report(json.load(open(LOG))[-1])
    else:
        report(snapshot())
