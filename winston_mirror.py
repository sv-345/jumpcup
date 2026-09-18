"""Winston-mirror: copy winston's live numbers on no-edge kinds, keep ours on
proven-edge kinds. The h2h math: copied markets score exactly 0 vs him; every
deviation is a kind where our settled record beats his (post-fix h2h, n=120,
2026-07-06) — so per-board E[h2h] > 0 with minimal variance.

    .venv/bin/python winston_mirror.py <match_id> [--live]

Without --live: prints the copy plan. With --live: writes data/overrides.json
entries (expire at lock) for the copied markets and forces one live daemon
cycle. Run at T-20 and again at T-6 (lock window: moves uncapped, catches his
late edits). Safety: never copy a number >25pp from our market-tier price on
direct-line kinds (his rare contrarian swings; the de-vigged line is truth).
"""
import json
import sqlite3
import subprocess
import sys

import crowd_stats as cs

WINSTON = "2e9d0d26-cb3f-414e-a424-472a12a55691"
ROOT = "/Users/svelaga/Documents/jumpcup"

# Kinds where OUR settled record beats his (keep our numbers — the h2h alpha),
# plus line-backed kinds where we're even (no reason to inherit dependency).
KEEP_OURS = {
    "corners_compare", "any_brace", "header_goal", "hydration_event",
    "team_advance", "first_goal_game", "ht_lead", "player_involve",
    "match_tie", "total_under", "red_card", "total_cards", "player_sot",
}
# Direct-line kinds: if his number is >25pp from ours (a line price), keep ours.
LINE_KINDS = {"win_match", "team_advance", "total_over", "total_under", "btts",
              "player_scorer", "win_margin", "match_tie", "extra_time"}
SWING_CLAMP = 0.25


def plan(match_id: str):
    dec = {d["market_id"]: d for d in
           json.load(open(f"{ROOT}/data/decisions.json"))["decisions"]}
    try:
        ov = json.load(open(f"{ROOT}/data/overrides.json"))
    except Exception:  # noqa: BLE001
        ov = {}
    wins = cs._post("match-stats", {"matchId": match_id, "lobbyId": cs.LOBBY,
                                    "targetUserId": WINSTON})
    db = sqlite3.connect(f"{ROOT}/data/state.db")
    out, keep = [], []
    for m in wins.get("markets") or []:
        mid, wp = m.get("id"), m.get("prediction")
        if mid is None or wp is None:
            continue
        d = dec.get(mid)
        kind = d["kind"] if d else "?"
        ours = d["prob"] if d else None
        row = db.execute("SELECT close_ts FROM markets WHERE market_id=? "
                         "AND close_ts > strftime('%s','now')", (mid,)).fetchone()
        if row is None:
            continue  # locked/unknown
        q = m.get("question", "")[:56]
        prev = ov.get(mid)
        if prev and not str(prev.get("note", "")).startswith("winston-mirror"):
            # conviction override (user read / manual press) — mirror must not clobber
            keep.append((kind + "[conviction]", prev.get("prob"), wp / 100.0, q))
            continue
        if kind in KEEP_OURS:
            keep.append((kind, ours, wp / 100.0, q))
            continue
        if (kind in LINE_KINDS and ours is not None
                and abs(wp / 100.0 - ours) > SWING_CLAMP):
            keep.append((kind + "[swing-clamp]", ours, wp / 100.0, q))
            continue
        out.append((mid, row[0], kind, ours, wp / 100.0, q))
    return out, keep


def main():
    match_id = sys.argv[1]
    live = "--live" in sys.argv
    copies, keeps = plan(match_id)
    print(f"COPY {len(copies)} markets from winston:")
    for _, _, kind, ours, wp, q in copies:
        print(f"  {ours if ours is not None else '?':>5} -> {wp:.2f}  {kind:<18} {q}")
    print(f"KEEP {len(keeps)} markets (our edge / clamp):")
    for kind, ours, wp, q in keeps:
        print(f"  ours={ours!s:>5} (win {wp:.2f})  {kind:<18} {q}")
    if not live:
        print("(dry run — pass --live to apply)")
        return
    try:
        ov = json.load(open(f"{ROOT}/data/overrides.json"))
    except Exception:  # noqa: BLE001
        ov = {}
    for mid, close_ts, kind, _, wp, q in copies:
        ov[mid] = {"prob": round(wp, 4), "note": f"winston-mirror: {q[:40]}",
                   "expires_ts": close_ts}
    json.dump(ov, open(f"{ROOT}/data/overrides.json", "w"), indent=1)
    print(f"wrote {len(copies)} mirror overrides; submitting live...")
    subprocess.run([f"{ROOT}/.venv/bin/python", "-m", "jumpbot.daemon", "once", "--live"],
                   timeout=240, check=False, cwd=ROOT)


if __name__ == "__main__":
    main()
