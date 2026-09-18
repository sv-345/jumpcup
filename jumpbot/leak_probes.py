"""Structural leak probes — auto-flag watchlist leaks once enough RBP samples land.

Base-calibration leaks are caught by the BASE-vs-realized audit (corners/cards fixes).
What's LEFT can't be fixed by a base tweak — it needs a structural test on accumulated
crowd RBP. Each probe has a sample threshold and a PRE-DEFINED test; when the threshold
is hit it logs a VERDICT (never auto-patches — the structural call stays human). Runs
after each crowd capture; also `python -m jumpbot.leak_probes` for the current read.
"""

from __future__ import annotations

from typing import Dict, List

from . import state
from .logging_setup import get_logger

log = get_logger("jumpbot.leak_probes")


def _rows() -> List[dict]:
    with state._conn() as c:  # noqa: SLF001 - same package
        return [dict(r) for r in c.execute(
            "SELECT kind,match_tag,our_prob,crowd_prob,outcome,market_rbp FROM crowd_results")]


def _favorite_tilt_probe(rows: List[dict], winp: Dict[str, float], kind: str, N: int = 8) -> Dict:
    """Do we UNDER-credit dominant/attacking teams' activity (corners, offsides)? Both share
    one root cause: favorites attack more -> more corners AND offsides. Test: calibration gap
    + direction + whether it's WORSE in lopsided games (= the favorite-tilt is too weak). Bases
    are already realized-correct, so a persistent gap means the per-team tilt needs strengthening."""
    tc = [r for r in rows if r["kind"] == kind]
    out = {"kind": kind, "n": len(tc), "threshold": N, "ready": len(tc) >= N}
    if len(tc) < N:
        out["verdict"] = f"waiting ({len(tc)}/{N} samples)"
        return out
    gap = sum(r["outcome"] - r["our_prob"] for r in tc) / len(tc)   # >0 = we under-predict
    under = sum(1 for r in tc if r["our_prob"] < r["crowd_prob"]) / len(tc)
    lop = [r for r in tc if r["match_tag"] in winp and abs(winp[r["match_tag"]] - 0.5) >= 0.18]
    lop_gap = (sum(r["outcome"] - r["our_prob"] for r in lop) / len(lop)) if lop else None
    metric = "corner" if "corner" in kind else "offsides"
    if gap > 0.05 and under > 0.6:
        v = f"UNDER-PREDICT (gap +{gap:.2f}, {under:.0%} below crowd)"
        if lop_gap is not None and lop_gap > gap + 0.03:
            v += f"; WORSE in lopsided games ({lop_gap:+.2f}) -> PATCH: strengthen favorite {metric}-tilt"
        else:
            v += f" -> PATCH: nudge {kind} opponent-adjustment up"
    elif gap < -0.05:
        v = f"OVER-PREDICT (gap {gap:.2f}) -> monitor"
    else:
        v = f"no consistent bias (gap {gap:+.2f}) -> variance, no patch"
    out.update(verdict=v, gap=round(gap, 3), under_frac=round(under, 2))
    return out


def _fouls_compare_probe(rows: List[dict]) -> Dict:
    """fouls_compare loses to the crowd, but is it because we're TOO TIMID (over-shrunk ->
    reduce shrink) or WRONG-DIRECTION (deeper issue -> don't touch)? Test: when we and the
    crowd land on opposite sides of 0.5, whose side is right? Scarred kind -> high bar."""
    fc = [r for r in rows if r["kind"] == "fouls_compare"]
    N = 10
    out = {"kind": "fouls_compare", "n": len(fc), "threshold": N, "ready": len(fc) >= N}
    if len(fc) < N:
        out["verdict"] = f"waiting ({len(fc)}/{N} samples; scarred -> high bar)"
        return out
    disagree = [r for r in fc if (r["our_prob"] - 0.5) * (r["crowd_prob"] - 0.5) < 0]
    agree = [r for r in fc if (r["our_prob"] - 0.5) * (r["crowd_prob"] - 0.5) >= 0]

    def crowd_closer(rs):
        if not rs:
            return None
        return sum(1 for r in rs if abs(r["crowd_prob"] - r["outcome"]) < abs(r["our_prob"] - r["outcome"])) / len(rs)

    dis_cc = crowd_closer(disagree)
    if dis_cc is not None and dis_cc > 0.6:
        v = (f"WRONG-DIRECTION (crowd's side wins {dis_cc:.0%} of {len(disagree)} disagreements) "
             f"-> do NOT unshrink; deeper foul-model issue")
    else:
        ag_cc = crowd_closer(agree)
        if ag_cc is not None and ag_cc > 0.6:
            v = (f"TOO-TIMID (same direction, crowd sharper {ag_cc:.0%}) "
                 f"-> PATCH candidate: reduce fouls_compare shrink")
        else:
            v = "mixed / no clear cause -> hold (keep shrink)"
    out["verdict"] = v
    return out


def run_probes(log_flags: bool = True) -> List[Dict]:
    """Run all structural probes on the accumulated crowd RBP. Logs a verdict for any probe
    whose sample threshold is now met. Guarded — never raises into the caller."""
    try:
        rows = _rows()
    except Exception as exc:  # noqa: BLE001
        log.info("leak_probes skipped: %s", exc)
        return []
    winp = {r["match_tag"]: r["our_prob"] for r in rows if r["kind"] == "win_match"}
    probes = [_favorite_tilt_probe(rows, winp, "team_corners"),
              _favorite_tilt_probe(rows, winp, "team_offsides"),
              _fouls_compare_probe(rows)]
    if log_flags:
        for p in probes:
            if p.get("ready"):
                log.info("LEAK PROBE FIRED — %s: %s", p["kind"], p["verdict"])
    return probes


def main() -> int:
    for p in run_probes(log_flags=False):
        status = "READY" if p.get("ready") else "armed"
        print(f"[{status:>5}] {p['kind']:<16} n={p['n']}/{p['threshold']}  {p['verdict']}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
