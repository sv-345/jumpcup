"""Capture per-market CROWD consensus + realized RBP for settled matches.

The bot API exposes only our own Brier; the WEB match-stats endpoint also exposes
the FIELD's consensus (prediction_average) and per-market relative_brier_score. We
store both at settlement so we can measure ACTUAL per-kind RBP (crowd_brier - our_brier)
instead of tuning against our own Brier as a proxy. This is the foundation for the
novelty base-rate anchor: the crowd average == the base rate we lack at pricing time.

Fully decoupled from the live daemon (the web token is a fragile ~7-day JWT). Run
opportunistically — manually or via cron — a few hours after games settle:

    .venv/bin/python -m jumpbot.crowd_capture            # capture + print per-kind RBP
    .venv/bin/python -m jumpbot.crowd_capture --summary  # print summary only (no fetch)

Reuses crowd_stats.py (MATCHES list + web token + match_stats endpoint).
"""

from __future__ import annotations

import sys
import time
from typing import Optional

from . import state
from .logging_setup import get_logger
from .mapping import classify

log = get_logger("jumpbot.crowd_capture")


def _crowd_stats():
    """Import the repo-root crowd_stats module (on sys.path when run from repo root)."""
    try:
        import crowd_stats  # noqa: PLC0415
        return crowd_stats
    except Exception as exc:  # noqa: BLE001
        log.warning("crowd_stats import failed (%s); run from repo root", exc)
        return None


def _match_stats(cs, match_id: str, attempts: int = 4) -> Optional[dict]:
    for i in range(attempts):
        try:
            return cs.match_stats(match_id)
        except Exception as exc:  # noqa: BLE001 - network/token flakiness; retry
            log.info("match_stats %s attempt %d failed: %s", match_id[:8], i + 1, exc)
            time.sleep(2)
    return None


def _kind_for(market_id: str, question: str) -> str:
    """kind from our settled results if present (authoritative), else classify."""
    with state._conn() as c:  # noqa: SLF001 - same package
        row = c.execute("SELECT kind FROM results WHERE market_id=?", (market_id,)).fetchone()
        if row and row["kind"]:
            return row["kind"]
    try:
        return classify(question, "").kind
    except Exception:  # noqa: BLE001
        return "unknown"


def _all_matches(cs) -> list:
    """crowd_stats.MATCHES ∪ every match the daemon has registered (markets.match_id,
    stored from the bot snapshot; web id == bot id). New rounds auto-capture with
    no manual MATCHES edit."""
    out = list(cs.MATCHES)
    seen = {mid for _, mid in out}
    try:
        with state._conn() as c:  # noqa: SLF001 - same package
            rows = c.execute("SELECT DISTINCT match_id, match_name FROM markets "
                             "WHERE match_id IS NOT NULL").fetchall()
        for r in rows:
            if r["match_id"] not in seen:
                out.append(((r["match_name"] or "?").replace(" vs ", "_"), r["match_id"]))
    except Exception as exc:  # noqa: BLE001
        log.info("state match discovery failed: %s", exc)
    return out


def capture(force: bool = False) -> int:
    """Pull crowd consensus for every fully-settled match and store per-market RBP.

    Idempotent: skips a match whose markets are already captured (unless force).
    Returns the number of newly captured markets.
    """
    cs = _crowd_stats()
    if cs is None:
        return 0
    new = 0
    for tag, match_id in _all_matches(cs):
        d = _match_stats(cs, match_id)
        if not d:
            continue
        markets = d.get("markets") or []
        settled = [m for m in markets if m.get("status") == "settled"
                   and m.get("prediction_average") is not None]
        if not settled:
            continue
        if not force and state.crowd_match_captured(tag) >= len(settled):
            continue  # already have this match
        for m in settled:
            mid = m.get("id")
            if not mid:
                continue
            our = _pct(m.get("prediction"))
            crowd = _pct(m.get("prediction_average"))
            cv = m.get("current_value")
            if our is None or crowd is None or cv is None:
                continue
            outcome = 1 if cv >= 50 else 0
            q = m.get("question", "")
            kind = _kind_for(mid, q)
            ob = (our - outcome) ** 2
            cb = (crowd - outcome) ** 2
            rbp = m.get("relative_brier_score")
            if rbp is None:
                rbp = (cb - ob) * 100.0
            state.record_crowd_result(mid, tag, q, kind, our, crowd, outcome, ob, cb, rbp)
            new += 1
        log.info("crowd capture %s: %d markets (field=%.1f ours=%.1f RBP=%.1f)",
                 tag, len(settled), d.get("brier_average", 0), d.get("brier_user", 0),
                 d.get("relative_brier_score", 0))
    if new:
        try:  # structural leak probes auto-flag any watchlist leak that now has enough samples
            from . import leak_probes
            leak_probes.run_probes()
        except Exception as exc:  # noqa: BLE001
            log.info("leak_probes skipped: %s", exc)
    return new


def _pct(x) -> Optional[float]:
    """Endpoint values are 0-100; normalize to 0-1."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v / 100.0 if v > 1.0 else v


def print_summary() -> None:
    summ = state.crowd_rbp_summary()
    if not summ:
        print("no crowd results captured yet")
        return
    tot_rbp = sum(v["total_rbp"] for v in summ.values())
    n_tot = sum(v["n"] for v in summ.values())
    print(f"\nCROWD RBP by kind  (n={n_tot}, net RBP={tot_rbp:+.0f})  "
          f"[+ = we beat the field, - = field beats us]")
    print(f"{'KIND':<20}{'n':>4}{'ourBr':>8}{'crowdBr':>9}{'RBP/mkt':>9}{'totRBP':>9}"
          f"{'ours':>7}{'crowd':>7}{'out':>6}")
    print("-" * 88)
    for kind, v in summ.items():
        print(f"{kind:<20}{v['n']:>4}{v['our_brier']:>8.3f}{v['crowd_brier']:>9.3f}"
              f"{v['rbp']:>9.1f}{v['total_rbp']:>9.1f}{v['our_pred']:>7.2f}"
              f"{v['crowd_pred']:>7.2f}{v['mean_outcome']:>6.2f}")


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if "--summary" not in argv:
        n = capture(force="--force" in argv)
        print(f"captured {n} new crowd-result markets")
    print_summary()
    from . import leak_probes
    print("\nSTRUCTURAL LEAK PROBES (auto-fire when sample threshold hit):")
    for p in leak_probes.run_probes(log_flags=False):
        status = "READY" if p.get("ready") else "armed"
        print(f"  [{status:>5}] {p['kind']:<16} n={p['n']}/{p['threshold']}  {p['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
