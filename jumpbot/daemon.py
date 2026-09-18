"""The daemon: standalone loop that prices and submits, unattended.

Cycle:
  1. GET /markets (one call -> all open markets).
  2. Price every market via the engine (de-vig + goals model + base rates).
  3. Sanity-gate, then submit new forecasts (batch) / update moved ones (PATCH),
     respecting the 60 req/min limit and the bounded-move cap.
  4. Persist state for idempotent resume; log everything.
  5. Sleep — short near a kickoff (to grab closing-line value), else the base
     cadence. A STOP file or SIGTERM ends the loop cleanly.

Commands: dry-run | once | run | sync | status
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from typing import Dict, List, Optional, Tuple

from . import calibration, config, engine, retune, state, stats
from .logging_setup import get_logger
from .mapping import classify
from .odds import _iso_to_epoch
from .submission import SportsPredictClient

log = get_logger("jumpbot.daemon")

_STOP = False


def _handle_signal(signum, frame):  # noqa: ARG001
    global _STOP
    _STOP = True
    log.info("received signal %s -> stopping after this cycle", signum)


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


def _killed() -> bool:
    return _STOP or config.KILL_FILE.exists()


def _read_prob(api_val) -> Optional[float]:
    """Normalize an API probability (1-99 int or 0-1 decimal) to [0,1]."""
    if api_val is None:
        return None
    try:
        v = float(api_val)
    except (TypeError, ValueError):
        return None
    return v / 100.0 if v > 1.0 else v


# --- snapshot / state ------------------------------------------------------

def build_snapshot(client: SportsPredictClient) -> dict:
    """One GET /markets -> the engine's snapshot dict. Lock = kickoff."""
    markets = client.get_markets(config.LOBBY_ID)
    out = []
    for m in markets:
        if m.get("status") != "open":
            continue
        match = m.get("match", {}) or {}
        # Lock is the kickoff: docs say predictions must be in before match start;
        # opening_time is the kickoff, closing_time is full-time. Use opening_time.
        lock_ts = _iso_to_epoch(match.get("opening_time", "")) or \
            _iso_to_epoch(match.get("closing_time", ""))
        out.append({
            "market_id": m["id"],
            "lobby_id": m.get("lobby_id", config.LOBBY_ID),
            "question": m.get("question", ""),
            "match_name": match.get("name", ""),
            "match_id": match.get("id"),
            "closing_ts": lock_ts,  # engine uses this as the lock for the move cap
        })
    return {"fetched_ts": time.time(), "markets": out}


def sync_state(client: SportsPredictClient) -> int:
    """Pull existing predictions so resume is idempotent (know prediction ids)."""
    preds = client.get_predictions(config.LOBBY_ID)
    n = 0
    for p in preds:
        mid = p.get("market_id")
        if not mid:
            continue
        prob = _read_prob(p.get("probability"))
        # Register (question/match unknown here; engine fills them from snapshot).
        existing = state.get(mid)
        if existing is None:
            state.upsert_market(mid, p.get("lobby_id", config.LOBBY_ID),
                                p.get("question", ""), "", None)
        state.record_submission(mid, prob if prob is not None else 0.5,
                                p.get("id"), "submitted")
        n += 1
    log.info("sync_state: recovered %d existing predictions", n)
    return n


def record_results(client: SportsPredictClient) -> int:
    """Pull settled predictions, store realized Brier by market kind (self-tuning).

    Recovers the binary outcome from brier=(p-o)^2, buckets by the market kind,
    and logs a calibration summary so the model can be tuned where it's off.
    """
    try:
        results = client.get_results(config.LOBBY_ID)
    except Exception as exc:  # noqa: BLE001
        log.warning("get_results failed: %s", exc)
        return 0
    new = 0
    for r in results:
        mid = r.get("market_id")
        brier = r.get("brier_score")
        if not mid or brier is None or state.have_result(mid):
            continue
        prob = _read_prob(r.get("probability_submitted") if r.get("probability_submitted") is not None
                          else r.get("probability"))
        if prob is None:
            continue
        # Recover the 0/1 outcome: brier == (prob - outcome)^2.
        outcome = 1 if abs((1 - prob) ** 2 - brier) <= abs(prob ** 2 - brier) else 0
        st = state.get(mid)
        question = r.get("question", "") or (st.question if st else "")
        match_name = st.match_name if st else ""
        kind = classify(question, match_name).kind
        state.record_result(mid, question, kind, prob, outcome, brier)
        new += 1
    if new:
        summ = state.calibration_summary()
        tot = sum(v["n"] for v in summ.values())
        overall = sum(v["mean_brier"] * v["n"] for v in summ.values()) / tot if tot else 0.0
        log.info("calibration: +%d settled; overall realized Brier=%.4f over %d markets",
                 new, overall, tot)
        for kind, v in list(summ.items())[:10]:
            log.info("  %-17s n=%-3d brier=%.3f  pred=%.2f actual=%.2f",
                     kind, v["n"], v["mean_brier"], v["mean_pred"], v["mean_outcome"])
        calibration.update_bias()  # feed realized outcomes back into pricing
        # Best-effort: capture the field's consensus + per-kind RBP for newly-settled
        # matches (needs the web JWT; fully guarded so token expiry never breaks settle).
        try:
            from . import crowd_capture
            got = crowd_capture.capture()
            if got:
                log.info("crowd RBP capture: +%d markets", got)
        except Exception as exc:  # noqa: BLE001
            log.info("crowd capture skipped: %s", exc)
    return new


# --- sanity gate -----------------------------------------------------------

def sanity_check(decisions: List[engine.Decision]) -> Tuple[bool, List[str]]:
    """Block submitting if the *pricing* is unhealthy.

    Evaluated over all *priced* decisions (anything not skipped, including holds)
    — not over the actionable subset. Zero actionable just means nothing moved
    this tick (the steady state), which is fine; a broken pass is one where we
    couldn't price the board or the odds feed vanished.
    """
    reasons: List[str] = []
    if not decisions:
        return False, ["no decisions produced (could not fetch markets?)"]
    priced = [d for d in decisions if d.action != "skip"]
    skipped = [d for d in decisions if d.action == "skip"]
    if not priced:
        return False, ["no priced decisions (mapping/pricing collapsed?)"]
    # Every priced probability must be valid and in band.
    for d in priced:
        if d.prob != d.prob or not (config.PROB_FLOOR <= d.prob <= config.PROB_CEIL) \
                or not (1 <= d.prob_int <= 99):
            reasons.append(f"out-of-band/NaN prob for {d.market_id}: {d.prob}")
            break
    # Mapping shouldn't collapse: most markets should be classifiable.
    if len(skipped) / len(decisions) > 0.5:
        reasons.append(f"high skip rate {len(skipped)}/{len(decisions)}")
    # At least one sharp market-tier price should exist (odds feed reachable).
    if not any(d.tier == "market" for d in priced):
        reasons.append("no market-tier (sharp odds) prices — odds feed may be down")
    return (len(reasons) == 0), reasons


# --- submission ------------------------------------------------------------

def submit_decisions(client: SportsPredictClient,
                     decisions: List[engine.Decision]) -> dict:
    """Submit new forecasts (batch) and update moved ones (PATCH)."""
    to_submit = [d for d in decisions if d.action == "submit"]
    to_update = [d for d in decisions if d.action == "update"]
    responses = {"ts": time.time(), "submitted": [], "updated": [], "errors": []}

    # New forecasts -> batch POST (chunks of 50).
    for i in range(0, len(to_submit), 50):
        chunk = to_submit[i:i + 50]
        payload = [{"market_id": d.market_id, "lobby_id": d.lobby_id,
                    "probability": d.prob_int} for d in chunk]
        try:
            res = client.submit_batch(payload)
        except Exception as exc:  # noqa: BLE001
            log.warning("batch submit failed: %s", exc)
            responses["errors"].append({"stage": "batch", "error": str(exc)})
            continue
        for d in chunk:
            ok, pred_id, err = res.per_market.get(d.market_id, (False, None, "missing"))
            if ok:
                state.record_submission(d.market_id, d.prob, pred_id, "submitted")
                responses["submitted"].append({"market_id": d.market_id, "p": d.prob_int})
            elif err == "already_predicted":
                # Recover id and PATCH instead.
                _patch_fallback(client, d, responses)
            else:
                state.mark_failed(d.market_id)
                responses["errors"].append({"market_id": d.market_id, "error": err})
        log.info("batch %d: %d/%d ok", i // 50 + 1, res.succeeded, res.total)

    # Moves on existing predictions -> PATCH.
    for d in to_update:
        st = state.get(d.market_id)
        pred_id = (st.prediction_id if st else None) or d.prediction_id
        if not pred_id:
            # No id known: try a POST; if 409, recover id and PATCH.
            ok, pid, err = client.submit_one(d.market_id, d.lobby_id, d.prob_int)
            if ok:
                state.record_submission(d.market_id, d.prob, pid, "submitted")
                responses["submitted"].append({"market_id": d.market_id, "p": d.prob_int})
                continue
            _patch_fallback(client, d, responses)
            continue
        ok, err = client.patch_prediction(pred_id, d.prob_int)
        if ok:
            state.record_submission(d.market_id, d.prob, pred_id, "updated")
            responses["updated"].append({"market_id": d.market_id, "p": d.prob_int})
        else:
            responses["errors"].append({"market_id": d.market_id, "error": err})

    with open(config.SUBMIT_RESPONSES, "w") as f:
        json.dump(responses, f, indent=2)
    return responses


def _patch_fallback(client: SportsPredictClient, d: engine.Decision, responses: dict) -> None:
    """Recover a prediction id from the API and PATCH it."""
    st = state.get(d.market_id)
    pred_id = st.prediction_id if st else None
    if not pred_id:
        for p in client.get_predictions(config.LOBBY_ID):
            if p.get("market_id") == d.market_id:
                pred_id = p.get("id")
                break
    if not pred_id:
        responses["errors"].append({"market_id": d.market_id, "error": "no_pred_id"})
        return
    ok, err = client.patch_prediction(pred_id, d.prob_int)
    if ok:
        state.record_submission(d.market_id, d.prob, pred_id, "updated")
        responses["updated"].append({"market_id": d.market_id, "p": d.prob_int})
    else:
        responses["errors"].append({"market_id": d.market_id, "error": err})


# --- cycle / loop ----------------------------------------------------------

def _force_odds_now(snapshot: dict) -> bool:
    """True if any open market locks within the lineup window (refresh for CLV)."""
    now = time.time()
    for m in snapshot.get("markets", []):
        lock = m.get("closing_ts")
        if lock and 0 < (lock - now) <= config.LINEUP_WINDOW_MIN * 60:
            return True
    return False


def run_cycle(client: SportsPredictClient, live: bool) -> List[engine.Decision]:
    snapshot = build_snapshot(client)
    force = _force_odds_now(snapshot)
    decisions = engine.build_decisions(snapshot, force_odds=force)

    table = engine.render_table(decisions)
    with open(config.DRY_RUN_TABLE, "w") as f:
        f.write(table + "\n")
    with open(config.DECISIONS_FILE, "w") as f:
        json.dump(engine.decisions_to_json(decisions), f, indent=2)

    ok, reasons = sanity_check(decisions)
    if not ok:
        log.warning("SANITY GATE failed: %s", "; ".join(reasons))
        if live:
            log.warning("staying in DRY-RUN this cycle (not submitting)")
        return decisions

    if live and not _killed():
        resp = submit_decisions(client, decisions)
        log.info("cycle submit: %d new, %d updated, %d errors",
                 len(resp["submitted"]), len(resp["updated"]), len(resp["errors"]))
    return decisions


def _next_sleep(decisions: List[engine.Decision]) -> float:
    """Short cadence near a kickoff (CLV), base cadence otherwise."""
    now = time.time()
    locks = [st.close_ts for st in state.all_states().values()
             if st.close_ts and st.close_ts > now]
    base = config.POLL_INTERVAL_MIN * 60
    if not locks:
        return base
    ttl = min(locks) - now
    if ttl <= config.LINEUP_WINDOW_MIN * 60:
        # Aim to fire ~3 min before the next lock, but never busy-spin.
        return max(60.0, min(ttl - 180.0, base))
    return base


_last_player_accum_ts = 0.0
_PLAYER_ACCUM_REFRESH_S = 12 * 3600


def _maybe_refresh_player_rates() -> None:
    """Fold any newly-finished WC fixtures into the per-player SoT rate
    accumulator. Idempotent (processed_fixtures), so it's a single cheap
    /fixtures call when nothing new has settled; keeps per-player SoT rates fresh
    through the tournament without a manual re-backfill. (Team rates already
    refresh via stats.update_team_accumulator inside prefetch.) Runs on the first
    cycle after start, then every 12h. Never raises into the loop."""
    global _last_player_accum_ts
    now = time.time()
    if now - _last_player_accum_ts < _PLAYER_ACCUM_REFRESH_S:
        return
    _last_player_accum_ts = now
    try:
        stats.backfill_player_accumulator()
    except Exception as exc:  # noqa: BLE001
        log.warning("player accumulator refresh failed: %s", exc)


_last_retune_ts = 0.0
_RETUNE_REFRESH_S = 2 * 24 * 3600  # re-validate optimal shrink every ~2 days


def _maybe_retune() -> None:
    """Re-validate the LOO-optimal per-kind shrink and loosen it only as growing
    settled data justifies (the shrink auto-tuner; see retune.py). Runs on the first
    cycle after start, then every ~2 days. Never raises into the loop."""
    global _last_retune_ts
    now = time.time()
    if now - _last_retune_ts < _RETUNE_REFRESH_S:
        return
    _last_retune_ts = now
    try:
        retune.retune()
    except Exception as exc:  # noqa: BLE001
        log.warning("retune failed: %s", exc)


_known_finished: Optional[set] = None


def _maybe_dynamic_refresh() -> None:
    """Event-driven calibration: diff the finished-fixture set each cycle and, the
    moment a WC game ends, fold its result into every API-Football-fed input (team &
    player accumulators, compare outcomes) and re-tune — so the next game (often
    concurrent, or only hours away) prices on fresh data instead of waiting on the
    6h/12h/2-day timers. (The bias self-tuner already runs every cycle.) Idempotent;
    never raises into the loop."""
    global _known_finished
    try:
        finished = stats.finished_fixture_ids()
    except Exception as exc:  # noqa: BLE001
        log.warning("settlement check failed: %s", exc)
        return
    if not finished:
        return
    if _known_finished is None:
        _known_finished = finished  # baseline on the first cycle; no refresh storm
        return
    new = finished - _known_finished
    if not new:
        return
    _known_finished = finished
    log.info("settlement: %d WC game(s) just finished -> folding calibration", len(new))
    for fn, label in ((stats.backfill_team_accumulator, "team-stats"),
                      (stats.backfill_player_accumulator, "player-rates"),
                      (retune.retune, "shrink-retune")):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            log.warning("dynamic refresh (%s) failed: %s", label, exc)


def loop(client: SportsPredictClient, live: bool) -> None:
    log.info("daemon loop starting (live=%s, poll=%dm)", live, config.POLL_INTERVAL_MIN)
    state.log_event("loop_start", detail=f"live={live}")
    cycle = 0
    while not _killed():
        cycle += 1
        _maybe_refresh_player_rates()
        _maybe_retune()
        _maybe_dynamic_refresh()
        t0 = time.time()
        try:
            decisions = run_cycle(client, live)
        except Exception as exc:  # noqa: BLE001 - never let one cycle kill the loop
            log.exception("cycle %d crashed: %s", cycle, exc)
            decisions = []
        try:
            record_results(client)  # self-tuning: track realized Brier by kind
        except Exception as exc:  # noqa: BLE001
            log.warning("record_results failed: %s", exc)
        sleep_s = _next_sleep(decisions)
        log.info("cycle %d done in %.1fs; sleeping %.0fs", cycle, time.time() - t0, sleep_s)
        # Sleep in short slices so STOP is honored promptly.
        slept = 0.0
        while slept < sleep_s and not _killed():
            time.sleep(min(5.0, sleep_s - slept))
            slept += 5.0
    log.info("daemon loop stopped (kill=%s)", config.KILL_FILE.exists())
    state.log_event("loop_stop")


# --- CLI -------------------------------------------------------------------

def _print_status(client: SportsPredictClient) -> None:
    preds = client.get_predictions(config.LOBBY_ID)
    results = client.get_results(config.LOBBY_ID)
    print(f"predictions on record: {len(preds)} | settled: {len(results)}")
    states = state.all_states()
    by_status: Dict[str, int] = {}
    for s in states.values():
        by_status[s.status] = by_status.get(s.status, 0) + 1
    print(f"local state: {len(states)} markets | {by_status}")
    if results:
        briers = [r.get("brier_score") for r in results if r.get("brier_score") is not None]
        if briers:
            print(f"settled Brier: mean={sum(briers)/len(briers):.4f} over {len(briers)}")
    record_results(client)
    summ = state.calibration_summary()
    if summ:
        print("calibration by market kind (realized Brier | predicted vs actual):")
        for kind, v in summ.items():
            print(f"  {kind:18s} n={v['n']:<3d} brier={v['mean_brier']:.3f}  "
                  f"pred={v['mean_pred']:.2f} actual={v['mean_outcome']:.2f}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="jumpbot.daemon")
    ap.add_argument("command", choices=["dry-run", "once", "run", "sync", "status"])
    ap.add_argument("--live", action="store_true", help="actually submit (for 'once')")
    args = ap.parse_args(argv)

    client = SportsPredictClient()
    client.join_lobby(config.LOBBY_ID)  # idempotent

    if args.command == "sync":
        sync_state(client)
        return 0
    if args.command == "status":
        _print_status(client)
        return 0
    if args.command == "dry-run":
        decisions = run_cycle(client, live=False)
        print(engine.render_table(decisions))
        return 0
    if args.command == "once":
        sync_state(client)
        decisions = run_cycle(client, live=args.live)
        print(engine.render_table(decisions))
        return 0
    if args.command == "run":
        sync_state(client)
        loop(client, live=True)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
