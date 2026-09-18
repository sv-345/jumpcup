"""Periodic shrink auto-tuner.

The static per-tier ``CONFIDENCE_SHRINK`` pulls model probabilities toward 0.5 to
hedge model error. Early in the tournament (~2 games/team) that hedge is *correct*:
leave-one-out Brier shows extra extremity overfits. As settled games accumulate the
optimal shrink loosens (we can justifiably be more extreme). This module re-validates
the LOO-optimal shrink for the validated *compare* kinds every couple of days and
nudges a per-kind override toward it — guardrailed so it can never overfit-jump:

  * only kinds in ``_KIND_METRIC`` (validated) are eligible,
  * a kind needs ``_MIN_SAMPLES`` settled comparisons before it's touched,
  * the factor moves at most ``_STEP`` per run and is clamped to [``_FLOOR``,``_CEIL``],
  * a move must improve LOO Brier by at least ``_EPS`` (no churn on noise).

Complements calibration.py: that tunes directional *bias*; this tunes how *extreme*
we're allowed to be. Auto-applied and logged; persisted so it survives restarts.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict

from . import config, stats, model

log = logging.getLogger("jumpbot.retune")

_CACHE_FILE = config.DATA_DIR / "compare_outcomes.json"
# validated compare kinds -> the per-team metric they compare (all base_rate tier)
_KIND_METRIC = {"corners_compare": "corners", "fouls_compare": "fouls",
                "cards_compare": "cards"}
_MIN_SAMPLES = 40            # settled comparisons needed before tuning a kind
_STEP = 0.05                 # max move of the shrink factor per run
_FLOOR, _CEIL = 0.85, 1.20   # clamp on the auto-tuned factor
_EPS = 0.0015                # min LOO-Brier gain required to move (else hold)
_EB = 3.0                    # EB prior weight (matches stats._EB_STRENGTH)
_FACTORS = [round(0.80 + 0.05 * i, 2) for i in range(11)]  # 0.80 .. 1.30


def _load_cache() -> dict:
    try:
        with open(_CACHE_FILE) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict) -> None:
    tmp = _CACHE_FILE.with_suffix(".tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(cache, f)
        tmp.replace(_CACHE_FILE)
    except OSError as exc:  # noqa: BLE001
        log.warning("retune cache save failed: %s", exc)


def _refresh_outcomes() -> dict:
    """Append any newly-finished WC fixtures' both-team compare counts. Idempotent
    via fixture id, so it's one cheap sweep when nothing new has settled."""
    cache = _load_cache()
    data = stats._api("/fixtures", {"league": stats._WC_LEAGUE, "season": 2026})
    added = 0
    for f in (data or {}).get("response", []):
        if f["fixture"]["status"]["short"] not in ("FT", "AET", "PEN"):
            continue
        fid = str(f["fixture"]["id"])
        if fid in cache:
            continue
        st = stats._api("/fixtures/statistics", {"fixture": f["fixture"]["id"]})
        if not st or not st.get("response"):
            continue
        rows = []
        for tb in st["response"]:
            code = stats.resolve_team(tb["team"]["name"])
            if not code:
                continue
            s = {x["type"]: x["value"] for x in tb["statistics"]}
            rows.append({"code": code, "corners": s.get("Corner Kicks") or 0,
                         "fouls": s.get("Fouls") or 0,
                         "cards": (s.get("Yellow Cards") or 0) + (s.get("Red Cards") or 0)})
        if len(rows) == 2:
            cache[fid] = {"a": rows[0], "b": rows[1]}
            added += 1
    if added:
        _save_cache(cache)
        log.info("retune: folded %d newly-settled fixtures (%d total)", added, len(cache))
    return cache


def _loo_brier_by_factor(cache: dict, metric: str):
    """Leave-one-out Brier of the opponent-adjusted compare at each shrink factor.
    Returns ({factor: brier}, n) or (None, 0)."""
    tg = defaultdict(list)
    num = den = 0
    for fid, fx in cache.items():
        a, b = fx["a"], fx["b"]
        tg[a["code"]].append((fid, a, b))
        tg[b["code"]].append((fid, b, a))
        num += a[metric] + b[metric]
        den += 2
    if den == 0:
        return None, 0
    league = num / den

    def rate(gl):
        g = len(gl)
        if g == 0:
            return league, league
        own = sum(x[1][metric] for x in gl) / g
        agn = sum(x[2][metric] for x in gl) / g
        return (own * g + league * _EB) / (g + _EB), (agn * g + league * _EB) / (g + _EB)

    def shrink(p, f):
        return min(0.98, max(0.02, 0.5 + f * (p - 0.5)))

    briers = {f: 0.0 for f in _FACTORS}
    n = 0
    for fid, fx in cache.items():
        a, b = fx["a"], fx["b"]
        if a[metric] == b[metric]:
            continue
        y = 1.0 if a[metric] > b[metric] else 0.0
        n += 1
        rAf, rAa = rate([x for x in tg[a["code"]] if x[0] != fid])
        rBf, rBa = rate([x for x in tg[b["code"]] if x[0] != fid])
        p = model.prob_a_more_than_b(rAf * (rBa / league), rBf * (rAa / league))
        for f in _FACTORS:
            briers[f] += (shrink(p, f) - y) ** 2
    if n == 0:
        return None, 0
    return {f: briers[f] / n for f in _FACTORS}, n


def retune() -> None:
    """Refresh settled data, re-validate optimal shrink per validated kind, and nudge
    the per-kind override toward it within guardrails. Auto-applied + logged. Never
    raises into the daemon loop."""
    try:
        cache = _refresh_outcomes()
    except Exception as exc:  # noqa: BLE001
        log.warning("retune: outcome refresh failed: %s", exc)
        return
    changed = 0
    for kind, metric in _KIND_METRIC.items():
        try:
            res, n = _loo_brier_by_factor(cache, metric)
        except Exception as exc:  # noqa: BLE001
            log.warning("retune %s: LOO failed: %s", kind, exc)
            continue
        if not res or n < _MIN_SAMPLES:
            log.info("retune %s: hold (n=%d < %d)", kind, n or 0, _MIN_SAMPLES)
            continue
        opt = min(res, key=res.get)
        cur = config.shrink_factor(kind, "base_rate")
        if opt > cur:
            new = min(round(cur + _STEP, 3), opt)
        elif opt < cur:
            new = max(round(cur - _STEP, 3), opt)
        else:
            new = cur
        new = round(max(_FLOOR, min(_CEIL, new)), 3)

        def b_at(f):  # nearest sampled factor's LOO Brier
            return res[min(res, key=lambda x: abs(x - f))]

        gain = b_at(cur) - b_at(new)
        if abs(new - cur) < 1e-9 or gain < _EPS:
            log.info("retune %s: hold (LOO-opt=%.2f cur=%.3f n=%d gain=%.4f)",
                     kind, opt, cur, n, gain)
            continue
        config.SHRINK_OVERRIDES[kind] = new
        changed += 1
        log.info("retune %s: shrink %.3f -> %.3f (LOO-opt=%.2f n=%d gain=%.4f)",
                 kind, cur, new, opt, n, gain)
    if changed:
        config.save_shrink_overrides()
        log.info("retune: applied %d shrink change(s); overrides=%s",
                 changed, dict(config.SHRINK_OVERRIDES))
