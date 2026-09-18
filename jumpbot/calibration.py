"""Self-tuning recalibration from settled results — two tiers.

* group bias (coarse, kicks in early at n>=GROUP_MIN_N): aggregates kinds into
  "anchored" (sharp/goal-derived markets) vs "prop" (base-rate count markets) and
  corrects each group's systematic over/under-prediction. This lets tuning start
  with one slate of data while per-kind samples are still tiny.
* per-kind bias (fine, n>=MIN_N): overrides the group bias once a single market
  kind has enough settled samples to calibrate on its own.

Every correction = (realized frequency - mean predicted), shrunk by sample size
and capped, so it can never swing a price wildly and self-corrects as data grows.
"""

from __future__ import annotations

import json
from typing import Dict

from . import config, state
from .logging_setup import get_logger

log = get_logger("jumpbot.calibration")

_FILE = config.DATA_DIR / "calibration_bias.json"
# 2026-07-02 (user: "press in EITHER direction"): bidirectional toward-truth audit
# found large residual biases confirmed by BOTH datasets (crowd + bot grading):
# total_offsides +.25/+.36 UNDER-priced, total_corners -.25/-.38 OVER, total_over
# +.22/+.20 UNDER (this WC scores hot even vs 24-book lines), cards -.15, ht_lead,
# btts. The old SMOOTH=25 reached half-strength only at n=25 — season pacing, not
# tournament pacing; with ~15 games left the correction must arrive while it still
# matters. Kinds where the two datasets DISAGREE (player_sot, win_match, pen_or_red,
# fouls_compare) have no true bias and correctly get little/no push either way.
MIN_N = 10           # per-kind sample threshold
GROUP_MIN_N = 20     # group sample threshold (coarse tuning starts here)
SMOOTH = 12.0        # shrinkage strength (half-strength at n=12)
MAX_BIAS = 0.12      # cap any correction at +/-12 percentage points
_GROUP_CAUTION = 0.6  # extra damping on the coarse group bias

# Goal-derived markets priced off the sharp de-vigged line / market-anchored model.
ANCHORED = {
    "win_match", "total_over", "total_under", "half_total_over", "btts",
    "btts_and_total", "team_score", "team_score_half", "half_goals_compare",
    "half_vs_half_goals", "ht_tie", "ht_lead", "first_goal_half", "first_goal_game",
    "compound_first_2h",
    # goals-model-derived KO novelties (2026-07-05): same anchored treatment
    "goal_each_half", "hold_lead", "extra_time",
    # goals-model-derived finals novelties (2026-07-16)
    "two_goals_same_half", "first_goal_assist",
}

_cache: Dict | None = None


def _group(kind: str) -> str:
    return "anchored" if kind in ANCHORED else "prop"


def load_bias() -> Dict:
    global _cache
    if _cache is not None:
        return _cache
    try:
        with open(_FILE) as f:
            _cache = json.load(f)
    except (OSError, json.JSONDecodeError):
        _cache = {"kind": {}, "group": {}}
    return _cache


def bias_for(kind: str) -> float:
    """Per-kind bias if available, else the group bias, else 0."""
    b = load_bias()
    if kind in b.get("kind", {}):
        return b["kind"][kind]
    return b.get("group", {}).get(_group(kind), 0.0)


_novelty_cache: Dict[str, float] | None = None


def _novelty_anchors() -> Dict[str, float]:
    """Per-kind field base-rate anchor for no-edge novelty markets, learned from
    settled crowd consensus (state.crowd_base_rate), falling back to config defaults
    when samples are thin. Cached; cleared on update_bias() as new games settle."""
    global _novelty_cache
    if _novelty_cache is not None:
        return _novelty_cache
    out: Dict[str, float] = {}
    K = config.NOVELTY_ANCHOR_PRIOR_K
    # Kinds to anchor: the always-on defaults, PLUS any allowlisted binary-novelty kind the
    # measured RBP now PROVES is no-edge (data-driven — high conviction only where data backs it).
    kinds = dict(config.NOVELTY_ANCHOR_DEFAULTS)  # kind -> fallback prior
    try:
        summ = state.crowd_rbp_summary()
        for kind in config.NOVELTY_ANCHOR_ALLOWLIST:
            v = summ.get(kind)
            if v and v["n"] >= config.NOVELTY_ANCHOR_RBP_N and v["rbp"] <= config.NOVELTY_ANCHOR_RBP_THRESH:
                kinds.setdefault(kind, 0.30)  # proven no-edge -> anchor it
    except Exception:  # noqa: BLE001
        pass
    for kind, default in kinds.items():
        anchor = default
        try:
            br = state.crowd_base_rate(kind)
            if br and br["n"] >= config.NOVELTY_ANCHOR_MIN_N and br["mean_outcome"] is not None:
                # Bayesian-shrunk REALIZED rate: trust ground-truth outcomes, shrunk toward
                # the field consensus (prior) by K pseudo-counts. The crowd is itself wrong
                # on hydration, so the realized rate (not crowd_pred) is the anchor.
                prior = br["crowd_pred"] if br["crowd_pred"] is not None else default
                anchor = (br["mean_outcome"] * br["n"] + K * prior) / (br["n"] + K)
        except Exception:  # noqa: BLE001 - DB/absent table must never break pricing
            pass
        out[kind] = anchor
    _novelty_cache = out
    return out


_edge_cache: set | None = None
# Compare kinds the LOO retuner already owns — don't let the edge-boost fight it.
_EDGE_EXCLUDE = {"corners_compare", "fouls_compare", "cards_compare"}


def _edge_kinds() -> set:
    """Kinds the measured RBP proves we BEAT the field on AND are well-calibrated on:
    n>=N, mean RBP>=thresh, and |our mean pred - realized| small (winning because we're
    right, not winning-while-miscalibrated). Cached; cleared on update_bias()."""
    global _edge_cache
    if _edge_cache is not None:
        return _edge_cache
    out: set = set()
    try:
        for k, v in state.crowd_rbp_summary().items():
            if k in _EDGE_EXCLUDE:
                continue
            if (v["n"] >= config.EDGE_BOOST_N and v["rbp"] >= config.EDGE_BOOST_RBP
                    and v["our_pred"] is not None and v["mean_outcome"] is not None
                    and abs(v["our_pred"] - v["mean_outcome"]) <= config.EDGE_BOOST_CAL_ERR):
                out.add(k)
    except Exception:  # noqa: BLE001 - never break pricing
        pass
    _edge_cache = out
    return out


def edge_boost_factor(kind: str, base_factor: float) -> float:
    """Loosen the shrink factor toward 1.0 (less shrink) for proven-edge well-calibrated
    kinds. No-op unless EDGE_BOOST_ENABLED. Market tier already has factor 1.0 -> unchanged."""
    if not config.EDGE_BOOST_ENABLED or kind not in _edge_kinds():
        return base_factor
    return min(1.0, base_factor + config.EDGE_BOOST_MAX)


# Hydration is NOT one base rate — it's window/event-specific. The model mis-orders it
# (over-rates the cautious early window at 0.58, under-rates the goal-dense LATE window at
# 0.37 — the frac caps at 90', missing goal-rich stoppage). Flat-anchoring crushed the
# late-game GOAL market (cost us NED/MAR). Per-(window,event) anchors, football-logic-led
# (thin data): early goals stay low (we WIN fading them, 0/3 settled NO); late goal highest.
_HYDRATION_ANCHORS = {
    ("before_first", "goal"): 0.33, ("before_first", "offside"): 0.32,
    ("before_first", "card"): 0.28, ("before_first", "corner"): 0.45,
    ("after_first", "goal"): 0.40,  ("before_second", "goal"): 0.42,
    # after_second goal 0.46->0.53 (2026-07-01 audit): 2/2 settled YES at field ~0.51 while
    # we sat 0.29-0.30 pre-table (-96 RBP); goal-timing physics (goals cluster late + stoppage,
    # lam~2.6 -> P(goal after ~70') ~0.58) says 0.46 was still a fade of a likely event.
    ("after_second", "goal"): 0.53, ("after_second", "card"): 0.38,
    ("after_second", "offside"): 0.32,
}
_HYDRATION_W = 0.85  # model is unreliable here -> lean on the window anchor, keep 15% for the
                     # goal-environment signal (high-scoring game -> likelier in any window)


def _hydration_anchor(window, event) -> float:
    return _HYDRATION_ANCHORS.get((window or "before_first", event or "goal"),
                                  0.40 if event == "goal" else 0.30)


def novelty_anchor(kind: str, p: float, source: str,
                   window=None, event=None) -> tuple[float, str]:
    """Blend a no-edge novelty price toward its base-rate anchor. No-op unless
    NOVELTY_ANCHOR_ENABLED. Hydration uses window/event-specific anchors (it's not one
    rate); other kinds use the learned field base-rate. Returns (price, source)."""
    if not config.NOVELTY_ANCHOR_ENABLED:
        return p, source
    if kind == "hydration_event":
        # Corner-count windows are Poisson-computable from the corner line — the flat
        # anchors were learned from goal/card/offside TIMING scars (zero settled corner
        # markets) and would drag a computable price off truth (2026-07-01 audit:
        # "2+ corners before 1st break" anchored to .49 vs Poisson/field ~.60).
        if event == "corner":
            return p, source
        anchor = _hydration_anchor(window, event)
        return _HYDRATION_W * anchor + (1.0 - _HYDRATION_W) * p, f"{source} →hyd{anchor:.2f}"
    anchor = _novelty_anchors().get(kind)  # set = defaults + data-proven no-edge allowlisted kinds
    if anchor is None:
        return p, source
    w = config.NOVELTY_ANCHOR_WEIGHT
    return w * anchor + (1.0 - w) * p, f"{source} →field{anchor:.2f}"


def is_anchored(kind: str) -> bool:
    """True for goal-derived markets always priced off sharp lines. Their bias is
    a legitimate market-residual correction; prop-group bias is not (it was learned
    from group-stage MODEL prices, so it must not touch market-tier prop lines)."""
    return _group(kind) == "anchored"


def update_bias() -> Dict:
    """Recompute per-kind and per-group biases from settled results; persist."""
    global _cache, _novelty_cache, _edge_cache
    _novelty_cache = None  # new settlements -> re-learn field anchors lazily
    _edge_cache = None     # ...and re-evaluate which kinds have earned looser shrink
    summ = state.calibration_summary()
    out: Dict = {"kind": {}, "group": {}}

    # Per-kind (fine).
    for kind, v in summ.items():
        n = v["n"]
        if n >= MIN_N and v["mean_outcome"] is not None and v["mean_pred"] is not None:
            raw = v["mean_outcome"] - v["mean_pred"]
            out["kind"][kind] = max(-MAX_BIAS, min(MAX_BIAS, raw * (n / (n + SMOOTH))))

    # Per-group (coarse) — aggregate kinds into anchored vs prop.
    agg: Dict[str, list] = {}
    for kind, v in summ.items():
        if v["mean_outcome"] is None:
            continue
        a = agg.setdefault(_group(kind), [0, 0.0, 0.0])
        a[0] += v["n"]
        a[1] += v["n"] * v["mean_pred"]
        a[2] += v["n"] * v["mean_outcome"]
    for g, (n, sp, so) in agg.items():
        if n >= GROUP_MIN_N:
            gap = (so - sp) / n
            out["group"][g] = max(-MAX_BIAS, min(MAX_BIAS, gap * (n / (n + SMOOTH)) * _GROUP_CAUTION))

    _cache = out
    try:
        with open(_FILE, "w") as f:
            json.dump(out, f)
    except OSError as exc:
        log.warning("bias persist failed: %s", exc)
    if out["kind"] or out["group"]:
        log.info("calibration updated: group=%s kind=%s",
                 {k: round(v, 3) for k, v in out["group"].items()},
                 {k: round(v, 3) for k, v in out["kind"].items()})
    return out
