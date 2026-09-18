"""Central configuration. One place to tune the bot's behavior.

Secrets come from the environment so nothing sensitive lands in the repo.
Free-only data plan: the-odds-api (free tier) and API-Football (free tier) are
the keyed sources; SofaScore / FBref / Elo are keyless fallbacks so the bot can
still price and floor the board with zero keys present.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Minimal .env loader so every module sees the keys without extra deps."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except OSError:
        pass


_load_dotenv()
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
LOG_DIR = ROOT / "logs"
STATE_DB = DATA_DIR / "state.db"

# File contract between the Python engine and the MCP-driven agent loop.
# The agent writes markets it pulled via MCP here; the engine reads it.
MARKETS_SNAPSHOT = DATA_DIR / "markets_snapshot.json"
# The engine writes its priced decisions here; the agent reads it and submits.
DECISIONS_FILE = DATA_DIR / "decisions.json"
# The agent writes MCP submit/update responses here; the engine reconciles them.
SUBMIT_RESPONSES = DATA_DIR / "submit_responses.json"
# Human-readable dry-run table.
DRY_RUN_TABLE = DATA_DIR / "dry_run.txt"
# Kill switch: if this file exists, the loop stops submitting.
KILL_FILE = ROOT / "STOP"

for _d in (DATA_DIR, CACHE_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Contest identity (discovered live via MCP) ----------------------------
EVENT_ID = "aa5572ec-5930-4d99-b06b-f8966333d172"  # Jump Trading Probability Cup
LOBBY_ID = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"  # public "SportsPredict" lobby

# --- Secrets / data sources ------------------------------------------------
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()
APIFOOTBALL_KEY = os.getenv("APIFOOTBALL_KEY", "").strip()
FOOTBALLDATA_KEY = os.getenv("FOOTBALLDATA_KEY", "").strip()
# BetsAPI (Bet365 proxy): player/team props + niche lines (offsides/fouls/pens).
BETSAPI_TOKEN = os.getenv("BETSAPI_TOKEN", "").strip()
# Raw SportsPredict API token (sp_live_...). Usable only if the base URL/endpoint
# shapes are known; otherwise submission goes through the MCP surface.
SPORTPREDICT_API_KEY = os.getenv("SPORTPREDICT_API_KEY", "").strip()
SPORTPREDICT_API_BASE = os.getenv("SPORTPREDICT_API_BASE", "").strip()
# the-odds-api sport key for FIFA World Cup. Adjusted at runtime if needed.
ODDS_API_SPORT = os.getenv("ODDS_API_SPORT", "soccer_fifa_world_cup")
# Single region (eu) halves credit cost (markets x regions). Free tier = 500/mo,
# which must last the whole tournament, so we ration hard (see odds.py budget).
ODDS_API_REGIONS = os.getenv("ODDS_API_REGIONS", "eu")
ODDS_WC_END = os.getenv("ODDS_WC_END", "2026-07-19")  # ration the 500 credits to here
# Player props (per-event endpoint): need us/uk/eu books for coverage (Pinnacle
# supplies the clean Yes/No anytime-scorer line we anchor on). Fetched only inside
# the pre-KO window below and cached, so the extra credit cost stays small.
ODDS_PROP_REGIONS = os.getenv("ODDS_PROP_REGIONS", "us,uk,eu")
PROP_WINDOW_MIN = int(os.getenv("PROP_WINDOW_MIN", "300"))  # fetch props within 5h of KO
PROP_TTL_MIN = int(os.getenv("PROP_TTL_MIN", "30"))         # cache per-event props 30m

# --- Loop cadence & timing (minutes) ---------------------------------------
POLL_INTERVAL_MIN = 12          # main loop cadence
LINEUP_WINDOW_MIN = 60          # resubmit when lineups confirm (~60m pre-KO)
LOCK_WINDOW_MIN = 12            # within this of close_time -> "lock window"
CLOSING_WINDOW_MIN = 8          # within this of lock -> hard-refresh the line every cycle
                                # (bypass cache) so the final submit lands on the closing number
# A de-vigged line hours old still dominates the Elo prior, and budget rationing
# means we only refetch a few times a day — so trust a cached line for up to 12h
# (the near-kickoff force-refresh still captures the closing line where it counts).
STALE_ODDS_MIN = 720            # minutes; beyond this, fall back to the Elo model
STALE_STATS_HOURS = 24          # team-rate stats older than this get refreshed

# --- Move guards & bounds --------------------------------------------------
# A normal-tick resubmission may not swing more than this many *percentage
# points* from the prior forecast, so one bad data tick can't wipe a good line.
# Inside the lock window this cap is lifted (closing line is the truth).
MOVE_CAP_PP = 0.15
# Only resubmit if the new probability differs from the last by at least this,
# to avoid churning the API with no-op updates.
MIN_MOVE_TO_RESUBMIT_PP = 0.03
# Final clamp on any submitted probability (the API itself requires 1..99).
# We avoid the extremes unless a sharp market line genuinely justifies them.
PROB_FLOOR = 0.02
PROB_CEIL = 0.98

# --- Confidence tiers & shrink toward 0.5 ----------------------------------
# Lower-confidence estimates are shrunk toward 0.5 to reduce Brier variance.
# A sharp de-vigged market line is trusted as-is.
# Shrink only hedges *model error* — it must not flatten well-calibrated numbers
# toward the crowd, which would forfeit RBP edge. Sharp lines are trusted as-is;
# only the rougher Elo/base-rate tiers get a light hedge.
CONFIDENCE_SHRINK = {
    "market": 1.00,          # de-vigged sharp betting line — trust fully
    "model_anchored": 0.97,  # goals model fit to the de-vigged line — near-sharp
    "model_elo": 0.90,       # goals model from Elo prior (no odds) — light hedge
    "base_rate": 0.90,       # calibrated count base rate — light hedge
}
# Below this effective confidence we skip rather than submit (unless we still
# need to clear the participation floor, handled in the engine).
MIN_TIER_TO_SUBMIT = "base_rate"  # i.e. submit everything we can price at all
TIER_ORDER = ["base_rate", "model_elo", "model_anchored", "market"]

# corners_compare backtests WORSE than a coinflip (brier 0.363 vs 0.25, 2026-06-27)
# — corner dominance doesn't track match dominance, so our confident calls fail.
# No demonstrated edge -> shrink hard toward 0.5 (match the no-edge baseline),
# keeping a little signal. Applied in engine BEFORE the base_rate 0.90 shrink.
CORNERS_COMPARE_SHRINK = 0.5

# Weight on the (opponent-line-adjusted) for-sum vs the league base for total_offsides.
# Offside FOR-rates barely predict (scar) + the opponent-line lean is speculative (no sharp
# line) -> trust it only this much, base the rest. Floors low pairs toward base (CIV/NOR fix)
# AND tempers the matchup lean: 0.4 put FRA/SWE at ~51% (full conviction); hedged to 0.25 ->
# ~48% (Sharpe-optimal — don't over-bet a read we can't anchor to a line). Balanced games
# stay at base (.44) regardless; the corner anchor + CIV/NOR floor are unaffected.
OFFSIDE_FOR_W = 0.25

# team_corners off the Bet365 total-corners line: split the sharp total by a dominance-tilted
# share. Raw corner FOR-rates under-credit the favorite (it camps in the opp half → more
# corners than its flat rate), so pull the for-rate share toward the goal-lambda share by this
# much. Validated on FRA/SWE: raw share .48 -> tilted .57 -> France 6+ corners 32% -> ~45%
# (matches the sharp line + a 57/43 corner-split prediction; crowd's 58% over-shoots).
# 2026-07-01: briefly raised .35->.45 + cap .68->.74 to close the -34 RBP team_corners leak, then
# REVERTED after an adversarial post-mortem (ENG/COD). The escalation was justified by a "bus-park
# underdog concedes a 70/30 corner split" STORY, which ENG/COD FALSIFIED — the game ran OPEN (COD
# scored), England took <8 corners, and the tilt directionally cost RBP (we beat the crowd .43 vs
# .49 only because the crowd was even higher). The -34 leak is real but must be closed with DATA,
# not a premise: crowd RBP is now captured per-game for the whole R16 slate (crowd_stats.MATCHES),
# so DERIVE the right corner correction once several games settle rather than betting a narrative.
CORNER_DOMINANCE_TILT = 0.35

# No-edge NOVELTY markets: measured (crowd_results) that the field beats our blind
# model badly (hydration_event -42.9 RBP, sub_scorer -22.3) — confident calls in
# markets we cannot model. Anchor toward the learned field base-rate instead.
# OFF by default: a single kind-average anchor is too crude for heterogeneous kinds
# (hydration sub-types differ) — enable only after per-sub-market base rates firm up.
NOVELTY_ANCHOR_ENABLED = os.getenv("NOVELTY_ANCHOR", "0").strip() in ("1", "true", "True")
NOVELTY_ANCHOR_WEIGHT = float(os.getenv("NOVELTY_ANCHOR_WEIGHT", "0.7"))  # weight on the base-rate anchor
NOVELTY_ANCHOR_MIN_N = 4    # need >= this many settled samples to use the learned rate
NOVELTY_ANCHOR_PRIOR_K = 6  # pseudo-count of the prior in the Bayesian-shrunk realized rate
# Anchor = REALIZED base rate (ground truth), shrunk toward this prior by PRIOR_K. The
# crowd is also wrong on hydration (6/6 settled NO at avg field 0.50) so we anchor to the
# realized outcome, not the field. Defaults are the prior + the thin-data fallback.
NOVELTY_ANCHOR_DEFAULTS = {"hydration_event": 0.30, "sub_scorer": 0.30}
# DATA-DRIVEN conviction guardrail: un-modelable binary occurrence/timing markets eligible
# to be conviction-capped IF the measured RBP proves we have no edge there. Keeps high
# conviction only where data backs it; a kind auto-anchors once it crosses the RBP gate.
# (Allowlist, not auto-detect-all: threshold-count kinds have no single base rate to anchor.)
NOVELTY_ANCHOR_ALLOWLIST = {"hydration_event", "sub_scorer", "sub_before_ht", "stoppage_goal"}
NOVELTY_ANCHOR_RBP_N = 6        # need >= this many settled crowd markets to judge edge
NOVELTY_ANCHOR_RBP_THRESH = -3.0  # mean RBP/market <= this (proven losing to field) -> anchor

# INVERSE guardrail (blowout amplifier): LOOSEN shrink on kinds the RBP data PROVES we beat
# the field on AND are well-calibrated -> diverge harder from the crowd where edge is earned.
# Higher bar than the no-edge cap (overfitting risk). LOO-retuned compare kinds are excluded
# (the retuner already owns their shrink). OFF by default; enable via EDGE_BOOST=1 in .env.
EDGE_BOOST_ENABLED = os.getenv("EDGE_BOOST", "0").strip() in ("1", "true", "True")
# 2026-07-02 go-for-the-win: gates widened 12->8 / .06->.08. At the old gates only
# player_sot boosted; total_cards (n=8, +10.9 RBP/mkt, cal_err .062 — and the .06
# miss is in the direction the boost CORRECTS: we price ~6pp above realized on a
# fade kind) was excluded by a rounding hair. team_sot (cal .099) stays out — its
# miscalibration is real. Auto-extends as kinds reach n>=8.
EDGE_BOOST_N = 8           # need >= this many settled crowd markets
EDGE_BOOST_RBP = 3.0       # mean RBP/market >= this (proven beating the field)
EDGE_BOOST_CAL_ERR = 0.08  # AND |our_pred - realized| <= this (well-calibrated, not just lucky)
EDGE_BOOST_MAX = 0.20      # max bump to the shrink factor (toward 1.0 = less shrink)

# CROWD-CONSENSUS GUARDRAIL: the web match-stats endpoint exposes the live field average
# per market pre-lock. Counterfactual on the 174 settled crowd markets (2026-07-01 audit):
# pulling any >GAP divergence halfway to the field on kinds WITHOUT measured positive RBP
# = +107 RBP, positive on every affected kind, and +86 even with NO kind exemptions (so
# the rule is robust, not fit to the exemption list). Proven-edge kinds are exempt — their
# divergence IS the edge. ON by default; kill switch CROWD_GUARD=0. Fetch errors -> no-op.
CROWD_GUARD_ENABLED = os.getenv("CROWD_GUARD", "1").strip() in ("1", "true", "True")
CROWD_GUARD_GAP = 0.15   # only act beyond this divergence from the field average
CROWD_GUARD_PULL = 0.5   # pull this fraction of the way toward the field

# RANK-PROTECTION CAP — OFF (2026-07-02, user: "remove the big losses directive,
# let's go for the win"). From rank 231/3725 the climb needs +150/game upside, so
# proven edges now run at FULL model size (cards fades back to the bet365 lambda,
# shots overs at model conviction). The cap remains available for the FINAL or a
# defend-the-rank endgame: RISK_CAP=1 re-enables (and set RISK_WEIGHT=3.0 for the
# final). While OFF, breakage protection still comes from the guard (non-exempt
# kinds pulled to field), the n>=4 exemption floor, and field-sit kinds — those
# layers are EV-positive, not variance trades, so they stay in both regimes.
# History: cap ON would have been +140 RBP over the first 200 settled markets
# (USA/BIH 22+shots -70 -> -30); top-8-study max |dev| 0.165 ~= cap at 30.
RISK_CAP_ENABLED = os.getenv("RISK_CAP", "0").strip() in ("1", "true", "True")
MAX_MKT_LOSS_RBP = 30.0
RISK_WEIGHT = 2.0  # knockout stage; set 3.0 for the FINAL (its markets weigh 3x)

# Share of goals scored with the head (WC-era ~18-20%) -> P(>=1 header goal) =
# 1 - exp(-share * lam_total). Prices the new R16 "header goal" novelty kind.
HEADER_GOAL_SHARE = 0.19

# Same thinning for "goal from outside the penalty area" (~11-14% of goals).
OUTSIDE_BOX_GOAL_SHARE = 0.13
# P(extra time produces no winner | level after 90') — prices "decided by a
# penalty shootout" as p_draw * ET_TO_PENS. Big-tournament empirical ~0.45-0.55.
ET_TO_PENS = 0.5

# P(first goal credited with an assist | >=1 goal). Our WC26 player accumulator:
# 211 assists / 281 goals = .751; haircut to .72 since own goals sit in the match
# goal count but are never assisted (pens already dilute the numerator).
FIRST_GOAL_ASSIST_SHARE = 0.72

# Kinds whose market-tier price is DERIVED from a line (lambda->threshold transform)
# rather than directly quoted — the learned per-kind calibration bias stays ON for
# these (the transform carries measured tournament-level error; see engine bias
# suppression comment). Cards only for now: bias +0.10 measured over n=10, press
# counterfactual monotone positive to +10pp. Corners NOT included — their bias
# sample is confounded by the favvol siege effect (both directions present).
BIAS_ON_DERIVED_MARKET = {"total_cards", "team_cards",
                          # corners lambdas are the same class of transform; measured
                          # bias: total_corners -.25/-.38 (overs die), team_corners
                          # +.10 (aligned with the favvol lift, no conflict)
                          "total_corners", "team_corners"}

# Weight on a single-book de-vigged SoT line when blending with the WC-only model.
# Was 0.5 — that diluted Bet365's sharp Bellingham 2+SoT 2.625 (~.36) with a thin-n
# this-WC model (~.20) down to .28 (crowd .40, settled YES, -23 RBP). Doctrine: a sharp
# line is data-backed by definition; the WC-only model carries no career priors and
# systematically under-rates star players until they produce IN this tournament. Keep
# 25% model for the lineup/rotation signal a stale line may lag.
SOT_1BK_LINE_W = 0.75

# --- HTTP / retry ----------------------------------------------------------
HTTP_TIMEOUT = 15
HTTP_RETRIES = 4
HTTP_BACKOFF_BASE = 1.5  # seconds; exponential
USER_AGENT = "jumpcup-floorbot/1.0 (+probability-forecasting)"


def clamp_prob(p: float) -> float:
    """Clamp a probability into the safe submission band."""
    return max(PROB_FLOOR, min(PROB_CEIL, p))


def to_api_int(p: float) -> int:
    """Convert a [0,1] probability to the integer 1..99 the API expects."""
    return int(max(1, min(99, round(p * 100))))


# Per-kind shrink overrides, maintained by retune.py (the shrink auto-tuner) as
# settled data justifies loosening the static tier shrink. Empty until then, so
# behavior is unchanged until the tuner has earned a change on out-of-sample data.
SHRINK_OVERRIDES_FILE = DATA_DIR / "shrink_overrides.json"


def _load_shrink_overrides() -> dict:
    try:
        with open(SHRINK_OVERRIDES_FILE) as f:
            return {k: float(v) for k, v in json.load(f).items()}
    except (OSError, json.JSONDecodeError, ValueError, AttributeError):
        return {}


SHRINK_OVERRIDES = _load_shrink_overrides()


def save_shrink_overrides() -> None:
    """Persist the per-kind shrink overrides (atomic) so they survive a restart."""
    tmp = SHRINK_OVERRIDES_FILE.with_suffix(".tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(SHRINK_OVERRIDES, f)
        tmp.replace(SHRINK_OVERRIDES_FILE)
    except OSError:
        pass


def shrink_factor(kind: Optional[str], tier: str) -> float:
    """Effective shrink factor: an auto-tuned per-kind override if present, else the
    static tier default."""
    if kind and kind in SHRINK_OVERRIDES:
        return SHRINK_OVERRIDES[kind]
    return CONFIDENCE_SHRINK.get(tier, 0.7)


def shrink(p: float, tier: str, kind: Optional[str] = None) -> float:
    """Shrink probability toward 0.5 by its confidence tier (or per-kind override)."""
    return 0.5 + shrink_factor(kind, tier) * (p - 0.5)
