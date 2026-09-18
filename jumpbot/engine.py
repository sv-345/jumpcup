"""The pricing engine: turn open markets into auditable, gated decisions.

Flow per tick:
  1. Read the markets snapshot the agent pulled via MCP.
  2. Per match: resolve teams, fetch de-vigged odds, build a goals model
     (market-anchored when we have odds, Elo-derived otherwise).
  3. Classify each question and price it, tagging a confidence tier.
  4. Apply confidence shrink + clamp; compare to prior state; enforce the
     bounded-move cap outside the lock window; decide submit / update / hold /
     skip with a logged reason.

Nothing here calls MCP or the network's submission surface — it only reads odds
and writes Decision objects. One bad market never raises out of here.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from . import calibration, config, crowd_guard, model, odds as odds_mod, props_model, ratings, state, stats
from .logging_setup import get_logger
from .mapping import Classification, classify, parse_match_name

log = get_logger("jumpbot.engine")


@dataclass
class Decision:
    market_id: str
    lobby_id: str
    question: str
    match_name: str
    kind: str
    tier: str
    prob: float                 # final probability in [0,1] after shrink+clamp
    prob_int: int               # 1..99 for the API
    action: str                 # submit | update | hold | skip
    reason: str
    source: str
    raw_prob: Optional[float] = None   # pre-shrink model/market probability
    prediction_id: Optional[str] = None
    inputs: dict = field(default_factory=dict)


def _avg_lambda(gm: model.GoalsModel) -> float:
    return (gm.lam_h + gm.lam_a) / 2.0


def _build_goals_model(home: Optional[str], away: Optional[str],
                       mo: Optional[odds_mod.MatchOdds]) -> model.GoalsModel:
    """Market-anchored goals model if odds are fresh, else Elo-derived."""
    fresh = mo is not None and mo.age_min() <= config.STALE_ODDS_MIN
    if fresh:
        lam_h, lam_a, tier = model.lambdas_from_market(
            mo.p_home, mo.p_away, total_line=mo.total_line, p_over=mo.p_over
        )
    else:
        lam_h, lam_a, tier = ratings.lambdas_for_match(home, away)
    return model.GoalsModel(lam_h, lam_a, tier)


def _price(c: Classification, home: Optional[str], away: Optional[str],
           mo: Optional[odds_mod.MatchOdds], gm: model.GoalsModel,
           outrights: Optional[Dict[str, float]]):
    """Return (prob, tier, source, inputs) or None to skip. Pure pricing."""
    odds_fresh = mo is not None and mo.age_min() <= config.STALE_ODDS_MIN
    avg_lam = _avg_lambda(gm)
    is_home = c.side == "home"

    def attack(side: str) -> float:
        lam = gm.team_lambda(side == "home")
        return max(0.6, min(1.6, lam / max(avg_lam, 0.4)))

    def foul_f(side: str) -> float:  # weaker team commits/receives more
        return max(0.6, min(1.7, 1.0 / attack(side)))

    def code_of(side: str) -> Optional[str]:
        return home if side == "home" else away

    # Half-of-match scaling relative to a full-match team rate.
    HALF = {"sot1": 0.49, "sot2": 0.51, "corners1": 0.46, "corners2": 0.54,
            "cards1": 0.43, "cards2": 0.57}

    def team_mean(side: str, metric: str, base_key: str, tilt: str = "attack",
                  scale: float = 1.0):
        """Prefer the live per-team rate; else league base with a strength tilt."""
        code = code_of(side)
        tr = stats.team_rate(code) if code else None
        if tr and tr.get(metric, 0) > 0:
            return tr[metric] * scale, "team-data"
        f = attack(side) if tilt == "attack" else (foul_f(side) if tilt == "foul" else 1.0)
        return model.BASE[base_key] * f * scale, "base"

    def total_mean(metric: str, total_base_key: str, scale: float = 1.0):
        th = stats.team_rate(code_of("home")) if code_of("home") else None
        ta = stats.team_rate(code_of("away")) if code_of("away") else None
        if th and ta and th.get(metric, 0) > 0 and ta.get(metric, 0) > 0:
            return (th[metric] + ta[metric]) * scale, "team-data"
        return model.BASE[total_base_key] * scale, "base"

    def adj_mean(side: str, metric: str, base_key: str, tilt: str = "attack",
                 scale: float = 1.0):
        """Opponent-adjusted expected count for a compare market: this team's
        for-rate scaled by how leaky the opponent is vs the league mean
        (attack x defense). Both rates are EB-shrunk in stats.team_rate, so it
        self-tempers toward the own-rate when data is thin. Falls back to
        team_mean (own-rate / base) when either side lacks accumulator data."""
        opp_side = "away" if side == "home" else "home"
        tr_me = stats.team_rate(code_of(side)) if code_of(side) else None
        tr_opp = stats.team_rate(code_of(opp_side)) if code_of(opp_side) else None
        if tr_me and tr_opp and tr_me.get(metric, 0) > 0:
            lg = stats.league_rate(metric) or 1.0
            opp_def = tr_opp.get(metric + "_against", lg)
            return tr_me[metric] * (opp_def / lg) * scale, "opp-adj"
        return team_mean(side, metric, base_key, tilt=tilt, scale=scale)

    k, t = c.kind, gm.tier

    # ---------- goal-derived (model-anchored) ----------
    if k == "win_match":
        if odds_fresh:
            p = mo.p_home if is_home else mo.p_away
            return p, "market", f"odds 1X2 ({mo.n_books} books, {mo.age_min():.0f}m)", \
                {"p_home": mo.p_home, "p_draw": mo.p_draw, "p_away": mo.p_away}
        return (gm.p_home_win if is_home else gm.p_away_win), t, "goals model", \
            {"lam_h": gm.lam_h, "lam_a": gm.lam_a}
    if k == "total_over":
        # "N or more goals" == Over (N-0.5). If a book quotes that exact line,
        # price straight off the de-vigged total (market tier) — the joint 1X2+
        # total fit weights 1X2 2x and can leave the total ~7pp off when they
        # conflict (e.g. ALG/AUT 1.5: fit 0.53 vs line 0.60). Else extrapolate.
        if odds_fresh and mo.p_over is not None and mo.total_line is not None \
                and abs(mo.total_line - (c.threshold - 0.5)) < 0.01:
            return mo.p_over, "market", f"odds O{mo.total_line:g} ({mo.n_books}bk)", {}
        return gm.p_total_ge(c.threshold), t, f"model total>={c.threshold}", {}
    if k == "total_under":
        # "N or fewer goals" == Under (N+0.5); mirror the over branch.
        if odds_fresh and mo.p_over is not None and mo.total_line is not None \
                and abs(mo.total_line - (c.threshold + 0.5)) < 0.01:
            return 1.0 - mo.p_over, "market", f"odds U{mo.total_line:g} ({mo.n_books}bk)", {}
        return 1.0 - gm.p_total_ge(c.threshold + 1), t, f"model total<={c.threshold}", {}
    if k == "half_total_over":
        return gm.p_half_total_ge(c.threshold, second_half=(c.half != 1)), t, \
            f"model H{c.half} total>={c.threshold}", {}
    _btts_window = (home and away and mo is not None and mo.commence_ts
                    and 0 <= (mo.commence_ts - time.time()) <= config.PROP_WINDOW_MIN * 60)
    if k == "btts":
        if _btts_window:
            bm = odds_mod.btts_market(home, away)
            if bm:
                return bm[0], "market", f"odds btts ({bm[1]}bk)", {}
        return gm.p_btts, t, "model BTTS", {}
    if k == "btts_and_total":
        joint = gm.p_btts_and_total_ge(c.threshold)
        if _btts_window:
            bm = odds_mod.btts_market(home, away)
            if bm and gm.p_btts > 0.05:  # anchor the btts marginal to the sharp line,
                anchored = max(0.01, min(0.97, joint * (bm[0] / gm.p_btts)))  # keep model's total|btts
                return anchored, "market", f"btts-anchored>={c.threshold} ({bm[1]}bk)", {}
        return joint, t, f"model BTTS&total>={c.threshold}", {}
    if k == "team_score":
        return gm.p_team_scores(is_home, c.threshold or 1), t, \
            f"model {c.side} scores>={c.threshold or 1}", {"lam": gm.team_lambda(is_home)}
    if k == "team_score_half":
        return gm.p_team_scores_in_half(is_home, c.half != 1), t, \
            f"model {c.side} scores in H{c.half}", {}
    if k == "half_goals_compare":
        return gm.p_half_goal_supremacy(is_home, c.half != 1), t, \
            f"model {c.side} more H{c.half} goals", {}
    if k == "half_vs_half_goals":
        return gm.p_2h_more_goals_than_1h(), t, "model 2H>1H goals", {}
    if k == "ht_tie":
        return gm.p_ht_draw(), t, "model HT draw", {}
    if k == "ht_lead":
        return gm.p_ht_lead(is_home), t, f"model {c.side} leads at HT", {}
    if k == "first_goal_half":
        return gm.p_first_goal_of_half(is_home, c.half != 1), t, f"model {c.side} first goal H{c.half}", {}
    if k == "first_goal_game":
        return gm.p_first_goal_game(is_home), t, f"model {c.side} first goal", {}
    if k == "compound_first_2h":
        p = gm.p_first_goal_game(c.side == "home") * gm.p_team_scores_in_half(c.side2 == "home", True)
        return p, t, "model first-goal & 2H-score", {}

    # ---------- novel knockout markets ----------
    if k == "clean_sheet":
        # P(opponent scores 0) off the market-anchored goals model.
        return 1.0 - gm.p_team_scores(not is_home, 1), t, \
            f"model {c.side} clean sheet", {"lam_opp": gm.team_lambda(not is_home)}
    if k == "header_goal":
        # Headers are ~HEADER_GOAL_SHARE of goals -> thin a Poisson total by it.
        lam_tot = gm.lam_h + gm.lam_a
        return 1.0 - math.exp(-config.HEADER_GOAL_SHARE * lam_tot), "base_rate", \
            f"header share={config.HEADER_GOAL_SHARE}", {"lam_tot": lam_tot}
    if k == "match_tie":  # full-time (regulation) draw
        if odds_fresh:
            return mo.p_draw, "market", f"odds draw ({mo.n_books}bk)", {}
        return gm.p_draw, t, "model draw", {}
    if k == "team_advance":  # knockout: advance = win the tie (90' + ET/pens share)
        if odds_fresh:
            pw = mo.p_home if is_home else mo.p_away
            pl = mo.p_away if is_home else mo.p_home
            et = pw / (pw + pl) if (pw + pl) > 1e-9 else 0.5
            return pw + mo.p_draw * et, "market", f"odds advance ({mo.n_books}bk)", {}
        return gm.p_advance(is_home), t, "model advance", {}
    if k == "extra_time":  # level after 90' sends it to ET == the draw price
        if odds_fresh:
            return mo.p_draw, "market", f"odds ET=draw ({mo.n_books}bk)", {}
        return gm.p_draw, t, "model ET=draw", {}
    if k == "goal_each_half":
        p = gm.p_half_total_ge(1, False) * gm.p_half_total_ge(1, True)
        return p, t, "model goal each half", {}
    if k == "shootout":  # reaches penalties = 90' draw AND extra time undecided
        if odds_fresh:
            return mo.p_draw * config.ET_TO_PENS, "market", \
                f"shootout draw({mo.n_books}bk)x{config.ET_TO_PENS}", {}
        return gm.p_draw * config.ET_TO_PENS, t, "model shootout", {}
    if k == "hold_lead":  # side strictly ahead at some point (regulation model;
        # ET lead chances add ~1-2pp — acceptable underestimate, calibration absorbs)
        return gm.p_ever_leads(is_home), t, f"model {c.side} ever leads", {}
    if k == "outside_box_goal":
        lam_tot = gm.lam_h + gm.lam_a
        return 1.0 - math.exp(-config.OUTSIDE_BOX_GOAL_SHARE * lam_tot), "base_rate", \
            f"outside-box share={config.OUTSIDE_BOX_GOAL_SHARE}", {"lam_tot": lam_tot}
    if k == "sub_scorer":  # any substitute scores — base rate scaled by goal env
        lam_tot = gm.lam_h + gm.lam_a
        p = max(0.04, min(0.40, 0.18 * lam_tot / model.BASE["total_goals"]))
        return p, "base_rate", f"sub-scorer[λ={lam_tot:.2f}]", {}
    if k == "any_brace":  # any player scores 2+
        lam_tot = gm.lam_h + gm.lam_a
        return model.p_any_brace(lam_tot), "base_rate", f"any-brace[λ={lam_tot:.2f}]", {}
    if k == "hydration_event":  # event before/after a hydration break (time window)
        ev = c.event or "goal"
        share2 = {"goal": model.BASE["second_half_goal_share"],
                  "card": model.BASE["total_cards_2h"] / model.BASE["total_cards"],
                  "corner": model.BASE["total_corners_2h"] / model.BASE["total_corners"],
                  "offside": 0.5}.get(ev, 0.5)
        frac = model.hydration_frac(c.window or "before_first", share2)
        if c.reason == "incl_et" and (c.window or "").startswith("after_second"):
            frac *= 1.25  # rough extra-time extension of the late window
        if ev == "goal":
            lam_tot = gm.lam_h + gm.lam_a
        elif ev == "card":
            lam_tot = total_mean("cards", "total_cards")[0]
        elif ev == "corner":
            lam_tot = total_mean("corners", "total_corners")[0]
        else:  # offsides — no total base key; sum team rates or fall back to 2x base
            th = stats.team_rate(home) if home else None
            ta = stats.team_rate(away) if away else None
            lam_tot = ((th["offsides"] + ta["offsides"]) if (th and ta
                       and th.get("offsides", 0) > 0 and ta.get("offsides", 0) > 0)
                       else 2 * model.BASE["team_offsides"])
        if ev == "corner" and (c.window or "").startswith("before_first"):
            frac *= 0.85  # corner rate ramps through a half; a 0-30' window runs cool
        n = max(1, c.threshold or 1)
        if n > 1:  # count question: P(>=n) in the window, NOT occurrence P(>=1)
            p = float(model.poisson.sf(n - 1, lam_tot * frac))
            return p, "base_rate", f"hydration {ev}>={n} {c.window}[f={frac:.2f}]", {}
        return model.p_event_in_window(lam_tot, frac), "base_rate", \
            f"hydration {ev} {c.window}[f={frac:.2f}]", {}
    if k == "total_shots":  # shots on+off target; scale base by the goal environment
        # ...but shots do NOT crater with goals — a cagey low-scoring game still generates
        # ~22 shots (SoT_sum ~0.35-0.40 of total shots). The old 0.7 floor over-suppressed
        # (MEX/ECU 18.2 vs their own 9.63 combined SoT implying ~24). Floor at 0.82 so volume
        # stays realistic in low-scoring games (MEX/ECU 20+ shots 36% -> ~54, matching the crowd).
        env = (gm.lam_h + gm.lam_a) / model.BASE["total_goals"]
        mean = model.BASE["total_shots"] * max(0.82, min(1.4, env))
        return model.prob_count_ge(mean, c.threshold or 1), "base_rate", f"total shots {mean:.1f}", {}
    if k == "total_offsides":
        trh = stats.team_rate(home) if home else None
        tra = stats.team_rate(away) if away else None
        lg = 2 * model.BASE["team_offsides"]   # league total (~3.4 -> P(4+)~.44)
        lgt = model.BASE["team_offsides"]
        if (trh and tra and trh.get("offsides", 0) > 0 and tra.get("offsides", 0) > 0):
            # Offsides are driven by the OPPONENT's line height (a structural signal, unlike
            # the scarred own-rate): pacey forwards vs a high line get caught a lot. Scale each
            # team's for-rate by the opp's offsides-against (clamped), THEN shrink to base — the
            # shrink keeps it scar-safe. FRA/SWE: SWE high line (2.85 against) -> France ~.52.
            def line_f(v):
                return max(0.55, min(1.9, (v or lgt) / lgt))
            adj = trh["offsides"] * line_f(tra.get("offsides_against")) \
                + tra["offsides"] * line_f(trh.get("offsides_against"))
            mean = config.OFFSIDE_FOR_W * adj + (1 - config.OFFSIDE_FOR_W) * lg
        else:
            mean = lg
        return model.prob_count_ge(mean, c.threshold or 1), "base_rate", f"total offsides {mean:.2f}", {}
    if k == "red_card":  # standalone red; scale the base red rate by the card environment
        cmean = total_mean("cards", "total_cards")[0]
        p = max(0.03, min(0.30, model.BASE["red_card_match"] * cmean / model.BASE["total_cards"]))
        return p, "base_rate", f"red card[cards {cmean:.1f}]", {}
    if k == "both_teams_card":
        # Bet365's both-teams-to-be-carded line — the model multiplies two group-
        # calibrated per-team rates and badly under-prices KO games (0.37 vs ~0.68).
        if (home and away and mo is not None and mo.commence_ts
                and 0 <= (mo.commence_ts - time.time()) <= config.PROP_WINDOW_MIN * 60):
            from . import bet365
            bp = bet365.both_teams_card_prob(home, away)
            if bp is not None:
                return bp, "market", "bet365 btc", {}
        trh = stats.team_rate(home) if home else None
        tra = stats.team_rate(away) if away else None
        ch = trh["cards"] if (trh and trh.get("cards")) else model.BASE["team_cards"]
        ca = tra["cards"] if (tra and tra.get("cards")) else model.BASE["team_cards"]
        return model.prob_at_least_one(ch) * model.prob_at_least_one(ca), "base_rate", "both teams card", {}
    if k == "any_card_half":
        s2 = model.BASE["total_cards_2h"] / model.BASE["total_cards"]
        cmean = total_mean("cards", "total_cards")[0] * (s2 if c.half == 2 else 1 - s2)
        return model.prob_at_least_one(cmean), "base_rate", f"any card H{c.half}", {}
    if k == "any_player_2sot":  # some player with 2+ shots on target
        if c.side in ("home", "away"):  # team-scoped QF variant ("any Portugal player")
            m_sot = props_model.sot_lambda(gm.team_lambda(c.side == "home"))
            return model.p_any_brace(m_sot, k_eff=12.5), "base_rate", \
                f"any-{c.side}-player-2SoT[{m_sot:.1f}]", {}
        tot_sot = total_mean("sot", "total_sot")[0]
        return model.p_any_brace(tot_sot, k_eff=25.0), "base_rate", f"any-player-2SoT[{tot_sot:.1f}]", {}
    if k == "sub_before_ht":  # a substitution before halftime (usually injury) — base
        return 0.22, "base_rate", "sub before HT base", {}
    if k == "stoppage_goal":  # goal in 1H / 2H stoppage time
        # WC26 stoppages run LONG (SAOT reviews + heat protocol): H1 ~3-4', H2 ~8-10'
        # observed (2022 precedent ~8.4' H2). The old 5' H2 default under-priced the
        # window ~2x (us .19 vs field .26 on USA/BIH, 2026-07-01 audit).
        lam_tot = gm.lam_h + gm.lam_a
        mins, sh = ((3.5, 1 - model.BASE["second_half_goal_share"]) if c.half == 1
                    else (8.5, model.BASE["second_half_goal_share"]))
        frac = (mins / 45.0) * sh
        return model.p_event_in_window(lam_tot, frac), "base_rate", f"stoppage goal H{c.half}[f={frac:.3f}]", {}
    if k == "team_score_both_halves":
        p = gm.p_team_scores_in_half(is_home, False) * gm.p_team_scores_in_half(is_home, True)
        return p, t, f"model {c.side} both halves", {}
    if k == "win_margin":
        return gm.p_win_by(is_home, c.threshold or 2), t, f"model {c.side} win by {c.threshold or 2}+", {}
    if k == "two_goals_same_half":
        # "either team scores >=n in the same half": 1 - prod over the four
        # (team, half) cells of P(cell < n); cell goals ~ Poisson(team λ x half share).
        n = c.threshold or 2
        p_none = 1.0
        for hm in (True, False):
            for h2 in (False, True):
                p_none *= float(model.poisson.cdf(n - 1, gm.half_team_lambda(hm, h2)))
        return 1.0 - p_none, t, f"model {n}+ same-half burst", {}
    if k == "first_goal_assist":
        # P(>=1 goal) x tournament assist share; settles No on 0-0, which the
        # p_total_ge(1) factor carries.
        return gm.p_total_ge(1) * config.FIRST_GOAL_ASSIST_SHARE, t, \
            f"first-goal-assist share={config.FIRST_GOAL_ASSIST_SHARE}", {}
    if k == "player_sot_compare":  # A strictly out-SoTs B (a tie settles No)
        # Independent Poissons on each player's EB-shrunk WC SoT/90, both damped
        # by the match goal environment (a tight final suppresses both equally).
        tight = max(0.6, min(1.4, (gm.lam_h + gm.lam_a) / model.BASE["total_goals"]))
        la = (stats.player_sot_rate(c.player or "") or 0.9) * tight
        lb = (stats.player_sot_rate(c.player2 or "") or 0.9) * tight
        p = sum(float(model.poisson.pmf(a, la)) * float(model.poisson.cdf(a - 1, lb))
                for a in range(1, 13))
        return p, "base_rate", f"SoT duel {la:.2f}v{lb:.2f}", {}
    if k == "distinct_shooters":
        # P(>=n different players of one team attempt a shot): Poissonize team
        # shots over a fixed shooter-concentration profile -> independent per-
        # player "shot taken" indicators -> Poisson-binomial tail by DP. The
        # 0.82 env floor mirrors total_shots: volume doesn't crater in cagey games.
        env = gm.team_lambda(is_home) / (model.BASE["total_goals"] / 2.0)
        lam_shots = (model.BASE["total_shots"] / 2.0) * max(0.82, min(1.4, env))
        dp = [1.0] + [0.0] * len(model.SHOT_WEIGHTS)
        for w in model.SHOT_WEIGHTS:
            pi = 1.0 - math.exp(-lam_shots * w)
            for j in range(len(dp) - 1, 0, -1):
                dp[j] = dp[j] * (1.0 - pi) + dp[j - 1] * pi
            dp[0] *= (1.0 - pi)
        n = c.threshold or 5
        return float(sum(dp[n:])), "base_rate", f"distinct shooters[λ={lam_shots:.1f}]", {}

    # ---------- shots on target (team rates when available) ----------
    if k == "team_offsides":
        # Offsides barely scale with attacking strength (even weak teams get
        # flagged), so use the team-neutral base rather than an attack tilt.
        mean, src = team_mean(c.side, "offsides", "team_offsides", tilt="none")
        # Backtest (2026-06-26, 15 settled "caught offside 2+" markets): a team's
        # offside rate ABOVE base carries no predictive signal — high-rate teams
        # went 1/4 to 2+ (SWE/KOR/TUR high->no) and the sharp crowd prices every
        # team flat ~0.47; meanwhile thin 2-game samples blow P(2+) up (COL 5.0/g
        # -> 0.82 vs crowd 0.48). Cap the rate at base so it only informs DOWNWARD
        # (low-offside teams keep their correct lower P(2+)). Ground-truth brier
        # 0.242->0.194, and COL/IRN/GHA land on the crowd. The cap is the settled-
        # optimum AND the crowd value — see [[jumpcup-crowd-divergence-finding]].
        mean = min(mean, model.BASE["team_offsides"])
        return model.prob_count_ge(mean, c.threshold), "base_rate", f"offsides {mean:.2f}[{src}]", {}
    # Validated NegBin SoT model (props_model): map the market goal-lambda ->
    # E[SoT] -> NegBin, opponent-adjusted instead of a flat base rate. Tier kept
    # at base_rate so ONLY the point estimate changes (same shrink + calibration),
    # isolating the out-of-sample-validated improvement.
    if k == "team_sot":
        lam = gm.team_lambda(is_home)
        return props_model.p_sot_ge(lam, c.threshold), "base_rate", f"SoT negbin lg{lam:.2f}", {}
    if k == "team_sot_half":
        lam = gm.team_lambda(is_home)
        return props_model.p_sot_ge(lam, c.threshold, share=props_model.SOT_SHARE_2H), \
            "base_rate", f"2H SoT negbin lg{lam:.2f}", {}
    if k == "total_sot":
        return props_model.p_total_sot_ge(gm.lam_h, gm.lam_a, c.threshold), \
            "base_rate", "total SoT negbin", {}
    if k == "total_sot_half":
        return props_model.p_total_sot_ge(gm.lam_h, gm.lam_a, c.threshold,
                                          share=props_model.SOT_SHARE_2H), \
            "base_rate", "2H total SoT negbin", {}
    if k == "both_teams_sot":
        n = c.threshold or 1
        share = {1: props_model.SOT_SHARE_1H, 2: props_model.SOT_SHARE_2H}.get(c.half, 1.0)
        return props_model.p_both_teams_sot_ge(gm.lam_h, gm.lam_a, n, share=share), \
            "base_rate", f"both>={n} SoT negbin H{c.half or 'full'}", {}
    if k == "half_sot_compare":
        return props_model.p_a_more_sot_than_b(gm.team_lambda(is_home),
                                               gm.team_lambda(not is_home),
                                               share=props_model.SOT_SHARE_2H), \
            "base_rate", "2H SoT cmp negbin", {}

    # ---------- players (lineup/squad-aware) ----------
    if k in ("player_scorer", "player_involve", "player_sot", "player_sot_half"):
        # Market-tier props (Pinnacle-anchored scorer; cross-book consensus SoT)
        # supersede the model near kickoff — they carry career priors the WC-only
        # model can't see (e.g. that a hot 4-in-2 streak is unsustainable).
        sot_mkt1 = None  # a 1-book SoT line: too thin to trust outright, blended w/ model below
        if home and away and c.player and mo is not None and mo.commence_ts \
                and 0 <= (mo.commence_ts - time.time()) <= config.PROP_WINDOW_MIN * 60:
            if k == "player_sot":
                qm = odds_mod.sot_market(home, away, c.player, c.threshold or 1)
                if qm is not None and qm[1] >= 2:  # clean cross-book consensus -> trust the line
                    nlbl = "" if (c.threshold or 1) == 1 else "≥%d" % c.threshold
                    return qm[0], "market", f"odds SoT{nlbl} ({qm[1]}bk)", {"player": c.player}
                if qm is not None and qm[1] == 1:  # thin 1-sided line -> blend 50/50 w/ model below
                    sot_mkt1 = qm[0]
            elif k in ("player_scorer", "player_involve"):
                sm = odds_mod.scorer_market(home, away, c.player)
                if sm is not None:
                    p_s, nb = sm
                    if k == "player_scorer":
                        return p_s, "market", f"odds scorer ({nb}bk)", {"player": c.player}
                    am = odds_mod.assist_market(home, away, c.player)  # involve = score OR assist
                    a = am if am is not None else min(0.25, 0.5 * p_s)
                    oa_inv = min(0.92, 1.0 - (1.0 - p_s) * (1.0 - a))
                    from . import bet365  # Bet365 'score or assist' as a 2nd book
                    b_inv = bet365.involve_prob(home, away, c.player)
                    if b_inv is not None:
                        return (oa_inv + b_inv) / 2.0, "market", "involve oa+b365", {"player": c.player}
                    return oa_inv, "market", f"odds involve ({nb}bk)", {"player": c.player}
        info = stats.player_info(c.player, home, away) if (home and away and c.player) else None
        if info:
            side = info["side"]
            lam = gm.team_lambda(side == "home")
            pos, starts = info["pos"], info["starts"]
            # SoT / scorer / involve now price off real per-player this-WC rates
            # (Pro tier, EB-shrunk; stats.player_*_rate), falling back to the
            # position share x market-anchored team lambda when a player has no
            # accumulated data, plus the reliable lineup start/bench signal.
            tag = ("lineup" if starts is not None else "squad") + f":{side}/{pos}" + \
                  ("/bench" if starts is False else "")
            if k == "player_scorer":
                gr = stats.player_goal_rate(c.player, pos, home, away)
                return model.player_goal_prob(lam, pos, starts, goal_per90=gr), "base_rate", \
                    f"scorer[{tag}{'·r' if gr else ''}]", {"player": c.player}
            if k == "player_involve":
                ir = stats.player_involve_rate(c.player, pos, home, away)
                return model.player_involve_prob(lam, pos, starts, involve_per90=ir), "base_rate", \
                    f"involve[{tag}{'·r' if ir else ''}]", {"player": c.player}
            if k == "player_sot":
                rate = stats.player_sot_rate(c.player, pos, home, away)
                pm = model.player_sot_prob(lam, pos, starts, sot_per90=rate, n=c.threshold or 1)
                if sot_mkt1 is not None:  # lean on the sharp 1-book line (career priors)
                    w = config.SOT_1BK_LINE_W
                    return w * sot_mkt1 + (1.0 - w) * pm, "market", \
                        f"SoT≥{c.threshold or 1} 1bk+model", {"player": c.player}
                return pm, "base_rate", \
                    f"SoT[{tag}{'·r' if rate else ''}{'' if (c.threshold or 1) == 1 else '≥%d' % c.threshold}]", {"player": c.player}
            rate = stats.player_sot_rate(c.player, pos, home, away)
            return model.player_sot_prob(lam, pos, starts, second_half=True, sot_per90=rate, n=c.threshold or 1), \
                "base_rate", f"2H SoT[{tag}{'·r' if rate else ''}]", {"player": c.player}
        # fallback: unresolved player. The per-player rates are name-keyed
        # (player_stats.json covers everyone who has featured this WC), so they
        # still apply when player_info can't resolve squad/side — the common case
        # for matches outside the near-KO prefetch window. Attacker prior + the
        # match-average team environment; flat base only if the player is unknown.
        if k == "player_scorer":
            gr = stats.player_goal_rate(c.player, "att", home, away)
            if gr:
                return model.player_goal_prob(avg_lam, "att", None, goal_per90=gr), "base_rate", \
                    "scorer[name·r]", {"player": c.player}
            # Unknown side: average the per-team probabilities rather than pricing
            # off the mean lambda. f(mean) over-credits a concave f in lopsided games
            # (e.g. an unresolved JOR player gets ARG's goals folded into the avg). 2026-06-27.
            p = 0.5 * (model.prob_at_least_one(gm.lam_h * 0.34) + model.prob_at_least_one(gm.lam_a * 0.34))
            return p, "base_rate", "scorer[base]", {"player": c.player}
        if k == "player_involve":
            ir = stats.player_involve_rate(c.player, "att", home, away)
            if ir:
                return model.player_involve_prob(avg_lam, "att", None, involve_per90=ir), "base_rate", \
                    "involve[name·r]", {"player": c.player}
            # Same Jensen correction as scorer[base]: avg the per-team involve probs.
            p = 0.5 * (model.prob_at_least_one(gm.lam_h * 0.5) + model.prob_at_least_one(gm.lam_a * 0.5))
            return p, "base_rate", "involve[base]", {"player": c.player}
        # SoT: the per-player rate is name-keyed (player_stats.json covers every
        # player who has featured this WC), so it still applies when player_info
        # can't resolve squad/side — which is the common case for matches outside
        # the near-KO squad-prefetch window. Default to an attacker prior + the
        # match-average team environment; flat base only if the player is unknown.
        if k in ("player_sot", "player_sot_half"):
            half = (k == "player_sot_half")
            rate = stats.player_sot_rate(c.player, "att", home, away)
            if rate:
                thr = c.threshold or 1
                pm = model.player_sot_prob(avg_lam, "att", None, second_half=half, sot_per90=rate, n=thr)
                if sot_mkt1 is not None and not half:  # lean on the sharp 1-book line
                    w = config.SOT_1BK_LINE_W
                    return w * sot_mkt1 + (1.0 - w) * pm, "market", \
                        f"SoT≥{thr} 1bk+model", {"player": c.player}
                tag = ("2H " if half else "") + "SoT[name·r" + ("≥%d" % thr if thr > 1 else "") + "]"
                return pm, "base_rate", tag, {"player": c.player}
            base = model.BASE["player_sot_2h"] if half else model.BASE["player_sot_full"]
            return base, "base_rate", ("2H " if half else "") + "SoT[base]", {"player": c.player}
        return model.BASE["player_sot_2h"], "base_rate", "2H SoT[base]", {"player": c.player}

    # ---------- corners ----------
    if k in ("team_corners", "total_corners", "corners_compare"):
        sc = HALF["corners1"] if c.half == 1 else (HALF["corners2"] if c.half == 2 else 1.0)
        if k == "team_corners":
            # Prefer the Bet365 sharp TOTAL-corners line, split by a dominance-tilted share:
            # raw for-rates under-credit a favorite's corners (it camps in the opp half), so
            # tilt the share toward the goal-lambda share. Full-match only, inside the window.
            if (c.half is None and home and away and mo is not None and mo.commence_ts
                    and 0 <= (mo.commence_ts - time.time()) <= config.PROP_WINDOW_MIN * 60):
                from . import bet365
                lam_tot = bet365.corners_market_lambda(home, away)
                if lam_tot is not None:
                    trh, tra = stats.team_rate(home), stats.team_rate(away)
                    cf_me = (trh or {}).get("corners", 0.0)
                    cf_opp = (tra or {}).get("corners", 0.0)
                    if c.side == "away":
                        cf_me, cf_opp = cf_opp, cf_me
                    raw_share = cf_me / (cf_me + cf_opp) if (cf_me + cf_opp) > 0 else 0.5
                    lam_me = gm.team_lambda(c.side == "home")
                    lam_opp = gm.team_lambda(c.side != "home")
                    goal_share = lam_me / (lam_me + lam_opp) if (lam_me + lam_opp) > 0 else 0.5
                    share = raw_share + config.CORNER_DOMINANCE_TILT * (goal_share - raw_share)
                    # cap .68: reverted from a brief .74 escalation (see config.CORNER_DOMINANCE_TILT)
                    # — the un-clip assumed extreme favorites always dominate corners, which the open
                    # ENG/COD game falsified. Re-derive from captured per-game corner RBP, not a story.
                    share = max(0.35, min(0.68, share))
                    tc = lam_tot * share
                    return model.prob_count_ge(tc, c.threshold), "market", \
                        f"bet365 corners λ{lam_tot:.1f}×{share:.0%}", {}
            mean, src = team_mean(c.side, "corners", "team_corners", scale=sc)
            return model.prob_count_ge(mean, c.threshold), "base_rate", f"corners {mean:.2f}[{src}]", {}
        if k == "total_corners":
            # Bet365 quotes a banded total-corners line — sharper than the model on
            # this otherwise line-less seam. Full-match only, inside the prop window.
            if (c.half is None and home and away and mo is not None and mo.commence_ts
                    and 0 <= (mo.commence_ts - time.time()) <= config.PROP_WINDOW_MIN * 60):
                from . import bet365
                bp = bet365.corners_total_prob(home, away, c.threshold)
                if bp is not None:
                    return bp, "market", f"bet365 corners≥{c.threshold}", {}
            mean, src = total_mean("corners", "total_corners", scale=sc)
            return model.prob_count_ge(mean, c.threshold), "base_rate", f"total corners[{src}]", {}
        other = "away" if is_home else "home"
        ma, src = adj_mean(c.side, "corners", "team_corners", scale=sc)
        mb, _ = adj_mean(other, "corners", "team_corners", scale=sc)
        # No demonstrated edge on corner dominance (backtests worse than a
        # coinflip — our confident calls fail), so shrink toward 0.5. See config.
        p = model.prob_a_more_than_b(ma, mb)
        p = 0.5 + config.CORNERS_COMPARE_SHRINK * (p - 0.5)
        return p, "base_rate", f"corner cmp[{src}]", {}

    # ---------- cards / fouls ----------
    if k == "team_cards":
        sc = {1: HALF["cards1"], 2: HALF["cards2"]}.get(c.half, 1.0)
        mean, src = team_mean(c.side, "cards", "team_cards", tilt="foul", scale=sc)
        return model.prob_count_ge(mean, c.threshold), "base_rate", f"cards {mean:.2f}[{src}]", {}
    if k == "total_cards":
        sc = HALF["cards2"] if c.half == 2 else 1.0
        # Bet365 prices KO-game card intensity the group-calibrated model under-rates;
        # anchor the Poisson card-rate to its line (full-match only, in the prop window).
        if (c.half is None and home and away and mo is not None and mo.commence_ts
                and 0 <= (mo.commence_ts - time.time()) <= config.PROP_WINDOW_MIN * 60):
            from . import bet365
            lam_mkt = bet365.cards_market_lambda(home, away)
            if lam_mkt is not None:
                return model.prob_count_ge(lam_mkt, c.threshold), "market", \
                    f"bet365 cards λ{lam_mkt:.1f}", {}
        mean, src = total_mean("cards", "total_cards", scale=sc)
        # Absolute card line: tilt by the assigned referee's EB-shrunk card rate
        # (a strict ref cards both teams). Only here — the compare markets below
        # cancel referee strictness, so they're left alone.
        ref = stats.match_referee(home, away) if (home and away) else None
        rr = stats.referee_rate(ref) if ref else None
        if rr:
            mean *= rr["cards_mult"]
            src = f"{src},ref×{rr['cards_mult']:.2f}"
        return model.prob_count_ge(mean, c.threshold), "base_rate", f"total cards[{src}]", {}
    if k == "cards_compare":
        other = "away" if is_home else "home"
        ma, src = adj_mean(c.side, "cards", "team_cards", tilt="foul")
        mb, _ = adj_mean(other, "cards", "team_cards", tilt="foul")
        return model.prob_a_more_than_b(ma, mb), "base_rate", f"cards cmp[{src}]", {}
    if k == "fouls_compare":
        # NB (2026-06-26): tried blending toward a gentle possession prior to
        # tame the URU/ESP outlier (we 0.16 vs crowd 0.58). Ground-truth backtest
        # on 15 settled foul markets REJECTED it — the data model already beats
        # base-rate pricing (brier 0.190 vs 0.249) via confident-correct calls on
        # clear favourites (GHA>ENG 0.94→YES, PAN>CRO 0.93→YES, COL>COD 0.06→no);
        # a 0.60-capped prior crushes exactly those. Kept the original. The real
        # over-confident leaks are offsides-2+/cards_total — see
        # [[jumpcup-crowd-divergence-finding]] — validate per-market before touching.
        other = "away" if is_home else "home"
        ma, src = adj_mean(c.side, "fouls", "team_fouls", tilt="foul")
        mb, _ = adj_mean(other, "fouls", "team_fouls", tilt="foul")
        return model.prob_a_more_than_b(ma, mb), "base_rate", f"fouls cmp[{src}]", {}

    # ---------- discipline events ----------
    if k in ("pen_or_red", "pen_only"):
        # Anchor the penalty component to Bet365's 'a penalty in the match' line when
        # it's posted (near KO); else the empirical WC base. Red stays the model rate.
        p_pen, src = model.BASE["pen_awarded_match"], "base"
        if (home and away and mo is not None and mo.commence_ts
                and 0 <= (mo.commence_ts - time.time()) <= config.PROP_WINDOW_MIN * 60):
            from . import bet365
            bp = bet365.penalty_prob(home, away)
            if bp is not None:
                p_pen, src = bp, "bet365"
        tier = "market" if src == "bet365" else "base_rate"
        if k == "pen_only":
            return p_pen, tier, f"penalty {src}", {}
        return model.prob_pen_or_red(p_pen=p_pen), tier, f"pen-or-red {src}", {}

    return None


def _fav_volume_flag(c: Classification, home: Optional[str], away: Optional[str],
                     mo) -> bool:
    """True when this market prices an EXTREME favorite's volume stat (team SoT /
    team corners / a favorite player's SoT). Measured across ENG/COD, GER/PAR and
    ESP/AUT (n=9, 2026-07-02): these run HOT — the field beat our fades 6/9 and our
    3 wins all came from one upset game. Counterfactual for pulling 0.75 to field
    when we price BELOW it: +104.5 RBP. Even-game fades (BEL/USA, Mané/Sarr class)
    are untouched — the flag needs win prob >= 0.70."""
    if mo is None or not (home and away) or c.kind not in ("team_sot", "team_corners", "player_sot"):
        return False
    if c.kind == "player_sot":
        info = stats.player_info(c.player, home, away) if c.player else None
        side = info["side"] if info else None
    else:
        side = c.side
    if side not in ("home", "away"):
        return False
    p_win = mo.p_home if side == "home" else mo.p_away
    p_opp = mo.p_away if side == "home" else mo.p_home
    # Trigger on EITHER a dominant favorite (>=.70) OR an overmatched opponent
    # (<=.12): the mechanism is "siege game script inflates the favorite's volume",
    # and COL/GHA (COL .67 but GHA .11) is the same siege as ESP/AUT (.74/.087).
    # POR/CRO (.58/.17) and SUI/ALG stay out — competitive games, fades still win.
    return (p_win is not None and p_win >= 0.70) or (p_opp is not None and p_opp <= 0.12)


_overrides_cache: dict = {"mtime": None, "data": {}}


def _overrides() -> dict:
    """data/overrides.json: {market_id: {prob, note, expires_ts}} — user conviction
    calls that outrank every pricing layer. Reloaded on mtime change; entries past
    expires_ts are ignored (a lock-time expiry keeps stale calls from leaking into
    re-listed or future markets)."""
    path = config.DATA_DIR / "overrides.json"
    try:
        mt = path.stat().st_mtime
        if _overrides_cache["mtime"] != mt:
            with open(path) as f:
                _overrides_cache.update(mtime=mt, data=json.load(f))
    except FileNotFoundError:
        _overrides_cache.update(mtime=None, data={})
    except Exception as exc:  # noqa: BLE001
        log.warning("overrides.json unreadable: %s", exc)
    now = time.time()
    return {k: v for k, v in _overrides_cache["data"].items()
            if not v.get("expires_ts") or v["expires_ts"] > now}


def price_market(m: dict, payload: Optional[dict], outrights: Optional[Dict[str, float]]) -> Decision:
    """Classify + price a single market into a Decision (action set later)."""
    question = m.get("question", "")
    match_name = m.get("match_name", "")
    mid = m["market_id"]
    lobby = m.get("lobby_id", config.LOBBY_ID)
    home, away = parse_match_name(match_name)

    c = classify(question, match_name)
    if not c.mapped:
        # Never forfeit a market. Sitting AT the field average has guaranteed
        # non-negative expected RBP (crowd brier = MEAN of individual briers, so
        # E[RBP at crowd mean] = Var_i(crowd) >= 0). Flat 0.5 if field invisible.
        fld = None
        try:
            fld = crowd_guard._crowd_for_match(  # noqa: SLF001
                match_name, m.get("match_id")).get(mid)
        except Exception as exc:  # noqa: BLE001
            log.debug("field lookup for unmapped failed: %s", exc)
        p = config.clamp_prob(fld if fld is not None else 0.5)
        src = f"field-avg{p:.2f}" if fld is not None else "flat0.5"
        # User conviction overrides outrank the field-sit too (found 2026-07-06:
        # winston-mirror copies on unmapped markets silently never submitted —
        # this early return bypassed the override hook at the end of the flow).
        ov = _overrides().get(mid)
        if ov is not None:
            p = config.clamp_prob(float(ov["prob"]))
            src = f"user-override({src})"
        return Decision(mid, lobby, question, match_name, "unmapped", "none",
                        p, config.to_api_int(p), "submit", c.reason, src)

    mo = None
    if home and away:
        try:
            mo = odds_mod.match_odds_for(home, away, payload)
        except Exception as exc:  # noqa: BLE001
            log.warning("odds lookup failed for %s: %s", match_name, exc)
    gm = _build_goals_model(home, away, mo)

    try:
        priced = _price(c, home, away, mo, gm, outrights)
    except Exception as exc:  # noqa: BLE001
        log.warning("pricing raised for %s (%s): %s", question[:50], c.kind, exc)
        return Decision(mid, lobby, question, match_name, c.kind, "none", 0.5, 50,
                        "skip", f"pricing error: {exc}", "error")
    if priced is None:
        return Decision(mid, lobby, question, match_name, c.kind, "none", 0.5, 50,
                        "skip", "no pricer for kind", "none")

    raw, tier, source, inputs = priced
    # No-edge novelty kinds: anchor toward the learned field base-rate (we've measured
    # the crowd beats our blind model there). No-op unless NOVELTY_ANCHOR_ENABLED.
    raw, source = calibration.novelty_anchor(c.kind, raw, source, c.window, c.event)
    # Shrink toward 0.5 by confidence tier, but LOOSEN it on kinds the RBP data proves we
    # beat the field on AND are well-calibrated on (inverse guardrail / blowout amplifier).
    factor = calibration.edge_boost_factor(c.kind, config.shrink_factor(c.kind, tier))
    shrunk = 0.5 + factor * (raw - 0.5)
    if abs(factor - config.shrink_factor(c.kind, tier)) > 1e-9:
        source = f"{source} edge+"
    bias = calibration.bias_for(c.kind)  # learned recalibration from settled results
    # Suppress the prop-group bias on market-tier prices: a sharp de-vigged line is
    # already efficient, but the prop bias was learned from group-stage MODEL prices
    # (props had no lines then), so applying it nudges our best prices off the line.
    # Anchored kinds keep their market-learned correction; model/base_rate tiers keep
    # the full bias (the model still needs it). Self-heals as market props settle.
    # DERIVED market-tier kinds keep the learned bias: their "market" price is a
    # transform of a line (bet365 cards-lambda -> threshold), not a direct quote,
    # and the transform carries a measured WC-level error the bias corrects.
    # total_cards (2026-07-02): we price .30 avg, outcomes realize .20 over n=10 —
    # the club-calibrated card lambdas run high all tournament; pressing the fade
    # is +14/+22/+31/+35 RBP at +3/5/8/10pp counterfactually, yet the learned
    # -0.05 bias was suppressed here. Direct lines (1X2, totals, player quotes)
    # stay bias-free — they ARE the truth tier.
    if tier == "market" and not calibration.is_anchored(c.kind) \
            and c.kind not in config.BIAS_ON_DERIVED_MARKET:
        bias = 0.0
    final = config.clamp_prob(shrunk + bias)
    if abs(bias) > 0.005:
        source = f"{source} cal{bias:+.2f}"
    # Game-theory guardrail: a big divergence from the live field average on a kind
    # with no measured edge is (per the settled-RBP data) usually US being wrong —
    # pull halfway to the field. Proven-edge kinds are exempt inside apply().
    try:
        guarded, gtag = crowd_guard.apply(c.kind, mid, match_name, final,
                                          tier=tier, threshold=c.threshold,
                                          match_id=m.get("match_id"),
                                          fav_volume=_fav_volume_flag(c, home, away, mo))
        if gtag:
            final = config.clamp_prob(guarded)
            source = f"{source} {gtag}"
    except Exception as exc:  # noqa: BLE001 - guard must never block pricing
        log.debug("crowd guard skipped: %s", exc)
    # User conviction overrides (data/overrides.json) apply LAST — the user's call
    # is final over every layer. Per-market, expire at lock.
    ov = _overrides().get(mid)
    if ov is not None:
        final = config.clamp_prob(float(ov["prob"]))
        source = f"user-override({source})"
    return Decision(
        mid, lobby, question, match_name, c.kind, tier, final, config.to_api_int(final),
        action="submit", reason="", source=source, raw_prob=raw, inputs=inputs,
    )


def _apply_action(d: Decision, st: Optional[state.MarketState], now: float,
                  is_focus: bool = True) -> Decision:
    """Set submit/update/hold based on prior state + bounded-move cap.

    Non-focus matches are floored once (coverage) then held — only the closest
    unstarted (focus) match is actively re-priced and updated.
    """
    if d.action == "skip":
        return d
    close_ts = st.close_ts if st else None
    in_lock = close_ts is not None and (close_ts - now) <= config.LOCK_WINDOW_MIN * 60
    prev = st.last_prob if st else None
    d.prediction_id = st.prediction_id if st else None

    # All unstarted matches are actively priced/updated (so model + calibration
    # improvements reach the whole upcoming board, not just the focus match). The
    # bounded-move cap below prevents churn; focus only governs odds/lineup fetch.
    if prev is None:
        d.action, d.reason = "submit", "new market"
        return d

    move = d.prob - prev
    if abs(move) < config.MIN_MOVE_TO_RESUBMIT_PP:
        d.action, d.reason = "hold", f"move {move:+.3f} below resubmit threshold"
        return d

    if not in_lock and abs(move) > config.MOVE_CAP_PP:
        capped = prev + math.copysign(config.MOVE_CAP_PP, move)
        d.reason = f"move {move:+.3f} capped to {config.MOVE_CAP_PP:+.3f} (outside lock window)"
        d.prob = config.clamp_prob(capped)
        d.prob_int = config.to_api_int(d.prob)
    else:
        d.reason = f"move {move:+.3f}" + (" (lock window: uncapped)" if in_lock else "")
    d.action = "update"
    return d


def build_decisions(snapshot: Optional[dict] = None, force_odds: bool = False) -> List[Decision]:
    """Price every market in the snapshot and assign actions vs. state.

    force_odds refreshes the odds line even if cached — used near kickoff to
    capture closing-line value.
    """
    if snapshot is None:
        with open(config.MARKETS_SNAPSHOT) as f:
            snapshot = json.load(f)
    stats.reset_cycle_budget()  # re-arm the per-cycle career-prior fetch cap
    markets = snapshot.get("markets", [])
    _now0 = time.time()
    _next_lock = min((m["closing_ts"] for m in markets
                      if m.get("closing_ts") and m["closing_ts"] > _now0), default=None)
    closing = _next_lock is not None and (_next_lock - _now0) <= config.CLOSING_WINDOW_MIN * 60
    payload = odds_mod.fetch_world_cup_odds(force=force_odds,
                                            ttl_min=5.0 if force_odds else 20.0,
                                            hard=closing)
    if closing:
        log.info("closing window: hard-refreshed line (next lock in %.1fm)",
                 (_next_lock - _now0) / 60.0)
    outrights = None
    now = time.time()

    # Directive: only ever actively work the closest match that hasn't kicked
    # off; never touch started matches (their markets are closed anyway). We
    # still floor every open market once (coverage), but re-pricing / CLV
    # updates / odds + lineup refreshes are restricted to the focus match.
    markets = [m for m in markets if (m.get("closing_ts") or now + 1) > now]
    _locks = [(m["closing_ts"], m.get("match_name")) for m in markets if m.get("closing_ts")]
    focus_match = min(_locks)[1] if _locks else None
    if focus_match:
        mins = (min(_locks)[0] - now) / 60.0
        log.info("focus = closest unstarted match: %s (locks in %.0fm; %d open markets)",
                 focus_match, mins, len(markets))

    # Warm the live-stats caches for the focus match only (saves API budget).
    if focus_match:
        h, a = parse_match_name(focus_match)
        lock = min(_locks)[0]
        stats.prefetch([{"home_code": h, "away_code": a, "lock_ts": lock}],
                       now, config.LINEUP_WINDOW_MIN * 60)

    # Register markets in state so close_ts is known for the lock window.
    for m in markets:
        close_ts = m.get("closing_ts")
        if close_ts is None and m.get("closing_time"):
            close_ts = odds_mod._iso_to_epoch(m["closing_time"])
        state.upsert_market(m["market_id"], m.get("lobby_id", config.LOBBY_ID),
                            m.get("question", ""), m.get("match_name", ""), close_ts,
                            match_id=m.get("match_id"))

    states = state.all_states()
    decisions: List[Decision] = []
    for m in markets:
        try:
            d = price_market(m, payload, outrights)
            d = _apply_action(d, states.get(d.market_id), now,
                              is_focus=(m.get("match_name") == focus_match))
        except Exception as exc:  # noqa: BLE001 - never let one market kill the pass
            log.warning("decision failed for %s: %s", m.get("market_id"), exc)
            continue
        decisions.append(d)
    return decisions


def render_table(decisions: List[Decision]) -> str:
    """Human-readable dry-run / audit table."""
    rows = []
    header = f"{'MATCH':<16}{'KIND':<17}{'TIER':<14}{'P':>5}{'INT':>5}  {'ACTION':<7} QUESTION"
    rows.append(header)
    rows.append("-" * len(header))
    by_match: Dict[str, List[Decision]] = {}
    for d in decisions:
        by_match.setdefault(d.match_name, []).append(d)
    n_sub = n_skip = 0
    for match in sorted(by_match):
        for d in by_match[match]:
            if d.action == "skip":
                n_skip += 1
            else:
                n_sub += 1
            p = "--" if d.action == "skip" else f"{d.prob:.2f}"
            i = "--" if d.action == "skip" else f"{d.prob_int}"
            rows.append(f"{d.match_name:<16}{d.kind:<17}{d.tier:<14}{p:>5}{i:>5}  "
                        f"{d.action:<7} {d.question[:46]}")
    rows.append("-" * len(header))
    tier_counts: Dict[str, int] = {}
    for d in decisions:
        if d.action != "skip":
            tier_counts[d.tier] = tier_counts.get(d.tier, 0) + 1
    rows.append(f"TOTAL {len(decisions)} markets | actionable {n_sub} | skipped {n_skip} "
                f"| tiers {tier_counts}")
    return "\n".join(rows)


def decisions_to_json(decisions: List[Decision]) -> dict:
    return {"generated_ts": time.time(),
            "decisions": [asdict(d) for d in decisions]}
