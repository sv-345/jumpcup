"""Statistical pricing for markets without a clean de-vigged line.

Two layers:

1. A goals model (independent Poisson with a Dixon-Coles low-score correction).
   Given expected goals (lambda_home, lambda_away) it prices every goal-derived
   market: 1X2, totals, BTTS, combos, half-totals. The lambdas come from the
   market itself when we have odds (best), from Elo when we don't, or from team
   scoring rates as a last resort.

2. Count base-rate pricers for props that have no betting line at all (offside
   counts, shots-on-target counts, cards/penalties, scorer/SoT player props).
   These use a Poisson on the relevant per-match mean, taken from team stats
   when available and from documented league base rates otherwise.

All probabilities are returned raw in [0,1]; the engine applies confidence
shrink and clamping. Nothing here submits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import poisson

# --- Documented base rates (per match / per team-match), used only as a last
# resort when we have no team-specific data. Conservative, men's-international
# football ballpark figures. Sources: aggregate WC/Euro shot & discipline data.
BASE = {
    "total_goals": 2.65,        # avg total goals, men's international tournament
    "home_share": 0.55,         # share of supremacy-neutral goals to the nominal home side
    "second_half_goal_share": 0.55,  # ~55% of goals fall in the 2nd half
    # mean offsides/team/match. 1.9->1.6 (2026-06-26) chased the crowd's ~0.475, but 21
    # settled markets prove the crowd was WRONG (realized P(2+)=0.62); 150-game realized mean
    # is 1.70, so -> 1.7 (2026-06-30). NB: negbin/dispersion does NOT lift P(2+) (it raises
    # P(0)); the residual gap is a FAVORITE/attacking-team tilt (same as team_corners) — the
    # leak-probe is testing that hypothesis (don't crank the base past the realized mean).
    "team_offsides": 1.7,
    "team_sot": 4.3,            # mean shots on target per team per match
    "team_sot_2h": 2.2,         # mean shots on target per team in the 2nd half
    "total_sot": 8.6,           # mean total shots on target per match
    "total_sot_2h": 4.4,        # mean total shots on target in the 2nd half
    "total_shots": 24.9,        # mean total shots on+off target per match (SoT ~0.35 of all)
    "pen_awarded_match": 0.15,  # P(penalty awarded) — empirical WC rate (60 fixtures; was 0.28)
    "red_card_match": 0.083,    # P(red card shown) — empirical WC rate (60 fixtures; was 0.13)
    "forward_scores": 0.32,     # P(a starting forward scores)
    "player_sot_2h": 0.45,      # P(an attacking starter has >=1 SoT in 2H)
    "player_sot_full": 0.55,    # P(a featured attacker has >=1 SoT in the match)
    "player_involve": 0.46,     # P(a featured attacker scores or assists)
    # Corners ran high vs reality (2026-06-30): realized 8.81 total/match (4.41/team)
    # over 150 settled team-games; old base 10.2 implied P(9+)=.69 but realized .52 and
    # Bet365 sharp lines ~.41 — our HIGH corner calls all lost (RSA/CAN 79%→no, etc, -42 RBP).
    # Lowered to realized. corners_compare is EB-insulated (shifts both teams equally) + shrunk .5.
    "team_corners": 4.4,        # mean corner kicks per team per match (was 5.0)
    "total_corners": 8.8,       # mean total corner kicks per match (was 10.2)
    "total_corners_2h": 4.7,    # was 5.4 (kept 2h share)
    "team_corners_1h": 2.0,     # was 2.3
    "team_fouls": 11.0,         # mean fouls committed per team per match
    # Cards run far lower than the old base implied. Realized over 66 settled matches:
    # 2.62 total / 1.31 per team (Poisson(2.62) nails P(3+)=.50 P(4+)=.29 P(5+)=.12), and
    # SportsPredict's own settlements agree (we priced "4+ cards" .57 vs .29 actual). Base
    # 4.6/2.3 -> 2.7/1.35 cut backtest brier 1.445->0.928 over 5 card-market types. 2026-06-27.
    "team_cards": 1.35,         # mean cards (yellow+red) per team per match (2.3->1.35)
    "total_cards": 2.7,         # (4.6->2.7; realized 2.62, hedged up slightly)
    "total_cards_2h": 1.53,     # keep 0.565 2H share (cards skew late)
    "team_cards_2h": 0.76,
}

MAX_GOALS = 10


def _poisson_vec(lam: float, n: int = MAX_GOALS) -> np.ndarray:
    return poisson.pmf(np.arange(n + 1), lam)


@dataclass
class GoalsModel:
    """Joint scoreline distribution and all goal-derived market probabilities."""

    lam_h: float
    lam_a: float
    tier: str                  # confidence tier of the lambda source
    rho: float = -0.05         # Dixon-Coles low-score dependence

    def __post_init__(self) -> None:
        h = _poisson_vec(self.lam_h)
        a = _poisson_vec(self.lam_a)
        m = np.outer(h, a)
        # Dixon-Coles adjustment on the four low-score cells.
        lam, mu, rho = self.lam_h, self.lam_a, self.rho
        m[0, 0] *= 1.0 - lam * mu * rho
        m[0, 1] *= 1.0 + lam * rho
        m[1, 0] *= 1.0 + mu * rho
        m[1, 1] *= 1.0 - rho
        m = np.clip(m, 0.0, None)
        self.matrix = m / m.sum()

    # --- 1X2 ---------------------------------------------------------------
    @property
    def p_home_win(self) -> float:
        return float(np.tril(self.matrix, -1).sum())  # home goals > away goals

    @property
    def p_away_win(self) -> float:
        return float(np.triu(self.matrix, 1).sum())

    @property
    def p_draw(self) -> float:
        return float(np.trace(self.matrix))

    # --- totals / btts -----------------------------------------------------
    def p_total_ge(self, n: int) -> float:
        idx = np.add.outer(np.arange(MAX_GOALS + 1), np.arange(MAX_GOALS + 1))
        return float(self.matrix[idx >= n].sum())

    def p_over(self, line: float) -> float:
        # Over a .5 line == total >= ceil(line).
        return self.p_total_ge(math.ceil(line))

    @property
    def p_btts(self) -> float:
        return float(self.matrix[1:, 1:].sum())

    def p_btts_and_total_ge(self, n: int) -> float:
        idx = np.add.outer(np.arange(MAX_GOALS + 1), np.arange(MAX_GOALS + 1))
        mask = np.zeros_like(self.matrix, dtype=bool)
        mask[1:, 1:] = True
        return float(self.matrix[mask & (idx >= n)].sum())

    def team_lambda(self, home: bool) -> float:
        return self.lam_h if home else self.lam_a

    # --- half-time totals --------------------------------------------------
    def _half_share(self, second_half: bool) -> float:
        return BASE["second_half_goal_share"] if second_half else 1 - BASE["second_half_goal_share"]

    def half_team_lambda(self, home: bool, second_half: bool) -> float:
        return self.team_lambda(home) * self._half_share(second_half)

    def p_half_total_ge(self, n: int, second_half: bool = True) -> float:
        lam_half = (self.lam_h + self.lam_a) * self._half_share(second_half)
        return prob_count_ge(lam_half, n)

    def p_team_scores_in_half(self, home: bool, second_half: bool, n: int = 1) -> float:
        return prob_count_ge(self.half_team_lambda(home, second_half), n)

    def p_team_scores(self, home: bool, n: int = 1) -> float:
        """P(team scores at least n goals in the match)."""
        return prob_count_ge(self.team_lambda(home), n)

    def p_ht_draw(self) -> float:
        lh, la = self.half_team_lambda(True, False), self.half_team_lambda(False, False)
        ph, pa = _poisson_vec(lh), _poisson_vec(la)
        return float(sum(ph[k] * pa[k] for k in range(len(ph))))

    def p_ht_lead(self, home: bool) -> float:
        """P(the given side is winning at half-time)."""
        lh, la = self.half_team_lambda(True, False), self.half_team_lambda(False, False)
        return prob_a_more_than_b(lh, la) if home else prob_a_more_than_b(la, lh)

    def p_ever_leads(self, home: bool) -> float:
        """P(the side is strictly ahead at some point during regulation).

        With independent Poisson goals at iid-uniform times, every interleaving
        of a final score (s, b) is equally likely, and the ballot-style count
        gives P(the side's running total never exceeds the opponent's)
        = (b + 1 - s) / (b + 1) for s <= b, and 0 for s > b (the side finishes
        ahead, so it was ahead at least after the final goal)."""
        m = self.matrix if home else self.matrix.T
        p = 0.0
        for s in range(m.shape[0]):
            for b in range(m.shape[1]):
                pr = float(m[s, b])
                if pr <= 0.0 or s == 0:  # never scores -> never leads
                    continue
                never = (b + 1.0 - s) / (b + 1.0) if s <= b else 0.0
                p += pr * (1.0 - never)
        return p

    def p_half_goal_supremacy(self, home: bool, second_half: bool = True) -> float:
        """P(side scores strictly more goals than opponent in the given half)."""
        ls = self.half_team_lambda(home, second_half)
        lo = self.half_team_lambda(not home, second_half)
        return prob_a_more_than_b(ls, lo)

    def p_2h_more_goals_than_1h(self) -> float:
        lam_tot = self.lam_h + self.lam_a
        return prob_a_more_than_b(lam_tot * self._half_share(True),
                                  lam_tot * self._half_share(False))

    def p_first_goal_of_half(self, home: bool, second_half: bool = True) -> float:
        """Approx P(side scores the first goal of the half).

        = P(>=1 goal in the half) * side's share of half goal rate.
        """
        ls = self.half_team_lambda(home, second_half)
        lo = self.half_team_lambda(not home, second_half)
        any_goal = 1.0 - math.exp(-(ls + lo))
        share = ls / (ls + lo) if (ls + lo) > 0 else 0.5
        return any_goal * share

    def p_first_goal_game(self, home: bool) -> float:
        any_goal = 1.0 - math.exp(-(self.lam_h + self.lam_a))
        share = self.lam_h / (self.lam_h + self.lam_a) if (self.lam_h + self.lam_a) > 0 else 0.5
        return any_goal * (share if home else 1 - share)

    def p_advance(self, home: bool) -> float:
        """Knockout 'advance' = win the tie: P(win in 90) plus the draw mass
        resolved by ET/shootout, split by a strength-tilted conditional win share
        (a coin flip for evenly-matched sides, more for the favorite)."""
        pw = self.p_home_win if home else self.p_away_win
        pl = self.p_away_win if home else self.p_home_win
        et_share = pw / (pw + pl) if (pw + pl) > 1e-9 else 0.5
        return pw + self.p_draw * et_share

    def p_win_by(self, home: bool, margin: int) -> float:
        """P(the given side wins by >= margin goals) from the score matrix."""
        n = self.matrix.shape[0]
        diff = np.subtract.outer(np.arange(n), np.arange(n))  # home_goals - away_goals
        return float(self.matrix[diff >= margin].sum() if home
                     else self.matrix[diff <= -margin].sum())


def p_event_in_window(lam_total: float, frac: float) -> float:
    """P(>=1 event) in a sub-window holding ``frac`` of the match's expected
    count (Poisson thinning). For the hydration-break time-segmented markets."""
    return 1.0 - math.exp(-max(0.0, lam_total) * max(0.0, frac))


_HYDRATION_BREAKS = {"before_first": (0, 30), "after_first": (30, 45),
                     "before_second": (45, 75), "after_second": (75, 90)}


def hydration_frac(window: str, share_2h: float) -> float:
    """Fraction of a match metric's expected count falling in a hydration-break
    window. Breaks sit ~30' and ~75'; per-minute intensity is split by half so a
    late window inherits the 2H share (goals/cards skew later)."""
    a, b = _HYDRATION_BREAKS.get(window, (0, 90))
    s2 = max(0.0, min(1.0, share_2h))
    s1 = 1.0 - s2
    m1 = max(0.0, min(45, b) - min(45, a))   # minutes inside the 1st half
    m2 = max(0.0, min(90, b) - max(45, a))   # minutes inside the 2nd half
    return m1 / 45.0 * s1 + m2 / 45.0 * s2


# Share of a team's shots by shooter rank (a fixed concentration profile over
# ~11 shooters, top shooter ~20%). Drives the distinct-shooters occupancy model:
# Poissonized team shots -> per-player P(shot taken) = 1-exp(-lam*w). Sums to 1.
SHOT_WEIGHTS = [0.20, 0.15, 0.13, 0.11, 0.09, 0.08, 0.07, 0.06, 0.05, 0.03, 0.03]


def p_any_brace(lam_total: float, k_eff: float = 22.0) -> float:
    """P(some player scores 2+ goals): a birthday-collision over ``k_eff``
    effective scorers, averaged across a Poisson match-goal count. k_eff is set
    so the base case (lam~2.65) lands near the empirical ~0.13 brace rate; the
    convex shape then extrapolates sensibly to higher-scoring games."""
    p = 0.0
    for g in range(2, 13):
        pg = math.exp(-lam_total) * lam_total ** g / math.factorial(g)
        distinct = 1.0
        for i in range(g):
            distinct *= max(0.0, k_eff - i) / k_eff
        p += pg * (1.0 - distinct)
    return p


def prob_count_ge(mean: float, threshold: int) -> float:
    """P(X >= threshold) for X ~ Poisson(mean)."""
    if threshold <= 0:
        return 1.0
    return float(poisson.sf(threshold - 1, mean))


def prob_at_least_one(rate: float) -> float:
    """P(>=1 event) for a Poisson with given expected count."""
    return float(1.0 - math.exp(-max(rate, 0.0)))


def prob_a_more_than_b(mean_a: float, mean_b: float, n: int = 15) -> float:
    """P(A > B) for independent Poisson counts (e.g. 2H shots on target)."""
    pa = _poisson_vec(mean_a, n)
    pb = _poisson_vec(mean_b, n)
    # P(A > B) = sum_b P(B=b) * P(A > b)
    total = 0.0
    cum_a = np.cumsum(pa)  # cum_a[k] = P(A <= k)
    for b in range(n + 1):
        p_a_gt_b = 1.0 - cum_a[b]  # P(A > b) = 1 - P(A <= b)
        total += pb[b] * p_a_gt_b
    return float(total)


def prob_pen_or_red(p_pen: float = BASE["pen_awarded_match"],
                    p_red: float = BASE["red_card_match"]) -> float:
    """P(penalty awarded OR red card shown), treating them as independent."""
    return 1.0 - (1.0 - p_pen) * (1.0 - p_red)


# --- player props (position + team goal environment + start status) --------
_POS_GOAL_SHARE = {"att": 0.38, "mid": 0.13, "def": 0.05, "gk": 0.005}
_POS_SOT_FULL = {"att": 0.62, "mid": 0.42, "def": 0.20, "gk": 0.02}  # P(>=1 SoT, match)
_BENCH_FACTOR = 0.22  # a confirmed non-starter plays limited minutes


def player_goal_prob(lam_team: float, pos: str, starts: Optional[bool],
                     goal_rate: Optional[float] = None, team_avg: float = 1.45,
                     goal_per90: Optional[float] = None) -> float:
    if goal_per90 is not None and goal_per90 > 0:
        # this-WC goals/90 (EB-shrunk), tilted to the match goal environment
        lam = goal_per90 * max(0.8, min(1.3, lam_team / 1.3))
    elif goal_rate is not None and goal_rate > 0:
        # player's national-team goals/appearance, scaled to this match's goal env
        lam = goal_rate * (max(lam_team, 0.2) / team_avg)
    else:
        lam = max(lam_team, 0.2) * _POS_GOAL_SHARE.get(pos, 0.13)
    p = 1.0 - math.exp(-lam)
    if starts is False:
        p *= _BENCH_FACTOR
    return min(0.97, p)


def player_involve_prob(lam_team: float, pos: str, starts: Optional[bool],
                        goal_rate: Optional[float] = None,
                        involve_per90: Optional[float] = None) -> float:
    # Score OR assist.
    if involve_per90 is not None and involve_per90 > 0:
        # this-WC (goals+assists)/90 (EB-shrunk), tilted to the match goal env
        lam = involve_per90 * max(0.8, min(1.3, lam_team / 1.3))
        p = 1.0 - math.exp(-lam)
        if starts is False:
            p *= _BENCH_FACTOR
        return min(0.90, p)
    # No per-player data: combine the goal prob with a smaller assist prob and
    # saturate: P = 1 - (1-g)(1-0.7g), capped — a single player can't be near
    # certain to be involved in a goal. (Was g*1.7, which overshot to ~0.92.)
    g = player_goal_prob(lam_team, pos, starts, goal_rate)
    return min(0.85, 1.0 - (1.0 - g) * (1.0 - 0.7 * g))


def player_sot_prob(lam_team: float, pos: str, starts: Optional[bool],
                    second_half: bool = False, sot_per90: Optional[float] = None,
                    n: int = 1) -> float:
    n = max(1, int(n or 1))
    if sot_per90 is not None and sot_per90 > 0:
        # Per-player rate (this-WC SoT/90, EB-shrunk) -> P(>=n SoT), mild team tilt.
        scale = max(0.8, min(1.25, lam_team / 1.3))
        rate = sot_per90 * scale * (0.5 if second_half else 1.0)
        p = min(0.95, float(poisson.sf(n - 1, rate)))
    else:
        base = _POS_SOT_FULL.get(pos, 0.42)
        scale = max(0.7, min(1.3, lam_team / 1.3))  # mild team-environment tilt
        p = min(0.95, base * scale)
        if second_half:  # P(>=1 in 2H) given a match rate, Poisson-split
            p = 1.0 - (1.0 - p) ** 0.5
        if n >= 2:  # no per-player rate: P(>=n) from the implied Poisson lambda
            lam = -math.log(max(1e-6, 1.0 - p))
            p = min(0.95, float(poisson.sf(n - 1, lam)))
    if starts is False:
        p *= 0.25
    return p


# --- lambda estimation -----------------------------------------------------

def lambdas_from_market(p_home: float, p_away: float,
                        total_line: Optional[float] = None,
                        p_over: Optional[float] = None) -> Tuple[float, float, str]:
    """Fit (lam_h, lam_a) to reproduce a de-vigged 1X2 (and a total if given).

    Returns (lam_h, lam_a, tier). tier is 'model_anchored'.
    """
    # Initial guess seeded from the market: invert p_over for the total goal
    # mean, take supremacy from the 1X2 log-ratio. A good start matters a lot for
    # skewed (heavy-favorite) books where a naive start lands in a poor local min.
    if total_line is not None and p_over is not None:
        try:
            t0 = brentq(lambda t: float(poisson.sf(math.ceil(total_line) - 1, t)) - p_over,
                        0.3, 8.0)
        except (ValueError, RuntimeError):
            t0 = BASE["total_goals"]
    else:
        t0 = BASE["total_goals"]
    sup0 = max(-3.0, min(3.0, math.log((p_home + 1e-3) / (p_away + 1e-3)) * 0.85))
    start_h, start_a = max(0.1, (t0 + sup0) / 2), max(0.1, (t0 - sup0) / 2)

    def residuals(x: np.ndarray) -> np.ndarray:
        lam_h, lam_a = max(x[0], 1e-2), max(x[1], 1e-2)
        gm = GoalsModel(lam_h, lam_a, tier="tmp")
        # Weight the 1X2 match higher — it drives supremacy and win-consistency.
        res = [(gm.p_home_win - p_home) * 2.0, (gm.p_away_win - p_away) * 2.0]
        if total_line is not None and p_over is not None:
            res.append(gm.p_over(total_line) - p_over)
        return np.array(res)

    sol = least_squares(
        residuals, x0=[start_h, start_a],
        bounds=([1e-2, 1e-2], [6.0, 6.0]), max_nfev=500,
    )
    return float(sol.x[0]), float(sol.x[1]), "model_anchored"


def lambdas_from_elo(elo_h: float, elo_a: float, home_adv: float = 60.0,
                     base_total: float = BASE["total_goals"]) -> Tuple[float, float, str]:
    """Derive (lam_h, lam_a) from Elo ratings.

    Win expectancy gives a supremacy; we split a base total by that supremacy.
    """
    diff = (elo_h + home_adv) - elo_a
    # Expected goal supremacy ~ logistic of Elo diff, scaled to football range.
    # ~400 Elo edge ~ +1.0 goal supremacy is a reasonable, conservative mapping.
    supremacy = 2.0 * (1.0 / (1.0 + 10 ** (-diff / 400.0)) - 0.5) * 2.0
    lam_h = max(0.2, base_total / 2.0 + supremacy / 2.0)
    lam_a = max(0.2, base_total / 2.0 - supremacy / 2.0)
    return lam_h, lam_a, "model_elo"


def lambdas_from_rates(gf_h: float, ga_h: float, gf_a: float, ga_a: float,
                       league_avg: float = BASE["total_goals"] / 2) -> Tuple[float, float, str]:
    """Dixon-Coles style attack/defense from goals-for/against per game.

    gf/ga are per-match goals for/against for each team. Returns lambdas via the
    classic attack*defense/league_avg construction.
    """
    la = max(league_avg, 1e-3)
    att_h, def_h = gf_h / la, ga_h / la
    att_a, def_a = gf_a / la, ga_a / la
    lam_h = max(0.2, att_h * def_a * la)
    lam_a = max(0.2, att_a * def_h * la)
    return lam_h, lam_a, "base_rate"


if __name__ == "__main__":  # pragma: no cover
    lam_h, lam_a, tier = lambdas_from_market(0.55, 0.20, total_line=2.5, p_over=0.52)
    gm = GoalsModel(lam_h, lam_a, tier)
    print(f"lambdas=({lam_h:.2f},{lam_a:.2f}) tier={tier}")
    print(f"1X2 = {gm.p_home_win:.3f}/{gm.p_draw:.3f}/{gm.p_away_win:.3f}")
    print(f"over2.5={gm.p_over(2.5):.3f} btts={gm.p_btts:.3f} btts&3+={gm.p_btts_and_total_ge(3):.3f}")
    print(f"2H 2+ goals={gm.p_half_total_ge(2):.3f}")
    print(f"team offside 2+={prob_count_ge(BASE['team_offsides'],2):.3f}")
    print(f"team SoT 4+={prob_count_ge(BASE['team_sot'],4):.3f}")
    print(f"pen-or-red={prob_pen_or_red():.3f}")
    print(f"2H A>B SoT={prob_a_more_than_b(2.4,1.9):.3f}")
