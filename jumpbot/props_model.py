"""Validated team shots-on-target model (promoted from probcup-research).

The floor priced team SoT from a flat base rate — Poisson(4.3) x a clamped
strength tilt — for every team regardless of opponent. This module instead maps
the market-implied per-team goal lambda to expected shots-on-target via a linear
map fit and validated out-of-sample (candidate2_sot_vs_floor, 37,810 team-games,
beats the floor by 0.004-0.0075 Brier/market, p~0), then prices the count as a
Negative-Binomial (overdispersion fit from data, unclamped).

    E[team SoT | goal lambda] = SOT_B0 + SOT_B1 * goal_lambda ; NegBin size r

The goal lambda comes from the de-vigged market (model.lambdas_from_market) when
odds are fresh, else the Elo prior — so the SoT rate is opponent-adjusted and
team-specific instead of flat. Pure functions; no I/O.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import nbinom

# Out-of-sample validated (probcup-research/reports/candidate2_sot_vs_floor.md):
SOT_B0 = 1.284
SOT_B1 = 2.197
SOT_NB_R = 33.78
# Half-of-match shot shares (2nd half slightly higher); 1H = 1 - 2H.
SOT_SHARE_2H = 0.52
SOT_SHARE_1H = 0.48


def sot_lambda(goal_lambda: float, share: float = 1.0) -> float:
    """Expected shots-on-target for a team given its market goal lambda."""
    return max(0.05, (SOT_B0 + SOT_B1 * max(goal_lambda, 0.0)) * share)


def _nb_p(lam: float) -> float:
    return SOT_NB_R / (SOT_NB_R + lam)


def p_sot_ge(goal_lambda: float, n: int, share: float = 1.0) -> float:
    """P(team SoT >= n)."""
    lam = sot_lambda(goal_lambda, share)
    return float(nbinom.sf(n - 1, SOT_NB_R, _nb_p(lam)))


def p_total_sot_ge(lam_home: float, lam_away: float, n: int, share: float = 1.0) -> float:
    """P(total SoT >= n). Sum of two independent per-team NegBins, approximated
    as a single NegBin matched to the summed mean with ~doubled size (variance
    adds, so dispersion-per-count roughly halves)."""
    lam = sot_lambda(lam_home, share) + sot_lambda(lam_away, share)
    r = 2.0 * SOT_NB_R
    return float(nbinom.sf(n - 1, r, r / (r + lam)))


def p_a_more_sot_than_b(lam_a_goals: float, lam_b_goals: float, share: float = 1.0) -> float:
    """P(team A SoT > team B SoT), two independent NegBins."""
    la = sot_lambda(lam_a_goals, share)
    lb = sot_lambda(lam_b_goals, share)
    ks = np.arange(0, 40)
    pa = nbinom.pmf(ks, SOT_NB_R, _nb_p(la))
    pb = nbinom.pmf(ks, SOT_NB_R, _nb_p(lb))
    cb = np.cumsum(pb)  # cb[a-1] = P(B <= a-1) = P(B < a)
    return float(sum(pa[a] * (cb[a - 1] if a >= 1 else 0.0) for a in range(len(ks))))


def p_both_teams_sot_ge(lam_home: float, lam_away: float, n: int, share: float = 1.0) -> float:
    """P(both teams SoT >= n), assuming approximate independence."""
    return p_sot_ge(lam_home, n, share) * p_sot_ge(lam_away, n, share)
