"""Team-strength prop models for the contest's no-line markets (shots-on-target,
discipline). The live floor prices these from a STATIC base rate (e.g. SoT lambda=4.3
for every team regardless of opponent), then shrinks toward 0.5. These models instead
make the rate team- and opponent-specific, which matters enormously for the lopsided
group-stage matchups a World Cup is full of.

Ratings use a regularized attack x defense decomposition (shrunk toward league mean
by a pseudocount) to avoid overfitting small samples. Counts are modelled as Poisson
or Negative-Binomial (overdispersion fit from data).
"""
from __future__ import annotations
import numpy as np
from scipy.stats import poisson, nbinom


def fit_team_rates(df, for_col, against_home_col, against_away_col, home_team="HomeTeam",
                   away_team="AwayTeam", k=6.0):
    """Generic attack/defense rate fitter for a per-team match count.

    for_col pair: (home_for, away_for) columns giving the count produced by the home
    and away team respectively. We pool a team's home+away games for its ratings but
    keep separate league home/away baselines.

    Returns dict with league baselines and per-team attack/defense multipliers
    (regularized toward 1.0 with pseudocount k).
    """
    raise NotImplementedError  # specialized fitters below


class SoTModel:
    """Shots-on-target model. Predicts P(team SoT >= n).

    lambda_team = lg_base(side) * attack[team] * defense[opp], where attack/defense
    are league-relative multipliers shrunk toward 1.0. Distribution: Poisson or NegBin.
    """

    def __init__(self, k=6.0, dist="nbinom"):
        self.k = k
        self.dist = dist

    def fit(self, df):
        h = df.copy()
        H_FOR, A_FOR = "HST", "AST"  # home SoT, away SoT
        h = h.dropna(subset=[H_FOR, A_FOR, "HomeTeam", "AwayTeam"])
        h[H_FOR] = h[H_FOR].astype(float); h[A_FOR] = h[A_FOR].astype(float)
        self.lg_home = h[H_FOR].mean()
        self.lg_away = h[A_FOR].mean()
        lg_for = (h[H_FOR].sum() + h[A_FOR].sum()) / (2 * len(h))   # avg SoT-for per team-game
        # per-team: SoT generated (attack) and SoT conceded (defense), pooled home+away
        teams = set(h["HomeTeam"]) | set(h["AwayTeam"])
        att, dfn = {}, {}
        for t in teams:
            sot_for = np.concatenate([h.loc[h.HomeTeam == t, H_FOR].values,
                                      h.loc[h.AwayTeam == t, A_FOR].values])
            sot_against = np.concatenate([h.loc[h.HomeTeam == t, A_FOR].values,
                                          h.loc[h.AwayTeam == t, H_FOR].values])
            # regularized multiplier toward 1.0 (=lg_for) with pseudocount k
            att[t] = (sot_for.sum() + self.k * lg_for) / ((len(sot_for) + self.k) * lg_for)
            dfn[t] = (sot_against.sum() + self.k * lg_for) / ((len(sot_against) + self.k) * lg_for)
        self.att, self.dfn, self.lg_for = att, dfn, lg_for
        # dispersion: fit NegBin size r from pooled team-game counts vs their model means
        means, obs = [], []
        for _, row in h.iterrows():
            lam_h = self.lam(row.HomeTeam, row.AwayTeam, home=True)
            lam_a = self.lam(row.AwayTeam, row.HomeTeam, home=False)
            means += [lam_h, lam_a]; obs += [row[H_FOR], row[A_FOR]]
        means = np.array(means); obs = np.array(obs)
        var = np.mean((obs - means) ** 2)
        mbar = means.mean()
        # NegBin: var = mu + mu^2/r  -> r = mu^2/(var-mu)
        self.r = max(2.0, mbar ** 2 / max(var - mbar, 1e-6)) if var > mbar else 50.0
        return self

    def lam(self, team, opp, home):
        base = self.lg_home if home else self.lg_away
        return base * self.att.get(team, 1.0) * self.dfn.get(opp, 1.0)

    def p_ge(self, team, opp, home, n):
        lam = self.lam(team, opp, home)
        if self.dist == "poisson":
            return float(poisson.sf(n - 1, lam))
        # NegBin parameterized by mean lam and size r: p = r/(r+lam)
        p = self.r / (self.r + lam)
        return float(nbinom.sf(n - 1, self.r, p))


class StaticBaseRate:
    """Mimics the live floor: fixed Poisson(base) for every team, optional shrink."""

    def __init__(self, base=4.3, shrink=1.0):
        self.base = base; self.shrink = shrink

    def p_ge(self, team, opp, home, n):
        p = float(poisson.sf(n - 1, self.base))
        return 0.5 + self.shrink * (p - 0.5)
