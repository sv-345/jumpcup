"""De-vig (overround removal) methods for converting bookmaker odds -> fair probabilities.

Implements the family the brief asks to compare — Shin / power / log(odds-ratio) /
additive — plus multiplicative (the live floor's fallback) as a baseline.

All functions take decimal odds (list/array, length n>=2) and return fair probs
summing to 1. Pure functions, no I/O, no network. Mirrors the algebra in
jumpbot/devig.py for Shin/power so candidate-1 comparisons are apples-to-apples,
and adds the methods the live bot lacks (additive, odds-ratio/log).

References:
  Shin (1992,1993); Strumbelj (2014) "On determining probability forecasts from betting odds".
  Power method: Clarke, Kovalchik et al.
  Odds-ratio ("log"): Cheung / Buchdahl "Wisdom of the Crowd" (constant odds-ratio transform).
  Additive: subtract equal share of the margin from each implied probability.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq

EPS = 1e-9


def implied(odds):
    """Raw implied probabilities q_i = 1/o_i (sum = booksum >= 1)."""
    o = np.asarray(odds, dtype=float)
    return 1.0 / o


def booksum(odds):
    return float(np.sum(implied(odds)))


def overround(odds):
    """Margin = booksum - 1 (the vig)."""
    return booksum(odds) - 1.0


# ---------------------------------------------------------------------------
# 1. Multiplicative / basic (normalize). Live bot's universal fallback.
# ---------------------------------------------------------------------------
def multiplicative(odds):
    q = implied(odds)
    return q / q.sum()


# ---------------------------------------------------------------------------
# 2. Additive: p_i = q_i - margin/n  (then renormalize if any clipped).
# ---------------------------------------------------------------------------
def additive(odds):
    q = implied(odds)
    n = len(q)
    margin = q.sum() - 1.0
    p = q - margin / n
    p = np.clip(p, EPS, None)
    return p / p.sum()


# ---------------------------------------------------------------------------
# 3. Power: p_i = q_i^k, solve k so sum = 1.
# ---------------------------------------------------------------------------
def power(odds):
    q = implied(odds)
    if q.sum() <= 1.0 + EPS:
        return q / q.sum()

    def f(k):
        return np.sum(q ** k) - 1.0

    # booksum>1 => need k>1 to shrink. Bracket generously.
    try:
        k = brentq(f, 1.0, 50.0, xtol=1e-10)
    except ValueError:
        return multiplicative(odds)
    return q ** k


# ---------------------------------------------------------------------------
# 4. Shin: insider-trader proportion z. Closed form for p_i given z, solve z.
#    p_i = [sqrt(z^2 + 4(1-z) q_i^2 / Sigma) - z] / (2(1-z)),  Sigma = sum q.
# ---------------------------------------------------------------------------
def _shin_probs(q, z):
    Sigma = q.sum()
    inner = z * z + 4.0 * (1.0 - z) * (q * q) / Sigma
    return (np.sqrt(inner) - z) / (2.0 * (1.0 - z))


def shin(odds):
    q = implied(odds)
    if q.sum() <= 1.0 + EPS:
        return q / q.sum()

    def g(z):
        return np.sum(_shin_probs(q, z)) - 1.0

    try:
        z = brentq(g, 0.0, 0.5, xtol=1e-10)
    except ValueError:
        return multiplicative(odds)
    p = _shin_probs(q, z)
    p = np.clip(p, EPS, None)
    return p / p.sum()


# ---------------------------------------------------------------------------
# 5. Odds-ratio / "log" (Cheung): constant odds-ratio c maps book->fair.
#    p_i = c*q_i / (1 - q_i + c*q_i), solve c so sum p_i = 1.  c<1 when booksum>1.
# ---------------------------------------------------------------------------
def odds_ratio(odds):
    q = implied(odds)
    q = np.clip(q, EPS, 1 - EPS)
    if q.sum() <= 1.0 + EPS:
        return q / q.sum()

    def f(c):
        p = c * q / (1.0 - q + c * q)
        return p.sum() - 1.0

    try:
        c = brentq(f, 1e-6, 1.0, xtol=1e-12)
    except ValueError:
        return multiplicative(odds)
    p = c * q / (1.0 - q + c * q)
    return p / p.sum()


METHODS = {
    "multiplicative": multiplicative,
    "additive": additive,
    "power": power,
    "shin": shin,
    "log": odds_ratio,   # odds-ratio / logarithmic family
}


def devig(odds, method="shin"):
    """Dispatch. Auto: shin for 3-way, power for 2-way (matches live bot)."""
    if method == "auto":
        method = "shin" if len(odds) == 3 else "power"
    return METHODS[method](odds)


def two_way(odds_yes, odds_no, method="power"):
    """De-vig a two-outcome (yes/no) market -> P(yes)."""
    p = devig([odds_yes, odds_no], method=method)
    return float(p[0])
