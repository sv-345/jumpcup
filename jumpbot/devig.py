"""Margin removal (de-vigging) for betting odds.

The contest is scored by weighted Brier, a strictly proper rule, so the
score-maximizing submission is the honest de-vigged market estimate. This
module turns a vector of bookmaker decimal odds into fair probabilities.

Two methods, chosen by market arity:

* Shin (1992) for 3-way 1X2. Models a proportion ``z`` of insider traders
  and solves for the fair probabilities that account for it. Root-finds ``z``
  with ``scipy.optimize.brentq`` so the recovered probabilities sum to 1. This
  corrects favorite-longshot bias better than naive normalization.
* Power method for 2-way markets (totals, BTTS, match-winner-as-binary).
  Assumes ``p_i = implied_i ** k`` and solves ``k`` so probabilities sum to 1.
  Also applies a favorite-longshot correction and degrades gracefully.

Every result is self-checked: sums to ~1, all strictly inside (0, 1), ordering
preserved (shortest odds -> highest probability). Failing checks fall back to
plain multiplicative normalization and record a warning rather than emitting a
number we don't trust.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from scipy.optimize import brentq

# Numerical tolerances.
_SUM_TOL = 1e-6
_EPS = 1e-12


@dataclass
class DevigResult:
    """Fair probabilities plus the diagnostics needed to audit them."""

    probs: List[float]
    method: str
    booksum: float  # sum of raw inverse odds; > 1 by the bookmaker margin
    margin: float  # booksum - 1
    param: float  # z (Shin) or k (power); nan for multiplicative
    checks_passed: bool
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "probs": [round(p, 6) for p in self.probs],
            "method": self.method,
            "booksum": round(self.booksum, 6),
            "margin": round(self.margin, 6),
            "param": round(self.param, 6) if self.param == self.param else None,
            "checks_passed": self.checks_passed,
            "warnings": list(self.warnings),
        }


def implied(odds: Sequence[float]) -> List[float]:
    """Raw (gross) inverse-odds implied probabilities; these sum to >1."""
    return [1.0 / o for o in odds]


def booksum(odds: Sequence[float]) -> float:
    return float(sum(1.0 / o for o in odds))


def _validate_odds(odds: Sequence[float]) -> None:
    if len(odds) < 2:
        raise ValueError("need at least two outcomes to de-vig")
    for o in odds:
        if o is None or o <= 1.0:
            raise ValueError(f"decimal odds must be > 1.0, got {o!r}")


def _run_checks(probs: Sequence[float], odds: Sequence[float]) -> List[str]:
    """Return a list of warning strings; empty means all checks passed."""
    warns: List[str] = []
    s = sum(probs)
    if abs(s - 1.0) > 1e-4:
        warns.append(f"probabilities sum to {s:.6f}, not ~1")
    if any(p <= 0.0 or p >= 1.0 for p in probs):
        warns.append("probability outside the open interval (0, 1)")
    # Favorite-longshot ordering: shortest odds must map to highest probability.
    order_by_odds = sorted(range(len(odds)), key=lambda i: odds[i])
    order_by_prob = sorted(range(len(probs)), key=lambda i: -probs[i])
    if order_by_odds != order_by_prob:
        warns.append("ordering not preserved (shortest odds != highest prob)")
    return warns


def multiplicative_devig(odds: Sequence[float]) -> DevigResult:
    """Proportional normalization: divide raw implied probs by the booksum."""
    _validate_odds(odds)
    raw = implied(odds)
    bs = sum(raw)
    probs = [r / bs for r in raw]
    warns = _run_checks(probs, odds)
    return DevigResult(
        probs=probs,
        method="multiplicative",
        booksum=bs,
        margin=bs - 1.0,
        param=float("nan"),
        checks_passed=not warns,
        warnings=warns,
    )


def _shin_prob(r_i: float, bs: float, z: float) -> float:
    """Shin's recovered true probability for one outcome given z."""
    # p_i = (sqrt(z^2 + 4(1-z) r_i^2 / booksum) - z) / (2(1-z))
    inner = z * z + 4.0 * (1.0 - z) * (r_i * r_i) / bs
    return (inner ** 0.5 - z) / (2.0 * (1.0 - z))


def shin_devig(odds: Sequence[float]) -> DevigResult:
    """Shin (1992) margin removal. Falls back to multiplicative on failure."""
    _validate_odds(odds)
    raw = implied(odds)
    bs = sum(raw)

    def excess(z: float) -> float:
        return sum(_shin_prob(r, bs, z) for r in raw) - 1.0

    warns: List[str] = []
    try:
        # excess(0+) = sqrt(booksum) - 1 > 0; excess near 1 is < 0 for a normal
        # book, so a root for z exists in (0, 1). Bracket safely inside.
        lo, hi = _EPS, 1.0 - 1e-7
        f_lo, f_hi = excess(lo), excess(hi)
        if f_lo * f_hi > 0:
            # No sign change -> Shin not applicable; degrade.
            res = multiplicative_devig(odds)
            res.warnings.append("Shin found no z bracket; used multiplicative")
            res.method = "multiplicative(shin-fallback)"
            return res
        z = brentq(excess, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=200)
        probs = [_shin_prob(r, bs, z) for r in raw]
        # Renormalize away the residual root-finder error so the sum is exact.
        s = sum(probs)
        probs = [p / s for p in probs]
        warns = _run_checks(probs, odds)
        if warns:
            res = multiplicative_devig(odds)
            res.warnings.append("Shin checks failed (" + "; ".join(warns) + "); used multiplicative")
            res.method = "multiplicative(shin-fallback)"
            return res
        return DevigResult(
            probs=probs, method="shin", booksum=bs, margin=bs - 1.0,
            param=z, checks_passed=True, warnings=[],
        )
    except Exception as exc:  # noqa: BLE001 - never let a bad book kill the caller
        res = multiplicative_devig(odds)
        res.warnings.append(f"Shin raised {type(exc).__name__}: {exc}; used multiplicative")
        res.method = "multiplicative(shin-fallback)"
        return res


def power_devig(odds: Sequence[float]) -> DevigResult:
    """Power method: p_i = implied_i ** k, solve k so probabilities sum to 1.

    Works for any arity but is the default for 2-way markets. ``k >= 1`` shrinks
    the inflated implied probabilities back onto the simplex.
    """
    _validate_odds(odds)
    raw = implied(odds)  # each in (0, 1), summing to >1
    bs = sum(raw)

    def excess(k: float) -> float:
        return sum(r ** k for r in raw) - 1.0

    warns: List[str] = []
    try:
        # At k=1 sum = booksum > 1 (excess>0); for large k each r^k -> 0
        # (excess<0). Root in (1, hi).
        lo, hi = 1.0, 1.0
        f_lo = excess(lo)
        if f_lo <= 0:
            # Booksum already <= 1 (shouldn't happen for a real book); just normalize.
            res = multiplicative_devig(odds)
            res.method = "multiplicative(power-fallback)"
            res.warnings.append("booksum<=1; used multiplicative")
            return res
        hi = 2.0
        while excess(hi) > 0 and hi < 1e6:
            hi *= 2.0
        k = brentq(excess, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=200)
        probs = [r ** k for r in raw]
        s = sum(probs)
        probs = [p / s for p in probs]
        warns = _run_checks(probs, odds)
        if warns:
            res = multiplicative_devig(odds)
            res.warnings.append("power checks failed (" + "; ".join(warns) + "); used multiplicative")
            res.method = "multiplicative(power-fallback)"
            return res
        return DevigResult(
            probs=probs, method="power", booksum=bs, margin=bs - 1.0,
            param=k, checks_passed=True, warnings=[],
        )
    except Exception as exc:  # noqa: BLE001
        res = multiplicative_devig(odds)
        res.warnings.append(f"power raised {type(exc).__name__}: {exc}; used multiplicative")
        res.method = "multiplicative(power-fallback)"
        return res


def devig(odds: Sequence[float], method: str = "auto") -> DevigResult:
    """Dispatch to the right method.

    ``auto`` picks Shin for 3-way markets and the power method for 2-way.
    """
    _validate_odds(odds)
    if method == "auto":
        method = "shin" if len(odds) == 3 else "power"
    if method == "shin":
        return shin_devig(odds)
    if method == "power":
        return power_devig(odds)
    if method == "multiplicative":
        return multiplicative_devig(odds)
    raise ValueError(f"unknown devig method {method!r}")


def two_way_prob(odds_yes: float, odds_no: float) -> float:
    """Convenience: fair P(yes) for a binary market from its two-sided odds."""
    res = devig([odds_yes, odds_no], method="power")
    return res.probs[0]


if __name__ == "__main__":  # pragma: no cover - quick self-test
    # A typical 1X2 book with ~5% overround.
    r = shin_devig([2.10, 3.40, 3.60])
    print("Shin 1X2:", r.as_dict())
    # A 2-way totals market.
    r2 = power_devig([1.90, 1.95])
    print("Power O/U:", r2.as_dict())
    # Lopsided favorite (tests bracketing + ordering check).
    r3 = shin_devig([1.20, 7.0, 13.0])
    print("Shin heavy fav:", r3.as_dict())
