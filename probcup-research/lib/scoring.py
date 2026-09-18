"""Scoring utilities: Brier, log-loss, calibration, and bootstrap significance.

The contest scores each market on Brier (binary). For multi-outcome 1X2 backtests
we use the multiclass Brier (sum of squared errors over outcomes) and multiclass
log-loss, and also a binary Brier for the per-question (yes/no) contest markets.
"""
from __future__ import annotations
import numpy as np

EPS = 1e-12


def brier_binary(p, y):
    """Contest scoring: (p - y)^2 for a yes/no market. p=P(yes), y in {0,1}."""
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    return (p - y) ** 2


def logloss_binary(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier_multi(P, y_idx):
    """Multiclass Brier. P: (N,K) probs, y_idx: (N,) true class index. Returns (N,)."""
    P = np.asarray(P, float)
    N, K = P.shape
    Y = np.zeros((N, K))
    Y[np.arange(N), np.asarray(y_idx, int)] = 1.0
    return np.sum((P - Y) ** 2, axis=1)


def logloss_multi(P, y_idx):
    P = np.clip(np.asarray(P, float), EPS, 1 - EPS)
    N = P.shape[0]
    return -np.log(P[np.arange(N), np.asarray(y_idx, int)])


def summarize(scores):
    s = np.asarray(scores, float)
    return {
        "n": int(s.size),
        "mean": float(s.mean()) if s.size else float("nan"),
        "se": float(s.std(ddof=1) / np.sqrt(s.size)) if s.size > 1 else float("nan"),
    }


def paired_bootstrap(scores_a, scores_b, n_boot=10000, seed=0):
    """Paired bootstrap on the per-item score difference (a - b).
    Negative mean diff => method A has lower (better) score. Returns dict with
    mean diff, 95% CI, and one-sided p that A<=B is not better (P(diff>=0))."""
    a = np.asarray(scores_a, float)
    b = np.asarray(scores_b, float)
    assert a.shape == b.shape, "paired arrays must align"
    d = a - b
    n = d.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    return {
        "mean_diff": float(d.mean()),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "p_A_not_better": float(np.mean(boot >= 0.0)),  # frac of resamples where A no better
        "n": int(n),
    }


def calibration(p, y, bins=10):
    """Reliability table. Returns list of (bin_lo, bin_hi, n, mean_pred, emp_rate)."""
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if m.sum() == 0:
            continue
        out.append((round(float(lo), 3), round(float(hi), 3), int(m.sum()),
                    float(p[m].mean()), float(y[m].mean())))
    return out
