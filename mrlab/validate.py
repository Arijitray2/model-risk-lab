"""
validate.py — The model validator's toolkit.

This module treats the desk's model the way a bank's model risk management
(MRM) function would treat any production model: as a hypothesis to be
tested against realised outcomes.

1. Kupiec (1995) proportion-of-failures test
--------------------------------------------
A VaR_alpha model predicts that losses exceed VaR on a fraction alpha of
days. Observing x exceptions in n days, the likelihood-ratio statistic

    LR_pof = -2 ln[ (1-alpha)^{n-x} alpha^x ]
             +2 ln[ (1-x/n)^{n-x} (x/n)^x ]

is asymptotically chi-squared(1) under H0: correct coverage. Rejection
means the model's tail is mis-sized (too many or too few exceptions).

2. Christoffersen (1998) independence & conditional coverage
------------------------------------------------------------
Correct *unconditional* coverage is not enough — exceptions must also be
independent (not clustered). With n_ij = # transitions from state i to j
(state 1 = exception), and pi_i = P(exception | state i):

    LR_ind = -2 ln[ (1-pi)^{n00+n10} pi^{n01+n11} ]
             +2 ln[ (1-pi0)^{n00} pi0^{n01} (1-pi1)^{n10} pi1^{n11} ]

chi-squared(1) under independence; LR_cc = LR_pof + LR_ind is
chi-squared(2). Clustered exceptions are the signature of a model missing
volatility dynamics or jumps.

3. Bootstrap inference on realised edge
---------------------------------------
"Is this desk's average edge per contract genuinely positive?" We resample
per-seed P/L with replacement (the seeds are iid by construction) to get a
percentile confidence interval, plus a one-sample t-test for H0: mean = 0.

4. Sequential monitoring (CUSUM)
--------------------------------
Validators do not wait a year to find a broken model. We monitor the
per-trade gap between realised P/L and the model-implied expectation
(the half-spread) with a one-sided CUSUM:

    C_0 = 0,   C_t = max(0, C_{t-1} + (m - X_t) - k)

where X_t is the realised edge of trade t, m the promised edge and k an
allowance (half a standard deviation). An alarm at threshold h flags
systematic under-delivery long before the equity curve makes it obvious.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2, ttest_1samp


# ---------------------------------------------------------------------------
# VaR backtests
# ---------------------------------------------------------------------------

def kupiec_pof(exceptions: np.ndarray, alpha: float) -> dict:
    """Kupiec proportion-of-failures test.

    exceptions : boolean array, True on periods where loss > VaR.
    """
    I = np.asarray(exceptions).astype(int)
    n, x = I.size, int(I.sum())
    pi_hat = x / n if n else 0.0
    if x in (0, n):
        # Degenerate MLE: handle by continuity (0 log 0 = 0).
        ll1 = 0.0
    else:
        ll1 = (n - x) * np.log(1 - pi_hat) + x * np.log(pi_hat)
    ll0 = (n - x) * np.log(1 - alpha) + x * np.log(alpha)
    lr = -2.0 * (ll0 - ll1)
    return {"n": n, "exceptions": x, "expected": alpha * n,
            "exception_rate": pi_hat, "lr_pof": float(lr),
            "p_value": float(chi2.sf(lr, df=1)),
            "reject_95": bool(lr > chi2.ppf(0.95, 1))}


def christoffersen(exceptions: np.ndarray, alpha: float) -> dict:
    """Christoffersen independence and conditional-coverage tests."""
    I = np.asarray(exceptions).astype(int)
    if I.size < 2:
        raise ValueError("need at least 2 observations")
    prev, curr = I[:-1], I[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)

    def _ll(p, a, b):  # a failures out of a+b with prob p
        if p in (0.0, 1.0):
            return 0.0
        return b * np.log(1 - p) + a * np.log(p)

    ll_null = _ll(pi, n01 + n11, n00 + n10)
    ll_alt = _ll(pi0, n01, n00) + _ll(pi1, n11, n10)
    lr_ind = -2.0 * (ll_null - ll_alt)

    pof = kupiec_pof(I, alpha)
    lr_cc = pof["lr_pof"] + lr_ind
    return {**pof,
            "lr_ind": float(lr_ind), "p_ind": float(chi2.sf(lr_ind, 1)),
            "lr_cc": float(lr_cc), "p_cc": float(chi2.sf(lr_cc, 2)),
            "transition_counts": {"n00": n00, "n01": n01, "n10": n10, "n11": n11}}


def var_exceptions_from_batch(pnl: np.ndarray, var_forecast: float) -> np.ndarray:
    """Exception indicator series: loss beyond the (static) VaR forecast."""
    return np.asarray(pnl, float) < -abs(var_forecast)


# ---------------------------------------------------------------------------
# Edge inference
# ---------------------------------------------------------------------------

def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 10000, level: float = 0.95,
                      seed: int = 0) -> dict:
    """Percentile bootstrap CI for the mean, plus a t-test against zero."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    t = ttest_1samp(x, 0.0)
    return {"mean": float(x.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "level": level, "t_stat": float(t.statistic),
            "p_value_two_sided": float(t.pvalue),
            "significantly_positive": bool(lo > 0.0)}


# ---------------------------------------------------------------------------
# Sequential monitoring
# ---------------------------------------------------------------------------

def cusum_monitor(realized_edges: np.ndarray, promised_edge: float,
                  k_frac: float = 0.5, h_sigmas: float = 8.0) -> dict:
    """One-sided CUSUM against under-delivery of the promised edge.

    realized_edges : per-trade realised P/L (edge) sequence, in time order.
    promised_edge  : what the model claims each trade earns (half-spread).
    k_frac         : allowance as a fraction of the edge std (drift slack).
    h_sigmas       : alarm threshold in units of the edge std.

    Returns the CUSUM path, the alarm index (or None) and the parameters.
    The std is estimated on a burn-in window (the chart's calibration
    period); monitoring starts AFTER burn-in, as a validator would do —
    you do not raise alarms on the data you used to set the control limits.
    Alarm indices are reported in original trade numbering.
    """
    x = np.asarray(realized_edges, float)
    burn = min(60, max(10, x.size // 5))
    s = float(np.std(x[:burn])) or 1.0
    k = k_frac * s
    h = h_sigmas * s

    xm = x[burn:]
    c = np.zeros(xm.size + 1)
    alarm = None
    for t in range(xm.size):
        c[t + 1] = max(0.0, c[t] + (promised_edge - xm[t]) - k)
        if alarm is None and c[t + 1] > h:
            alarm = burn + t + 1
    return {"cusum": c.tolist(), "alarm_at_trade": alarm,
            "threshold": h, "allowance": k, "edge_std_burnin": s,
            "burn_in": int(burn), "n_trades": int(x.size)}
