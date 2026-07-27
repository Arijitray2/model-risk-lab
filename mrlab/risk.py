"""
risk.py — Turning the desk's P/L distribution into risk numbers.

Given a Monte-Carlo sample {L_1, ..., L_n} of desk-level P/L (one draw per
seed), we compute the metrics a market-risk or model-risk team would ask
for.

Value-at-Risk (VaR)
-------------------
VaR_alpha is the loss threshold exceeded with probability alpha:

    VaR_alpha = -inf{ x : P(L <= x) >= alpha }        (loss sign convention)

Here alpha = 0.01 or 0.05 and we estimate by the empirical quantile.

Expected Shortfall (ES)
-----------------------
ES_alpha = -E[ L | L <= q_alpha ] — the average of the tail beyond VaR.
ES is coherent (subadditive) where VaR is not, and it is the number that
actually distinguishes a fat-tailed desk from a thin-tailed one with the
same VaR.

Model-risk premium
------------------
The extra half-spread the desk must charge so that its one-'year' P/L is
positive with confidence (1 - alpha), computed by re-running the desk
batch over a grid of half-spreads and interpolating the smallest s with

    P( PnL(s) < 0 ) <= alpha.

The DIFFERENCE between that break-even spread under a mismatched model and
under the matched model is a dollar price of model risk: what liquidity
consumers pay because the desk is unsure of its model.
"""

from __future__ import annotations

import numpy as np


def var_es(pnl: np.ndarray, alpha: float = 0.05) -> dict:
    """Empirical VaR and ES at level alpha from a P/L sample (profit +)."""
    x = np.sort(np.asarray(pnl, float))
    q = np.quantile(x, alpha)
    tail = x[x <= q]
    return {
        "alpha": alpha,
        "var": float(-q),
        "es": float(-tail.mean()) if tail.size else float(-q),
        "prob_loss": float(np.mean(x < 0.0)),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "skew": float(((x - x.mean()) ** 3).mean() / x.std() ** 3) if x.std() > 0 else 0.0,
        "min": float(x[0]),
        "max": float(x[-1]),
    }


def summarize_batch(batch: dict, alphas=(0.05, 0.01)) -> dict:
    """Risk summary of a run_batch() output."""
    out = {"n_seeds": int(batch["final_pnl"].size),
           "avg_trades": float(batch["trades"].mean()),
           "avg_edge_mean": float(batch["avg_edge"].mean()),
           "max_drawdown_mean": float(batch["max_drawdown"].mean()),
           "max_drawdown_p95": float(np.quantile(batch["max_drawdown"], 0.95))}
    for a in alphas:
        out[f"risk_{int(a*100)}pct"] = var_es(batch["final_pnl"], a)
    return out


def breakeven_spread(run_batch_fn, cfg, spreads, alpha: float = 0.05,
                     n_seeds: int = 150) -> dict:
    """Smallest half-spread with P(annual P/L < 0) <= alpha.

    run_batch_fn(cfg, n_seeds) -> batch dict; cfg is copied per spread.
    Returns the loss-probability curve and the interpolated break-even.
    """
    from dataclasses import replace

    probs = []
    for s in spreads:
        b = run_batch_fn(replace(cfg, half_spread=float(s)), n_seeds=n_seeds)
        probs.append(float(np.mean(b["final_pnl"] < 0.0)))
    probs = np.array(probs)
    spreads = np.asarray(spreads, float)

    be = None
    for i in range(len(spreads)):
        if probs[i] <= alpha:
            if i == 0:
                be = float(spreads[0])
            else:
                # Linear interpolation between the bracketing grid points.
                p0, p1 = probs[i - 1], probs[i]
                w = (p0 - alpha) / (p0 - p1) if p0 > p1 else 1.0
                be = float(spreads[i - 1] + w * (spreads[i] - spreads[i - 1]))
            break
    return {"spreads": spreads.tolist(), "prob_loss": probs.tolist(),
            "alpha": alpha, "breakeven_half_spread": be}
