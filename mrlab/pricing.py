"""
pricing.py — Option pricers, greeks, and implied volatility.

Models
------
1. Black–Scholes (1973). Underlying follows geometric Brownian motion (GBM)
   under the risk-neutral measure:

       dS_t = r S_t dt + sigma S_t dW_t

   European call price:

       C = S0 N(d1) - K e^{-rT} N(d2)
       d1 = [ln(S0/K) + (r + sigma^2/2) T] / (sigma sqrt(T)),   d2 = d1 - sigma sqrt(T)

2. Merton jump-diffusion (1976). GBM plus compound-Poisson jumps: jumps
   arrive at rate `lam` per year and each multiplies the price by e^Y with
   Y ~ N(mu_j, sigma_j^2). With k = E[e^Y] - 1, the risk-neutral price is a
   Poisson-weighted mixture of Black–Scholes prices:

       C_Merton = sum_{n>=0} e^{-lam' T} (lam' T)^n / n! * BS(S0, K, T, r_n, sigma_n)

   where lam' = lam (1+k),
         sigma_n^2 = sigma^2 + n sigma_j^2 / T,
         r_n = r - lam k + n (mu_j + sigma_j^2/2) / T.

   Intuition: conditional on n jumps having occurred, the terminal log-price
   is still Gaussian — so each term is a Black–Scholes price with jump-
   adjusted drift and variance, weighted by the Poisson probability of n.

All functions are vectorised over numpy arrays where sensible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

# Truncation of the Merton series. P(N > 60) is astronomically small for any
# lam*T used here; the tail terms also carry vanishing BS weights.
_MERTON_N_TERMS = 60


# ---------------------------------------------------------------------------
# Black–Scholes
# ---------------------------------------------------------------------------

def bs_price(S, K, T, r, sigma, option="call"):
    """Black–Scholes European option price.

    Handles T -> 0 and sigma -> 0 by returning discounted intrinsic value.
    """
    S = np.asarray(S, dtype=float)
    if np.any(np.asarray(T) <= 0) or np.any(np.asarray(sigma) <= 0):
        disc = np.exp(-r * np.maximum(T, 0.0))
        if option == "call":
            return np.maximum(S - K * disc, 0.0)
        return np.maximum(K * disc - S, 0.0)

    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    if option == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _ncdf(x: float) -> float:
    """Fast scalar standard-normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_delta(S, K, T, r, sigma, option="call"):
    """Black–Scholes delta = dV/dS. Call: N(d1); put: N(d1) - 1."""
    if T <= 0 or sigma <= 0:
        intrinsic = (S > K) if option == "call" else (S < K)
        return float(intrinsic) if option == "call" else float(intrinsic) - 1.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return _ncdf(d1) if option == "call" else _ncdf(d1) - 1.0


def bs_vega(S, K, T, r, sigma):
    """dV/dsigma (same for call and put)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return S * math.sqrt(T) * norm.pdf(d1)


def implied_vol(price, S, K, T, r, option="call", lo=1e-4, hi=5.0):
    """Invert Black–Scholes for sigma with Brent's method.

    Returns np.nan when the quote violates static no-arbitrage bounds
    (price below intrinsic or above the underlying), which does happen in
    raw market snapshots.
    """
    intrinsic = max(S - K * math.exp(-r * T), 0.0) if option == "call" \
        else max(K * math.exp(-r * T) - S, 0.0)
    if price <= intrinsic + 1e-12 or price >= (S if option == "call" else K * math.exp(-r * T)):
        return float("nan")
    try:
        return brentq(lambda s: bs_price(S, K, T, r, s, option) - price, lo, hi, xtol=1e-8)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Merton jump-diffusion
# ---------------------------------------------------------------------------

@dataclass
class MertonParams:
    """Risk-neutral Merton parameters.

    sigma   : diffusive volatility (annualised)
    lam     : jump intensity, jumps per year
    mu_j    : mean of the log jump size Y
    sigma_j : std of the log jump size Y
    """
    sigma: float
    lam: float
    mu_j: float
    sigma_j: float

    @property
    def k(self) -> float:
        """Expected relative jump size E[e^Y] - 1."""
        return math.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1.0


def _merton_terms(T, r, p: MertonParams):
    """Poisson weights and per-term (r_n, sigma_n) for the Merton series,
    vectorised over the truncated summation index n."""
    n = np.arange(_MERTON_N_TERMS)
    lam_p = p.lam * (1.0 + p.k)
    # log weights for numerical stability: log w_n = -lam'T + n log(lam'T) - log n!
    with np.errstate(divide="ignore"):
        logw = -lam_p * T + n * np.log(max(lam_p * T, 1e-300)) - \
            np.array([math.lgamma(i + 1) for i in n])
    w = np.exp(logw)
    sigma_n = np.sqrt(p.sigma**2 + n * p.sigma_j**2 / T)
    r_n = r - p.lam * p.k + n * (p.mu_j + 0.5 * p.sigma_j**2) / T
    return w, r_n, sigma_n


def merton_price(S, K, T, r, p: MertonParams, option="call"):
    """Merton (1976) price as a truncated Poisson mixture of BS prices."""
    if T <= 0:
        return max(S - K, 0.0) if option == "call" else max(K - S, 0.0)
    w, r_n, sigma_n = _merton_terms(T, r, p)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r_n + 0.5 * sigma_n**2) * T) / (sigma_n * sqrtT)
    d2 = d1 - sigma_n * sqrtT
    disc = np.exp(-r_n * T)
    if option == "call":
        px = S * norm.cdf(d1) - K * disc * norm.cdf(d2)
    else:
        px = K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)
    return float(np.sum(w * px))


def merton_delta(S, K, T, r, p: MertonParams, option="call"):
    """Delta of the Merton price — same Poisson mixture applied to BS deltas."""
    if T <= 0:
        d = 1.0 if S > K else 0.0
        return d if option == "call" else d - 1.0
    w, r_n, sigma_n = _merton_terms(T, r, p)
    d1 = (math.log(S / K) + (r_n + 0.5 * sigma_n**2) * T) / (sigma_n * math.sqrt(T))
    deltas = norm.cdf(d1) if option == "call" else norm.cdf(d1) - 1.0
    return float(np.sum(w * deltas))


# ---------------------------------------------------------------------------
# Unified model interface used by the simulator
# ---------------------------------------------------------------------------

class Pricer:
    """A pricing model the desk can quote and hedge with.

    name = "bs"     : Black–Scholes with volatility `sigma`
    name = "merton" : Merton jump-diffusion with MertonParams
    """

    def __init__(self, name: str, r: float, sigma: float | None = None,
                 merton: MertonParams | None = None):
        assert name in ("bs", "merton")
        if name == "bs" and sigma is None:
            raise ValueError("BS pricer needs sigma")
        if name == "merton" and merton is None:
            raise ValueError("Merton pricer needs MertonParams")
        self.name, self.r, self.sigma, self.merton_p = name, r, sigma, merton

    def price(self, S, K, T, option="call"):
        if self.name == "bs":
            return float(bs_price(S, K, T, self.r, self.sigma, option))
        return merton_price(S, K, T, self.r, self.merton_p, option)

    def delta(self, S, K, T, option="call"):
        if self.name == "bs":
            return float(bs_delta(S, K, T, self.r, self.sigma, option))
        return merton_delta(S, K, T, self.r, self.merton_p, option)


if __name__ == "__main__":
    # Sanity checks: put-call parity and Merton >= BS under fat tails.
    S0, K, T, r, sig = 100.0, 100.0, 0.25, 0.02, 0.2
    c = bs_price(S0, K, T, r, sig, "call")
    pp = bs_price(S0, K, T, r, sig, "put")
    parity = c - pp - (S0 - K * math.exp(-r * T))
    print(f"BS call {c:.4f} put {pp:.4f} parity residual {parity:.2e}")

    mp = MertonParams(sigma=sig, lam=3.0, mu_j=-0.08, sigma_j=0.15)
    cm = merton_price(S0, K, T, r, mp, "call")
    print(f"Merton call {cm:.4f} (> BS {c:.4f}: {cm > c})")

    iv = implied_vol(cm, S0, K, T, r, "call")
    print(f"Merton price back through BS => implied vol {iv:.4f} (> {sig})")
