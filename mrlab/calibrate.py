"""
calibrate.py — Fitting the models to real market data.

Two calibrations, two probability measures
------------------------------------------
1. PHYSICAL (P-measure), from the underlying's return history by maximum
   likelihood. Answers: "what process actually generates returns?"

   * GBM: log-returns x_i over step dt are iid Normal(
         (mu - sigma^2/2) dt,  sigma^2 dt ).
     The MLE is available in closed form.

   * Merton jump-diffusion: each log-return is an infinite Poisson mixture
     of Gaussians,

         f(x) = sum_{n>=0} e^{-lam dt} (lam dt)^n / n!
                * phi( x ; (mu - sigma^2/2 - lam k) dt + n mu_j,
                           sigma^2 dt + n sigma_j^2 )

     maximised numerically. Standard errors come from the inverse of the
     numerically-differentiated Hessian of the negative log-likelihood
     (the observed Fisher information).

   Model comparison: the models are nested (Merton -> GBM as lam -> 0), so
   we report a likelihood-ratio statistic LR = 2(l_M - l_G) alongside AIC
   and BIC. Because lam = 0 sits on the parameter-space boundary, the naive
   chi-squared(3) reference is conservative; we say so rather than pretend.

2. RISK-NEUTRAL (Q-measure), from an option-chain snapshot by weighted
   least squares on implied volatilities. Answers: "what process is the
   options market pricing?" We minimise

       sum_i w_i ( IV_model(K_i) - IV_market(K_i) )^2

   over Merton parameters, with vega-informed weights (ATM quotes are the
   most informative). Fitting IVs rather than prices avoids the fit being
   dominated by expensive deep-ITM quotes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from .pricing import MertonParams, merton_price, implied_vol


# ---------------------------------------------------------------------------
# Physical-measure MLE
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    model: str
    params: dict
    se: dict
    loglik: float
    n_obs: int
    aic: float
    bic: float
    extra: dict = field(default_factory=dict)


def fit_gbm(log_returns: np.ndarray, dt: float) -> FitResult:
    """Closed-form Gaussian MLE for GBM log-returns."""
    x = np.asarray(log_returns, float)
    n = x.size
    m_hat = x.mean()
    v_hat = x.var()                      # MLE (divide by n)
    sigma = np.sqrt(v_hat / dt)
    mu = m_hat / dt + 0.5 * sigma**2

    # Log-likelihood at the optimum.
    ll = float(np.sum(norm.logpdf(x, m_hat, np.sqrt(v_hat))))

    # Delta-method standard errors from the exact Fisher information of
    # (m, v):  Var(m_hat) = v/n, Var(v_hat) = 2v^2/n.
    se_m = np.sqrt(v_hat / n)
    se_v = np.sqrt(2.0 * v_hat**2 / n)
    se_sigma = se_v / (2.0 * np.sqrt(v_hat * dt))          # sigma = sqrt(v/dt)
    # mu = m/dt + v/(2 dt): combine independent m̂, v̂ variances.
    se_mu = np.sqrt(se_m**2 / dt**2 + se_v**2 / (4.0 * dt**2))

    k = 2
    return FitResult(
        model="gbm",
        params={"mu": float(mu), "sigma": float(sigma)},
        se={"mu": float(se_mu), "sigma": float(se_sigma)},
        loglik=ll, n_obs=n,
        aic=2 * k - 2 * ll, bic=k * np.log(n) - 2 * ll,
    )


def _merton_neg_loglik(theta, x, dt, n_max=8):
    """Negative log-likelihood of Merton log-returns.

    theta = (mu, log sigma, log lam, mu_j, log sigma_j) — logs enforce
    positivity. The Poisson mixture is truncated at n_max jumps per step
    (dt is one day: P(N>8) is negligible for any sane lam).
    """
    mu, sigma, lam, mu_j, sigma_j = (
        theta[0], np.exp(theta[1]), np.exp(theta[2]), theta[3], np.exp(theta[4]))
    k = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    base_mean = (mu - 0.5 * sigma**2 - lam * k) * dt
    log_pois = -lam * dt
    dens = np.zeros_like(x)
    w = np.exp(log_pois)
    for n in range(n_max + 1):
        mean_n = base_mean + n * mu_j
        var_n = sigma**2 * dt + n * sigma_j**2
        dens += w * norm.pdf(x, mean_n, np.sqrt(var_n))
        w *= lam * dt / (n + 1)
    dens = np.maximum(dens, 1e-300)
    return -np.sum(np.log(dens))


def fit_merton(log_returns: np.ndarray, dt: float,
               n_starts: int = 6, seed: int = 0) -> FitResult:
    """Numerical MLE for Merton jump-diffusion with multi-start optimisation.

    The likelihood is multi-modal (jumps vs diffusion can trade off), so we
    run several starts and keep the best. SEs come from the numerical
    Hessian at the optimum, mapped back through the log-parameterisation by
    the delta method.
    """
    x = np.asarray(log_returns, float)
    n = x.size
    rng = np.random.default_rng(seed)

    g = fit_gbm(x, dt)
    best = None
    for i in range(n_starts):
        lam0 = rng.uniform(0.5, 30.0)
        theta0 = np.array([
            g.params["mu"] * rng.uniform(0.5, 1.5),
            np.log(g.params["sigma"] * rng.uniform(0.5, 1.0)),
            np.log(lam0),
            rng.uniform(-0.05, 0.0),
            np.log(rng.uniform(0.01, 0.06)),
        ])
        opt = minimize(_merton_neg_loglik, theta0, args=(x, dt),
                       method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8})
        if best is None or opt.fun < best.fun:
            best = opt
    # Polish with BFGS from the best simplex point.
    opt = minimize(_merton_neg_loglik, best.x, args=(x, dt), method="BFGS",
                   options={"maxiter": 500})
    if opt.fun > best.fun:
        opt = best

    mu, sigma, lam, mu_j, sigma_j = (
        opt.x[0], np.exp(opt.x[1]), np.exp(opt.x[2]), opt.x[3], np.exp(opt.x[4]))
    ll = -float(opt.fun)

    # Numerical Hessian (central differences) in theta-space.
    se = {}
    try:
        h = 1e-4
        d = len(opt.x)
        H = np.zeros((d, d))
        f0 = opt.fun
        for a in range(d):
            for b in range(a, d):
                ea, eb = np.zeros(d), np.zeros(d)
                ea[a] = h; eb[b] = h
                fpp = _merton_neg_loglik(opt.x + ea + eb, x, dt)
                fpm = _merton_neg_loglik(opt.x + ea - eb, x, dt)
                fmp = _merton_neg_loglik(opt.x - ea + eb, x, dt)
                fmm = _merton_neg_loglik(opt.x - ea - eb, x, dt)
                H[a, b] = H[b, a] = (fpp - fpm - fmp + fmm) / (4 * h * h)
        cov_theta = np.linalg.inv(H)
        sd_theta = np.sqrt(np.maximum(np.diag(cov_theta), 0.0))
        # Delta method through the exp() links.
        se = {
            "mu": float(sd_theta[0]),
            "sigma": float(sd_theta[1] * sigma),
            "lam": float(sd_theta[2] * lam),
            "mu_j": float(sd_theta[3]),
            "sigma_j": float(sd_theta[4] * sigma_j),
        }
    except np.linalg.LinAlgError:
        se = {p: float("nan") for p in ("mu", "sigma", "lam", "mu_j", "sigma_j")}

    kpar = 5
    lr = 2.0 * (ll - g.loglik)
    return FitResult(
        model="merton",
        params={"mu": float(mu), "sigma": float(sigma), "lam": float(lam),
                "mu_j": float(mu_j), "sigma_j": float(sigma_j)},
        se=se, loglik=ll, n_obs=n,
        aic=2 * kpar - 2 * ll, bic=kpar * np.log(n) - 2 * ll,
        extra={
            "lr_vs_gbm": float(lr),
            "lr_note": ("LR = 2(l_M - l_G). Under H0 the boundary (lam=0) makes "
                        "the chi2(3) p-value conservative; treat as indicative."),
            "gbm_loglik": g.loglik,
        },
    )


def qq_data(log_returns: np.ndarray, dt: float, fit: FitResult,
            n_grid: int = 200) -> dict:
    """Quantile-quantile data of standardised returns vs the fitted model.

    For GBM this is a Gaussian QQ plot. For Merton we compute model
    quantiles by numerically inverting the mixture CDF on a grid.
    """
    x = np.sort(np.asarray(log_returns, float))
    n = x.size
    probs = (np.arange(1, n + 1) - 0.5) / n
    if fit.model == "gbm":
        m = (fit.params["mu"] - 0.5 * fit.params["sigma"]**2) * dt
        s = fit.params["sigma"] * np.sqrt(dt)
        theo = norm.ppf(probs, m, s)
    else:
        p = fit.params
        k = np.exp(p["mu_j"] + 0.5 * p["sigma_j"]**2) - 1.0
        base_mean = (p["mu"] - 0.5 * p["sigma"]**2 - p["lam"] * k) * dt
        grid = np.linspace(x[0] * 1.5, x[-1] * 1.5, 20000)
        cdf = np.zeros_like(grid)
        w = np.exp(-p["lam"] * dt)
        for nj in range(9):
            cdf += w * norm.cdf(grid, base_mean + nj * p["mu_j"],
                                np.sqrt(p["sigma"]**2 * dt + nj * p["sigma_j"]**2))
            w *= p["lam"] * dt / (nj + 1)
        theo = np.interp(probs, cdf, grid)
    # Thin for plotting.
    idx = np.unique(np.linspace(0, n - 1, n_grid).astype(int))
    return {"empirical": x[idx].tolist(), "theoretical": theo[idx].tolist()}


# ---------------------------------------------------------------------------
# Risk-neutral calibration to an option chain
# ---------------------------------------------------------------------------

def chain_implied_vols(chain, spot: float, r: float):
    """Attach mid-price Black–Scholes implied vols to a chain DataFrame.

    Expects columns: strike, type ('call'/'put'), bid, ask, dte (days).
    Returns a copy with 'mid', 'T', 'iv' columns; drops unusable quotes.
    """
    df = chain.copy()
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    df["T"] = df["dte"] / 365.0
    df["iv"] = [
        implied_vol(m, spot, k, t, r, o)
        for m, k, t, o in zip(df["mid"], df["strike"], df["T"], df["type"])
    ]
    return df.dropna(subset=["iv"])


def fit_merton_riskneutral(strikes, ivs, types, spot, T, r,
                           weights=None, n_starts: int = 4, seed: int = 0):
    """Weighted least-squares fit of Merton parameters to market IVs.

    Objective: sum_i w_i (IV_model(K_i) - IV_mkt(K_i))^2. For speed we use
    the standard first-order equivalence IV error ~= price error / vega
    (vega evaluated at the MARKET IV, so it is a constant during the
    optimisation): minimising vega-scaled price residuals avoids a
    Black–Scholes inversion at every objective evaluation. The reported
    RMSE and fitted smile ARE computed with exact IV inversion at the end.
    Returns (MertonParams, diagnostics dict).
    """
    from .pricing import bs_vega, bs_price

    strikes = np.asarray(strikes, float)
    ivs = np.asarray(ivs, float)
    if weights is None:
        # Gaussian weight in log-moneyness: ATM options carry the most
        # information per unit of quote noise (highest vega).
        weights = np.exp(-0.5 * (np.log(strikes / spot) / 0.15) ** 2)
    weights = np.asarray(weights, float) / np.sum(weights)

    mkt_price = np.array([bs_price(spot, K, T, r, iv, o)
                          for K, iv, o in zip(strikes, ivs, types)])
    vegas = np.array([max(bs_vega(spot, K, T, r, iv), 1e-4)
                      for K, iv in zip(strikes, ivs)])

    def params_of(theta):
        return MertonParams(np.exp(theta[0]), np.exp(theta[1]),
                            theta[2], np.exp(theta[3]))

    def obj(theta):
        # Soft box constraints: reject absurd regions the simplex may probe.
        if not (-5.0 < theta[0] < 1.5 and -14.0 < theta[1] < 4.0 and
                -2.0 < theta[2] < 1.0 and -7.0 < theta[3] < 0.7):
            return 1e6
        p = params_of(theta)
        resid = np.array([
            (merton_price(spot, K, T, r, p, o) - mp) / v
            for K, o, mp, v in zip(strikes, types, mkt_price, vegas)])
        return float(np.sum(weights * resid ** 2))

    rng = np.random.default_rng(seed)
    atm_iv = float(ivs[np.argmin(np.abs(strikes - spot))])
    best_t, best_f = None, np.inf
    for i in range(n_starts):
        theta0 = np.array([
            np.log(max(atm_iv * rng.uniform(0.5, 0.95), 0.03)),
            np.log(rng.uniform(0.2, 3.0)),
            rng.uniform(-0.25, -0.02),
            np.log(rng.uniform(0.05, 0.25)),
        ])
        opt = minimize(obj, theta0, method="Nelder-Mead",
                       options={"maxiter": 1200, "xatol": 1e-5, "fatol": 1e-10})
        if opt.fun < best_f:
            best_t, best_f = opt.x, opt.fun

    p = params_of(best_t)
    p = MertonParams(float(p.sigma), float(p.lam), float(p.mu_j), float(p.sigma_j))
    # Exact fitted smile via BS inversion of the Merton prices.
    fitted = np.array([
        implied_vol(merton_price(spot, K, T, r, p, o), spot, K, T, r, o)
        for K, o in zip(strikes, types)])
    ok = np.isfinite(fitted)
    rmse = float(np.sqrt(np.mean((fitted[ok] - ivs[ok]) ** 2)))
    return p, {"rmse_iv": rmse, "fitted_ivs": fitted.tolist(),
               "objective": best_f}
