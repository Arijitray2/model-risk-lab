"""
run_experiments.py — Reproduce every number and figure in the project.

Stages
------
1. Physical-measure calibration of GBM & Merton to SPY (2008–2025) and
   NIFTY 50 (2007–2026) daily log-returns.
2. Risk-neutral Merton calibration to four real SPY option-chain snapshots
   (calm 2017, COVID March 2020, 2022 bear market, mid-2025).
3. Market-making desk experiments across four (true model, believed model)
   scenarios with parameters taken from the calibrations.
4. Model-risk premium: break-even half-spread curves.
5. Validation suite: VaR backtests (Kupiec/Christoffersen), bootstrap edge
   inference, CUSUM monitoring.

Outputs JSON files into results/ that the website, the figures script and
the report all consume. Runtime: ~15 minutes on a laptop.

Usage: python scripts/run_experiments.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mrlab.calibrate import (fit_gbm, fit_merton, qq_data,
                             chain_implied_vols, fit_merton_riskneutral)
from mrlab.pricing import MertonParams, merton_price, implied_vol, bs_price
from mrlab.simulate import DeskConfig, run_desk, run_batch
from mrlab.risk import summarize_batch, breakeven_spread
from mrlab.validate import (kupiec_pof, christoffersen, bootstrap_mean_ci,
                            cusum_monitor)

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

DT = 1.0 / 252.0


def save(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1, default=float)
    print(f"  wrote results/{name}")


# ---------------------------------------------------------------------------
# Stage 1 — physical calibration
# ---------------------------------------------------------------------------

def stage1():
    print("== Stage 1: physical-measure calibration ==")
    out = {}
    series = {
        "spy": ("spy_underlying.csv", "adjusted_close", "2008-01-01", "SPY (US)"),
        "nifty": ("nifty50_history.csv", "close", "2007-09-01", "NIFTY 50 (India)"),
    }
    for key, (fname, col, start, label) in series.items():
        df = pd.read_csv(os.path.join(DATA, fname), parse_dates=["date"])
        df = df[df["date"] >= start].sort_values("date")
        px = df[col].to_numpy(float)
        x = np.diff(np.log(px))
        # Guard against zero-return holidays duplicated in some feeds.
        x = x[np.abs(x) > 0]

        g = fit_gbm(x, DT)
        m = fit_merton(x, DT, n_starts=8, seed=42)
        print(f"  {label}: n={x.size}  GBM sigma={g.params['sigma']:.4f}  "
              f"Merton sigma={m.params['sigma']:.4f} lam={m.params['lam']:.1f} "
              f"mu_j={m.params['mu_j']:.4f} sigma_j={m.params['sigma_j']:.4f} "
              f"LR={m.extra['lr_vs_gbm']:.1f}")

        # Empirical histogram + fitted densities on a common grid.
        lo, hi = np.quantile(x, [0.001, 0.999])
        grid = np.linspace(lo * 1.3, hi * 1.3, 400)
        from scipy.stats import norm as _n
        gbm_pdf = _n.pdf(grid, (g.params["mu"] - 0.5 * g.params["sigma"]**2) * DT,
                         g.params["sigma"] * np.sqrt(DT))
        p = m.params
        kj = np.exp(p["mu_j"] + 0.5 * p["sigma_j"]**2) - 1.0
        base_mean = (p["mu"] - 0.5 * p["sigma"]**2 - p["lam"] * kj) * DT
        mert_pdf = np.zeros_like(grid)
        w = np.exp(-p["lam"] * DT)
        for nj in range(9):
            mert_pdf += w * _n.pdf(grid, base_mean + nj * p["mu_j"],
                                   np.sqrt(p["sigma"]**2 * DT + nj * p["sigma_j"]**2))
            w *= p["lam"] * DT / (nj + 1)
        hist_counts, hist_edges = np.histogram(x, bins=120, density=True)

        out[key] = {
            "label": label,
            "n_obs": int(x.size),
            "date_range": [str(df['date'].iloc[0].date()), str(df['date'].iloc[-1].date())],
            "sample_moments": {
                "ann_vol": float(x.std() * np.sqrt(252)),
                "skew": float(((x - x.mean())**3).mean() / x.std()**3),
                "excess_kurtosis": float(((x - x.mean())**4).mean() / x.std()**4 - 3),
                "worst_day": float(x.min()), "best_day": float(x.max()),
            },
            "gbm": {"params": g.params, "se": g.se, "loglik": g.loglik,
                    "aic": g.aic, "bic": g.bic},
            "merton": {"params": m.params, "se": m.se, "loglik": m.loglik,
                       "aic": m.aic, "bic": m.bic, "lr_vs_gbm": m.extra["lr_vs_gbm"]},
            "qq_gbm": qq_data(x, DT, g),
            "qq_merton": qq_data(x, DT, m),
            "density": {"grid": grid.tolist(), "gbm": gbm_pdf.tolist(),
                        "merton": mert_pdf.tolist(),
                        "hist_x": (0.5 * (hist_edges[1:] + hist_edges[:-1])).tolist(),
                        "hist_y": hist_counts.tolist()},
        }
    save("calibration_physical.json", out)
    return out


# ---------------------------------------------------------------------------
# Stage 2 — risk-neutral calibration to real SPY chains
# ---------------------------------------------------------------------------

SNAPSHOTS = {
    "2017-01-03": "calm bull market",
    "2020-03-16": "COVID crash (VIX 82)",
    "2022-06-13": "2022 bear market",
    "2025-06-30": "recent",
}


def stage2():
    print("== Stage 2: risk-neutral calibration ==")
    rf = pd.read_csv(os.path.join(DATA, "us_risk_free_rate.csv"), parse_dates=["date"])
    out = {}
    for date, label in SNAPSHOTS.items():
        df = pd.read_csv(os.path.join(DATA, f"spy_chain_{date}.csv"),
                         parse_dates=["date", "expiration"])
        spot = float(df["spot"].iloc[0])
        r_row = rf[rf["date"] <= date].iloc[-1]["risk_free_rate"]
        r = float(r_row) / 100.0

        df["dte"] = (df["expiration"] - df["date"]).dt.days
        # Expiry closest to 90 calendar days with a rich strike set.
        cand = df[(df["dte"] >= 55) & (df["dte"] <= 130)]
        expiry = (cand.groupby("dte").size()
                  .to_frame("n").reset_index()
                  .assign(dist=lambda d: (d["dte"] - 90).abs())
                  .query("n >= 30").sort_values("dist")["dte"].iloc[0])
        sl = cand[cand["dte"] == expiry].copy()

        # OTM quotes only (the liquid side of the smile), sane moneyness.
        mny = sl["strike"] / spot
        otm = ((sl["type"] == "put") & (mny <= 1.0) |
               (sl["type"] == "call") & (mny > 1.0))
        sl = sl[otm & (mny > 0.70) & (mny < 1.30) & (sl["bid"] > 0.05)]

        ivd = chain_implied_vols(sl, spot, r)
        T = float(ivd["T"].iloc[0])

        p, diag = fit_merton_riskneutral(
            ivd["strike"].to_numpy(), ivd["iv"].to_numpy(),
            ivd["type"].tolist(), spot, T, r, n_starts=4, seed=1)

        atm_iv = float(ivd.iloc[(ivd["strike"] - spot).abs().argsort().iloc[0]]["iv"])
        print(f"  {date} ({label}): spot={spot:.1f} dte={expiry} n={len(ivd)} "
              f"ATM IV={atm_iv:.3f} | Merton sigma={p.sigma:.3f} lam={p.lam:.2f} "
              f"mu_j={p.mu_j:.3f} sigma_j={p.sigma_j:.3f} rmse={diag['rmse_iv']*100:.2f} vol pts")

        out[date] = {
            "label": label, "spot": spot, "r": r, "dte": int(expiry), "T": T,
            "atm_iv": atm_iv,
            "strikes": ivd["strike"].tolist(),
            "types": ivd["type"].tolist(),
            "market_iv": ivd["iv"].tolist(),
            "merton_iv": diag["fitted_ivs"],
            "merton_params": {"sigma": p.sigma, "lam": p.lam,
                              "mu_j": p.mu_j, "sigma_j": p.sigma_j},
            "rmse_iv": diag["rmse_iv"],
        }
    save("calibration_riskneutral.json", out)
    return out


# ---------------------------------------------------------------------------
# Stage 3 — desk experiments
# ---------------------------------------------------------------------------

def scenarios(phys, rn):
    """Scenario grid built from the calibrations.

    True world = Merton calibrated to the COVID chain (a jumpy world the
    options market actually priced); sigma_P from SPY physical fit for the
    matched/GBM worlds. Desk quotes 3-month ATM puts; clients net-buy
    protection (ask-heavy flow) — the realistic sign of index option flow.
    """
    sig_p = phys["spy"]["gbm"]["params"]["sigma"]
    m2020 = rn["2020-03-16"]["merton_params"]
    common = dict(S0=100.0, strike_ratio=1.0, T=0.25, option="put",
                  r=0.02, half_spread=0.15, fill_prob_ask=0.5,
                  fill_prob_bid=0.1, n_ticks=250, hedge_steps=30,
                  horizon=1.0, mu=0.06)
    return {
        "matched": DeskConfig(true_model="gbm", pricing_model="bs",
                              sigma=sig_p, **common),
        "steamroller": DeskConfig(true_model="merton", pricing_model="bs",
                                  sigma=m2020["sigma"], lam=m2020["lam"],
                                  mu_j=m2020["mu_j"], sigma_j=m2020["sigma_j"],
                                  **common),
        "right_model": DeskConfig(true_model="merton", pricing_model="merton",
                                  sigma=m2020["sigma"], lam=m2020["lam"],
                                  mu_j=m2020["mu_j"], sigma_j=m2020["sigma_j"],
                                  **common),
        "overcautious": DeskConfig(true_model="gbm", pricing_model="merton",
                                   sigma=sig_p,
                                   pricing_lam=m2020["lam"],
                                   pricing_mu_j=m2020["mu_j"],
                                   pricing_sigma_j=m2020["sigma_j"],
                                   **common),
    }


def stage3(phys, rn):
    print("== Stage 3: desk experiments ==")
    scen = scenarios(phys, rn)
    out = {}
    for name, cfg in scen.items():
        t0 = time.time()
        n_seeds = 300 if cfg.pricing_model == "bs" else 200
        batch = run_batch(cfg, n_seeds=n_seeds)
        risk = summarize_batch(batch)
        edge_ci = bootstrap_mean_ci(batch["avg_edge"], n_boot=10000)

        # Three representative single runs for the equity-curve figure.
        curves = []
        for s in (7, 21, 42):
            c = DeskConfig(**{**cfg.__dict__, "seed": s})
            r = run_desk(c)
            step = max(1, len(r.t) // 250)
            curves.append({"seed": s, "t": r.t[::step],
                           "realized": r.realized[::step],
                           "exp_model": r.exp_model[::step],
                           "exp_truth": r.exp_truth[::step],
                           "S": r.S[::step]})

        out[name] = {
            "config": {k: v for k, v in cfg.__dict__.items()},
            "n_seeds": n_seeds,
            "final_pnl": batch["final_pnl"].tolist(),
            "avg_edge": batch["avg_edge"].tolist(),
            "max_drawdown": batch["max_drawdown"].tolist(),
            "risk": risk,
            "edge_ci": edge_ci,
            "curves": curves,
        }
        print(f"  {name}: mean PnL={batch['final_pnl'].mean():8.2f}  "
              f"edge={batch['avg_edge'].mean():7.4f}  "
              f"ES5={risk['risk_5pct']['es']:8.2f}  ({time.time()-t0:.0f}s)")
    save("desk_experiments.json", out)
    return out


# ---------------------------------------------------------------------------
# Stage 4 — model-risk premium
# ---------------------------------------------------------------------------

def stage4(phys, rn):
    print("== Stage 4: break-even spreads / model-risk premium ==")
    scen = scenarios(phys, rn)
    grids = {
        "matched": np.round(np.arange(0.025, 0.45, 0.035), 3),
        # The mispriced desk needs DOLLARS of spread, not cents: the grid
        # must reach past the raw mispricing gap (~$10 per contract).
        "steamroller": np.round(np.concatenate([np.arange(0.25, 2.0, 0.5),
                                                np.arange(2.0, 15.1, 1.0)]), 3),
    }
    out = {}
    for name in ("matched", "steamroller"):
        t0 = time.time()
        out[name] = breakeven_spread(run_batch, scen[name], grids[name],
                                     alpha=0.05, n_seeds=120)
        print(f"  {name}: breakeven={out[name]['breakeven_half_spread']} "
              f"({time.time()-t0:.0f}s)")
    be_m = out["matched"]["breakeven_half_spread"]
    be_s = out["steamroller"]["breakeven_half_spread"]
    out["model_risk_premium"] = (None if be_m is None or be_s is None
                                 else be_s - be_m)
    save("breakeven.json", out)
    return out


# ---------------------------------------------------------------------------
# Stage 5 — validation suite
# ---------------------------------------------------------------------------

def stage5(phys, rn):
    print("== Stage 5: validation ==")
    scen = scenarios(phys, rn)
    out = {}

    for name in ("matched", "steamroller"):
        # ---- VaR backtest. The desk FORECASTS its per-tick P/L distribution
        # under its own believed (BS/GBM) model, then lives in the true world.
        # P/L is normalised by the spot level (P/L per unit of S0-notional):
        # an ATM option's dollar value scales with S, so a desk forecasts
        # risk per unit of notional, not in fixed dollars.
        believed = DeskConfig(**{**scen[name].__dict__,
                                 "true_model": "gbm",
                                 "lam": 0.0, "mu_j": 0.0, "sigma_j": 0.0})
        pool = []
        for s in range(25):
            r = run_desk(DeskConfig(**{**believed.__dict__, "seed": 1000 + s}))
            scale = np.array(r.S[:-1]) / believed.S0
            pool.extend(np.diff(r.realized) / scale)
        pool = np.array(pool)
        var5 = float(-np.quantile(pool, 0.05))
        out[f"var_forecast_5pct_per_tick_{name}"] = var5
        cfg = DeskConfig(**{**scen[name].__dict__,
                            "horizon": 6.0, "n_ticks": 1500, "seed": 11})
        r = run_desk(cfg)
        scale = np.array(r.S[:-1]) / cfg.S0
        dpnl = np.diff(r.realized) / scale
        exceptions = dpnl < -var5
        out[f"kupiec_{name}"] = kupiec_pof(exceptions, 0.05)
        out[f"christoffersen_{name}"] = christoffersen(exceptions, 0.05)
        out[f"exceptions_{name}"] = exceptions.astype(int).tolist()
        print(f"  {name}: exceptions {int(exceptions.sum())}/{len(exceptions)} "
              f"(expect {0.05*len(exceptions):.0f})  "
              f"Kupiec p={out[f'kupiec_{name}']['p_value']:.4f}  "
              f"CC p={out[f'christoffersen_{name}']['p_cc']:.4f}")

        # ---- CUSUM on per-trade realised edges (spot-scaled residuals:
        # promised half-spread minus realised, per unit of S0-notional).
        resid = ((np.array(r.trade_edges) - cfg.half_spread)
                 / (np.array(r.trade_S) / cfg.S0))
        cus = cusum_monitor(resid, 0.0)
        out[f"cusum_{name}"] = cus
        print(f"  {name}: CUSUM alarm at trade {cus['alarm_at_trade']} "
              f"of {cus['n_trades']}")

    # ---- Bootstrap edge inference across seeds (from stage 3 output).
    save("validation.json", out)
    return out


if __name__ == "__main__":
    t0 = time.time()
    phys = stage1()
    rn = stage2()
    desk = stage3(phys, rn)
    be = stage4(phys, rn)
    val = stage5(phys, rn)
    print(f"\nAll stages complete in {time.time()-t0:.0f}s. Results in results/.")
