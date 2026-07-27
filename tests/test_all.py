"""Unit tests for mrlab. Run: python -m pytest tests/ -q  (or python tests/test_all.py)"""
import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from mrlab.pricing import (bs_price, bs_delta, implied_vol,
                           MertonParams, merton_price, merton_delta, Pricer)
from mrlab.simulate import DeskConfig, run_desk, run_batch, simulate_path
from mrlab.calibrate import fit_gbm, fit_merton
from mrlab.risk import var_es
from mrlab.validate import kupiec_pof, christoffersen, bootstrap_mean_ci, cusum_monitor


S0, K, T, R, SIG = 100.0, 100.0, 0.25, 0.02, 0.2


def test_put_call_parity():
    c = bs_price(S0, K, T, R, SIG, "call")
    p = bs_price(S0, K, T, R, SIG, "put")
    assert abs((c - p) - (S0 - K * math.exp(-R * T))) < 1e-10


def test_bs_delta_matches_finite_difference():
    h = 1e-4
    fd = (bs_price(S0 + h, K, T, R, SIG) - bs_price(S0 - h, K, T, R, SIG)) / (2 * h)
    assert abs(bs_delta(S0, K, T, R, SIG) - fd) < 1e-6


def test_implied_vol_roundtrip():
    px = bs_price(S0, K, T, R, 0.31, "call")
    assert abs(implied_vol(px, S0, K, T, R, "call") - 0.31) < 1e-6


def test_merton_exceeds_bs_and_converges_to_bs():
    mp = MertonParams(SIG, lam=3.0, mu_j=-0.08, sigma_j=0.15)
    assert merton_price(S0, K, T, R, mp) > bs_price(S0, K, T, R, SIG)
    mp0 = MertonParams(SIG, lam=1e-12, mu_j=-0.08, sigma_j=0.15)
    assert abs(merton_price(S0, K, T, R, mp0) - bs_price(S0, K, T, R, SIG)) < 1e-8


def test_merton_delta_matches_finite_difference():
    mp = MertonParams(SIG, 2.0, -0.05, 0.1)
    h = 1e-3
    fd = (merton_price(S0 + h, K, T, R, mp) - merton_price(S0 - h, K, T, R, mp)) / (2 * h)
    assert abs(merton_delta(S0, K, T, R, mp) - fd) < 1e-5


def test_matched_desk_realizes_half_spread_on_average():
    cfg = DeskConfig(true_model="gbm", pricing_model="bs", n_ticks=150,
                     hedge_steps=25)
    b = run_batch(cfg, n_seeds=40)
    mean_edge = b["avg_edge"].mean()
    # Across seeds the realised edge per contract should approximate the
    # quoted half-spread (0.15) with a modest tolerance.
    assert 0.10 < mean_edge < 0.20, mean_edge


def test_mismatched_short_vol_desk_underperforms_belief():
    cfg = DeskConfig(true_model="merton", pricing_model="bs", option="put",
                     lam=4.0, mu_j=-0.12, sigma_j=0.15,
                     fill_prob_ask=0.5, fill_prob_bid=0.1,
                     n_ticks=150, hedge_steps=25)
    b = run_batch(cfg, n_seeds=40)
    # The desk believes it earns +half-spread; in truth it earns far less.
    assert b["avg_edge"].mean() < 0.10


def test_path_generator_is_seeded_and_positive():
    cfg = DeskConfig()
    rng1 = np.random.default_rng(3)
    rng2 = np.random.default_rng(3)
    p1 = simulate_path(cfg, 100, 1 / 252, 100.0, rng1)
    p2 = simulate_path(cfg, 100, 1 / 252, 100.0, rng2)
    assert np.allclose(p1, p2) and (p1 > 0).all()


def test_gbm_mle_recovers_parameters():
    rng = np.random.default_rng(1)
    dt = 1 / 252
    mu, sigma = 0.08, 0.22
    x = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), 40000)
    fit = fit_gbm(x, dt)
    assert abs(fit.params["sigma"] - sigma) < 0.01
    assert abs(fit.params["mu"] - mu) < 3 * fit.se["mu"] + 0.02


def test_merton_mle_finds_jumps_in_jumpy_data():
    rng = np.random.default_rng(2)
    dt = 1 / 252
    n = 6000
    sigma, lam, mu_j, sigma_j = 0.15, 25.0, -0.03, 0.02
    N = rng.poisson(lam * dt, n)
    x = rng.normal(0.0004, sigma * np.sqrt(dt), n) \
        + rng.normal(mu_j * N, sigma_j * np.sqrt(np.maximum(N, 1e-12)))
    fit = fit_merton(x, dt, n_starts=4)
    # Jump component should be clearly detected (LR large, lam > 0).
    assert fit.extra["lr_vs_gbm"] > 20
    assert fit.params["lam"] > 5.0


def test_var_es_orders_correctly():
    rng = np.random.default_rng(4)
    pnl = rng.normal(0, 1, 20000)
    m = var_es(pnl, 0.05)
    assert m["es"] > m["var"] > 0


def test_kupiec_accepts_correct_and_rejects_bad_coverage():
    rng = np.random.default_rng(10)
    ok = rng.random(2000) < 0.05
    bad = rng.random(2000) < 0.15
    assert not kupiec_pof(ok, 0.05)["reject_95"]
    assert kupiec_pof(bad, 0.05)["reject_95"]


def test_christoffersen_flags_clustering():
    # Build an exception series with strong clustering but ~5% rate.
    e = np.zeros(2000, dtype=bool)
    e[100:150] = True; e[900:950] = True
    out = christoffersen(e, 0.05)
    assert out["p_ind"] < 0.01


def test_bootstrap_ci_detects_positive_mean():
    rng = np.random.default_rng(6)
    x = rng.normal(0.15, 0.05, 200)
    out = bootstrap_mean_ci(x, n_boot=2000)
    assert out["significantly_positive"]


def test_cusum_alarms_on_under_delivery_only():
    rng = np.random.default_rng(7)
    good = rng.normal(0.15, 0.3, 400)          # delivers promised 0.15
    bad = rng.normal(-0.05, 0.3, 400)          # under-delivers
    assert cusum_monitor(good, 0.15)["alarm_at_trade"] is None
    assert cusum_monitor(bad, 0.15)["alarm_at_trade"] is not None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print(f"PASS {f.__name__}")
    print(f"\n{len(fns)} tests passed.")
