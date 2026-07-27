"""
simulate.py — Options market-making desk under model risk.

The experiment
--------------
A market-making desk quotes a rolling, constant-maturity option (a fresh
T-year contract every tick, fixed strike ratio) around the fair value of the
model it BELIEVES in. The world, however, evolves under the TRUE model,
which may be different. Every fill is realised on the spot: we simulate the
contract's delta-hedged life to expiry under the true dynamics — the desk
hedging with ITS OWN model's delta — and bank the discounted P/L.

Why a hedged option realises `premium - fair value`
---------------------------------------------------
For a delta-hedged position rebalanced continuously under the model's own
dynamics, the dW term cancels and the portfolio grows at the risk-free rate:
the P/L of (option sold at `premium`, hedge held to expiry) converges to
`premium - V_fair`, independent of the real-world drift mu. So quoting
ask = V + s and bid = V - s banks the half-spread s per contract in
expectation — IF V is the true fair value and the hedge is the true delta.
Model risk breaks both legs at once: you quote around the wrong V (adverse
expected edge) and hedge the wrong delta (unhedged jump risk).

Design notes
------------
* This is a rewrite and extension of the teaching idea in Roman Paolucci's
  Quant Guild market-making simulator (see README acknowledgements); the
  code here is original and adds calibrated parameters, scenario grids,
  multi-seed batches and desk-level risk outputs.
* Fills are Bernoulli per side per tick — deliberately simple so the P/L
  story isolates pricing edge and hedge error, not microstructure alpha.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from .pricing import Pricer, MertonParams


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DeskConfig:
    # Contract quoted every tick: rolling constant maturity AND rolling
    # moneyness — at tick t the desk quotes a fresh option with maturity T
    # and strike K_t = strike_ratio * S_t. Constant moneyness keeps the
    # desk's risk profile stationary (a fixed strike would drift out of the
    # money as the underlying trends, silently shrinking the business).
    S0: float = 100.0
    strike_ratio: float = 1.0          # K_t = strike_ratio * S_t
    T: float = 0.25                    # constant time-to-maturity, years
    option: str = "call"

    # TRUE world dynamics.
    true_model: str = "gbm"            # "gbm" | "merton"
    mu: float = 0.06                   # physical drift
    sigma: float = 0.18                # diffusive vol
    lam: float = 0.0                   # jump intensity (per year)
    mu_j: float = 0.0                  # mean log jump
    sigma_j: float = 0.0               # std log jump
    r: float = 0.02                    # risk-free rate

    # Desk's BELIEVED pricing model.
    pricing_model: str = "bs"          # "bs" | "merton"
    pricing_sigma: Optional[float] = None   # defaults to true sigma
    pricing_lam: Optional[float] = None     # defaults to true jump params
    pricing_mu_j: Optional[float] = None
    pricing_sigma_j: Optional[float] = None

    # Market making.
    half_spread: float = 0.15          # $ per option
    fill_prob_ask: float = 0.35        # P(client lifts ask) per tick — desk SELLS
    fill_prob_bid: float = 0.35        # P(client hits bid) per tick — desk BUYS
    quote_size: float = 1.0
    # Flow elasticity: clients are not price-blind. The fill probability is
    # scaled by exp(-elasticity * excess), where excess measures how BAD the
    # quote is for the client relative to TRUE fair value, in units of the
    # half-spread (0 = fairly priced quote at the normal spread). A desk
    # quoting too rich gets no business; a desk quoting too cheap gets run
    # over by informed flow — adverse selection. Set to 0 for price-blind
    # clients (the naive textbook case).
    flow_elasticity: float = 1.0

    # Simulation.
    horizon: float = 1.0               # desk run length, years
    n_ticks: int = 250
    hedge_steps: int = 30              # rebalances over each option's life
    seed: Optional[int] = 42

    def true_merton(self) -> MertonParams:
        return MertonParams(self.sigma, self.lam, self.mu_j, self.sigma_j)

    def believed_pricer(self) -> Pricer:
        sig = self.pricing_sigma if self.pricing_sigma is not None else self.sigma
        if self.pricing_model == "bs":
            return Pricer("bs", self.r, sigma=sig)
        mp = MertonParams(
            sig,
            self.pricing_lam if self.pricing_lam is not None else self.lam,
            self.pricing_mu_j if self.pricing_mu_j is not None else self.mu_j,
            self.pricing_sigma_j if self.pricing_sigma_j is not None else self.sigma_j,
        )
        return Pricer("merton", self.r, merton=mp)

    def true_pricer(self) -> Pricer:
        if self.true_model == "merton":
            return Pricer("merton", self.r, merton=self.true_merton())
        return Pricer("bs", self.r, sigma=self.sigma)


# ---------------------------------------------------------------------------
# True-world path generation
# ---------------------------------------------------------------------------

def simulate_path(cfg: DeskConfig, n: int, dt: float, S0: float,
                  rng: np.random.Generator) -> np.ndarray:
    """One path of the TRUE model on an (n+1)-point grid.

    GBM exact log-Euler step:
        S_{t+dt} = S_t exp[(mu - sigma^2/2) dt + sigma sqrt(dt) Z]
    Merton adds N ~ Poisson(lam dt) jumps with total log size
        Normal(N mu_j, N sigma_j^2), drift-compensated by -lam k dt so `mu`
    stays the net expected growth rate.
    """
    S = np.empty(n + 1)
    S[0] = S0
    k = np.exp(cfg.mu_j + 0.5 * cfg.sigma_j**2) - 1.0
    Z = rng.standard_normal(n)
    base = (cfg.mu - 0.5 * cfg.sigma**2) * dt + cfg.sigma * np.sqrt(dt) * Z
    if cfg.true_model == "merton" and cfg.lam > 0:
        N = rng.poisson(cfg.lam * dt, size=n)
        J = np.where(N > 0,
                     rng.normal(cfg.mu_j * N, cfg.sigma_j * np.sqrt(np.maximum(N, 1))),
                     0.0)
        base += J - cfg.lam * k * dt
    S[1:] = S0 * np.exp(np.cumsum(base))
    return S


# ---------------------------------------------------------------------------
# Realising one trade: delta-hedged life under the true model
# ---------------------------------------------------------------------------

def realize_trade(cfg: DeskConfig, believed: Pricer, S_fill: float, K: float,
                  premium: float, side: str, rng: np.random.Generator) -> float:
    """Discounted P/L of one contract, hedged with the BELIEVED delta while
    the underlying follows the TRUE model.

    side = "sell": desk sold at `premium` (client lifted the ask)
    side = "buy" : desk bought at `premium` (client hit the bid)
    """
    m = cfg.hedge_steps
    dt = cfg.T / m
    path = simulate_path(cfg, m, dt, S_fill, rng)
    n_opt = -1.0 if side == "sell" else 1.0          # signed option position
    pnl = premium if side == "sell" else -premium    # premium cash flow
    growth = np.exp(cfg.r * dt)

    for j in range(m):
        tau = cfg.T - j * dt
        delta = believed.delta(path[j], K, tau, cfg.option)
        h = -n_opt * delta                            # hedge units of underlying
        step = h * (path[j + 1] - path[j] * growth)   # financing-correct increment
        pnl += np.exp(-cfg.r * (j * dt)) * step

    payoff = max(path[-1] - K, 0.0) if cfg.option == "call" else max(K - path[-1], 0.0)
    pnl += np.exp(-cfg.r * cfg.T) * n_opt * payoff
    return float(pnl * cfg.quote_size)


# ---------------------------------------------------------------------------
# Desk run
# ---------------------------------------------------------------------------

@dataclass
class DeskResult:
    t: list = field(default_factory=list)
    S: list = field(default_factory=list)
    bid: list = field(default_factory=list)
    ask: list = field(default_factory=list)
    v_model: list = field(default_factory=list)
    v_truth: list = field(default_factory=list)
    realized: list = field(default_factory=list)     # equity curve
    exp_model: list = field(default_factory=list)    # believed cumulative edge
    exp_truth: list = field(default_factory=list)    # true cumulative edge
    n_trades: list = field(default_factory=list)
    trade_edges: list = field(default_factory=list)   # per-trade realised P/L, in order
    trade_S: list = field(default_factory=list)       # spot at each fill (for scaling)
    summary: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def run_desk(cfg: DeskConfig) -> DeskResult:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_ticks
    dt = cfg.horizon / n
    S = simulate_path(cfg, n, dt, cfg.S0, rng)

    believed = cfg.believed_pricer()
    truth = cfg.true_pricer()

    res = DeskResult()
    realized = exp_model = exp_truth = 0.0
    trades = 0.0

    for i in range(n + 1):
        St = S[i]
        K = cfg.strike_ratio * St        # rolling constant-moneyness strike
        v_m = believed.price(St, K, cfg.T, cfg.option)
        v_t = truth.price(St, K, cfg.T, cfg.option)
        bid, ask = v_m - cfg.half_spread, v_m + cfg.half_spread

        res.t.append(i * dt); res.S.append(float(St))
        res.bid.append(float(bid)); res.ask.append(float(ask))
        res.v_model.append(float(v_m)); res.v_truth.append(float(v_t))
        res.realized.append(float(realized))
        res.exp_model.append(float(exp_model)); res.exp_truth.append(float(exp_truth))
        res.n_trades.append(float(trades))

        if i == n:
            break

        # Elastic client flow: excess = client's disadvantage in half-spreads
        # beyond the "fair" half-spread they normally tolerate.
        s = max(cfg.half_spread, 1e-9)
        exc_ask = (ask - v_t) / s - 1.0    # >0: ask rich; <0: ask cheap (bad for desk)
        exc_bid = (v_t - bid) / s - 1.0
        # Multiplier capped at e^2 (~7.4x): client arrival is finite even
        # when the desk is giving money away.
        m_ask = np.exp(np.clip(-cfg.flow_elasticity * exc_ask, -10.0, 2.0))
        m_bid = np.exp(np.clip(-cfg.flow_elasticity * exc_bid, -10.0, 2.0))
        p_ask = min(0.98, cfg.fill_prob_ask * m_ask)
        p_bid = min(0.98, cfg.fill_prob_bid * m_bid)

        if rng.random() < p_ask:   # client lifts ask -> desk SELLS
            trades += cfg.quote_size
            exp_model += (ask - v_m) * cfg.quote_size
            exp_truth += (ask - v_t) * cfg.quote_size
            pnl_i = realize_trade(cfg, believed, St, K, ask, "sell", rng)
            realized += pnl_i
            res.trade_edges.append(float(pnl_i / cfg.quote_size))
            res.trade_S.append(float(St))
        if rng.random() < p_bid:   # client hits bid -> desk BUYS
            trades += cfg.quote_size
            exp_model += (v_m - bid) * cfg.quote_size
            exp_truth += (v_t - bid) * cfg.quote_size
            pnl_i = realize_trade(cfg, believed, St, K, bid, "buy", rng)
            realized += pnl_i
            res.trade_edges.append(float(pnl_i / cfg.quote_size))
            res.trade_S.append(float(St))

    eq = np.array(res.realized)
    peak = np.maximum.accumulate(eq)
    res.summary = {
        "final_realized": float(eq[-1]),
        "final_exp_model": float(exp_model),
        "final_exp_truth": float(exp_truth),
        "mispricing_gap": float(exp_model - exp_truth),
        "trades": float(trades),
        "avg_edge_per_contract": float(eq[-1] / trades) if trades else 0.0,
        "half_spread": cfg.half_spread,
        "max_drawdown": float(np.max(peak - eq)),
        "pnl_step_vol": float(np.std(np.diff(eq))) if len(eq) > 1 else 0.0,
    }
    return res


# ---------------------------------------------------------------------------
# Multi-seed batches (the statistical object of interest)
# ---------------------------------------------------------------------------

def run_batch(cfg: DeskConfig, n_seeds: int = 200, base_seed: int = 0) -> dict:
    """Run the desk across independent seeds; return per-seed summaries.

    The distribution of `final_realized` across seeds is what the risk and
    validation modules consume: a Monte-Carlo sample of desk-level annual P/L
    under the chosen (true model, believed model) pair.
    """
    finals, edges, drawdowns, trades = [], [], [], []
    for s in range(n_seeds):
        c = DeskConfig(**{**asdict(cfg), "seed": base_seed + s})
        r = run_desk(c)
        finals.append(r.summary["final_realized"])
        edges.append(r.summary["avg_edge_per_contract"])
        drawdowns.append(r.summary["max_drawdown"])
        trades.append(r.summary["trades"])
    return {
        "final_pnl": np.array(finals),
        "avg_edge": np.array(edges),
        "max_drawdown": np.array(drawdowns),
        "trades": np.array(trades),
    }


if __name__ == "__main__":
    matched = DeskConfig(true_model="gbm", pricing_model="bs", seed=7)
    r1 = run_desk(matched)
    print("matched:", {k: round(v, 4) for k, v in r1.summary.items()})

    # Steamroller: jumpy world, BS pricing, clients net BUYERS of options
    # (e.g. crash protection) — the desk is systematically short jump risk.
    steam = DeskConfig(true_model="merton", pricing_model="bs", option="put",
                       lam=4.0, mu_j=-0.12, sigma_j=0.15,
                       fill_prob_ask=0.5, fill_prob_bid=0.1, seed=7)
    r2 = run_desk(steam)
    print("steamroller:", {k: round(v, 4) for k, v in r2.summary.items()})
