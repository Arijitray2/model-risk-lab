"""
make_figures.py — Publication figures from results/*.json.

Outputs PNG (300dpi, for the PDF report) and SVG (for the website) into
results/figures/. Styling follows a validated light-mode palette:
series blue/orange/aqua in fixed order, red reserved for losses.

Usage: python scripts/make_figures.py   (after run_experiments.py)
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

# --- palette (validated; see docs) -----------------------------------------
BLUE, ORANGE, AQUA, RED, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#4a3aa7"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "font.family": "sans-serif", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
})


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"),
                    dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig: {name}")


phys = load("calibration_physical.json")
rn = load("calibration_riskneutral.json")
desk = load("desk_experiments.json")
be = load("breakeven.json")
val = load("validation.json")


# ---------------------------------------------------------------------------
# 1. Return densities (log scale): empirical vs GBM vs Merton
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, key in zip(axes, ("spy", "nifty")):
    d = phys[key]["density"]
    ax.plot(d["hist_x"], d["hist_y"], lw=0, marker="o", ms=3, color=INK2,
            alpha=0.55, label="Empirical")
    ax.plot(d["grid"], d["gbm"], color=ORANGE, label="GBM (Gaussian)")
    ax.plot(d["grid"], d["merton"], color=BLUE, label="Merton jump-diffusion")
    ax.set_yscale("log")
    ax.set_ylim(bottom=max(min([y for y in d["hist_y"] if y > 0]) * 0.5, 1e-3))
    ax.set_title(phys[key]["label"])
    ax.set_xlabel("daily log-return")
    if key == "spy":
        ax.set_ylabel("density (log scale)")
        ax.legend(loc="lower center", fontsize=8.5)
save(fig, "fig1_densities")

# ---------------------------------------------------------------------------
# 2. QQ plots, SPY: GBM vs Merton
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, which, color in ((axes[0], "qq_gbm", ORANGE), (axes[1], "qq_merton", BLUE)):
    q = phys["spy"][which]
    lim = [min(q["theoretical"] + q["empirical"]), max(q["theoretical"] + q["empirical"])]
    ax.plot(lim, lim, color=MUTED, lw=1, ls="--")
    ax.plot(q["theoretical"], q["empirical"], lw=0, marker="o", ms=3.5,
            color=color, alpha=0.8)
    ax.set_title("GBM quantiles" if which == "qq_gbm" else "Merton quantiles")
    ax.set_xlabel("model quantile")
    ax.set_ylabel("empirical quantile" if which == "qq_gbm" else "")
fig.suptitle("SPY daily returns: which model owns the tails?", y=1.02,
             fontsize=11, fontweight="bold")
save(fig, "fig2_qq_spy")

# ---------------------------------------------------------------------------
# 3. Implied-volatility smiles, four real SPY snapshots + Merton fit
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, (date, s) in zip(axes.ravel(), rn.items()):
    m = np.array(s["strikes"]) / s["spot"]
    order = np.argsort(m)
    ax.plot(m[order], (np.array(s["market_iv"]) * 100)[order], lw=0, marker="o",
            ms=3.5, color=INK2, alpha=0.6, label="Market IV (OTM quotes)")
    ax.plot(m[order], (np.array(s["merton_iv"]) * 100)[order], color=BLUE,
            label="Merton fit")
    ax.axhline(s["atm_iv"] * 100, color=ORANGE, lw=1.6, ls="--",
               label="Flat Black–Scholes (ATM IV)")
    ax.set_title(f"{date} — {s['label']}  ({s['dte']} DTE)")
    ax.set_xlabel("moneyness  K / S")
    ax.set_ylabel("implied vol (%)")
    p = s["merton_params"]
    ax.text(0.02, 0.04,
            (f"$\\sigma$={p['sigma']:.3f}  $\\lambda$={p['lam']:.2f}/yr\n"
             f"$\\mu_J$={p['mu_j']:.3f}  $\\sigma_J$={p['sigma_j']:.3f}\n"
             f"RMSE {s['rmse_iv']*100:.2f} vol pts"),
            transform=ax.transAxes, fontsize=7.5, color=INK2, va="bottom")
axes[0, 0].legend(fontsize=8, loc="upper right")
fig.tight_layout()
save(fig, "fig3_smiles")

# ---------------------------------------------------------------------------
# 4. Equity curves: matched vs steamroller (three P/L views)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
for ax, name, title in ((axes[0], "matched", "Matched model: BS desk in a GBM world"),
                        (axes[1], "steamroller", "Model risk: BS desk in a jump world")):
    c = desk[name]["curves"][0]
    ax.plot(c["t"], c["exp_model"], color=AQUA, ls="--",
            label="Edge the desk believes (model)")
    ax.plot(c["t"], c["exp_truth"], color=ORANGE, ls="--",
            label="True expected edge")
    ax.plot(c["t"], c["realized"], color=BLUE, label="Realised P/L")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("time (years)")
axes[0].set_ylabel("cumulative P/L ($ per unit desk)")
axes[0].legend(fontsize=8.5, loc="upper left")
save(fig, "fig4_equity")

# ---------------------------------------------------------------------------
# 5. Distribution of one-year desk P/L across seeds (all four scenarios)
# ---------------------------------------------------------------------------
names = ["matched", "right_model", "steamroller", "overcautious"]
labels = ["Matched\n(BS in GBM world)", "Right model\n(Merton in jump world)",
          "Steamroller\n(BS in jump world)", "Overcautious\n(Merton in GBM world)"]
colors = [BLUE, AQUA, RED, ORANGE]
fig, ax = plt.subplots(figsize=(10, 4.6))
data = [np.array(desk[n]["final_pnl"]) for n in names]
parts = ax.violinplot(data, showextrema=False, widths=0.85)
for b, col in zip(parts["bodies"], colors):
    b.set_facecolor(col); b.set_alpha(0.55); b.set_edgecolor(col)
for i, (d, col) in enumerate(zip(data, colors)):
    q1, med, q3 = np.percentile(d, [25, 50, 75])
    ax.vlines(i + 1, q1, q3, color=col, lw=5, alpha=0.9)
    ax.plot(i + 1, med, "o", ms=5, color=SURFACE, mec=col, mew=1.5, zorder=5)
    ax.text(i + 1, np.max(d), f"mean {np.mean(d):,.0f}", ha="center",
            va="bottom", fontsize=8, color=INK2)
ax.axhline(0, color=MUTED, lw=1)
ax.set_xticks(range(1, 5)); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("one-year desk P/L across seeds ($)")
ax.set_title("The same desk, four model assumptions — P/L distribution over "
             f"{desk['matched']['n_seeds']}+ seeds")
ax.set_yscale("symlog", linthresh=100)
save(fig, "fig5_pnl_distributions")

# ---------------------------------------------------------------------------
# 6. Break-even spread curves (the model-risk premium)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.4))
m, s = be["matched"], be["steamroller"]
ax.plot(m["spreads"], np.array(m["prob_loss"]) * 100, color=BLUE, marker="o",
        ms=4, label="Matched model")
ax.plot(s["spreads"], np.array(s["prob_loss"]) * 100, color=RED, marker="o",
        ms=4, label="Mispriced desk (BS in jump world)")
ax.axhline(5, color=MUTED, lw=1, ls="--")
ax.text(0.05, 6.5, "5% loss-probability target", fontsize=8, color=MUTED)
if m["breakeven_half_spread"]:
    ax.axvline(m["breakeven_half_spread"], color=BLUE, lw=1, ls=":")
    ax.text(m["breakeven_half_spread"] * 1.15, 55,
            f"matched breaks even\nat {m['breakeven_half_spread']:.3f}",
            fontsize=8, color=BLUE)
ax.set_xscale("log")
ax.set_xlabel("quoted half-spread ($, log scale)")
ax.set_ylabel("P(one-year P/L < 0)  (%)")
ax.set_title("What spread buys safety? The model-risk premium in dollars")
ax.legend(fontsize=9)
save(fig, "fig6_breakeven")

# ---------------------------------------------------------------------------
# 7. Validation: CUSUM monitor paths
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.2))
for name, col, lab in (("matched", BLUE, "Matched desk"),
                       ("steamroller", RED, "Mispriced desk")):
    c = val[f"cusum_{name}"]
    ax.plot(range(len(c["cusum"])), c["cusum"], color=col, label=lab)
    if name == "steamroller":
        ax.axhline(c["threshold"], color=MUTED, lw=1, ls="--")
        ax.text(len(c['cusum']) * 0.4, c["threshold"] * 1.05,
                "alarm threshold h", fontsize=8, color=MUTED)
        if c["alarm_at_trade"]:
            ax.axvline(c["alarm_at_trade"] - c["burn_in"], color=RED, lw=1, ls=":")
            ax.annotate(f"alarm at trade {c['alarm_at_trade']}",
                        xy=(c["alarm_at_trade"] - c["burn_in"], c["threshold"]),
                        xytext=(c["alarm_at_trade"] - c["burn_in"] + 60,
                                c["threshold"] * 2.2),
                        fontsize=8.5, color=RED,
                        arrowprops=dict(arrowstyle="->", color=RED, lw=1))
ax.set_yscale("symlog", linthresh=1)
ax.set_xlabel("trades since monitoring start")
ax.set_ylabel("CUSUM statistic (spot-scaled)")
ax.set_title("Sequential monitoring catches the broken model early")
ax.legend(fontsize=9, loc="center right")
save(fig, "fig7_cusum")

# ---------------------------------------------------------------------------
# 8. VaR exceptions timeline (clustering, matched vs steamroller)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(10, 3.4), sharex=True)
for ax, name, col, lab in ((axes[0], "matched", BLUE, "Matched"),
                           (axes[1], "steamroller", RED, "Mispriced")):
    e = np.array(val[f"exceptions_{name}"])
    k = val[f"kupiec_{name}"]
    idx = np.where(e == 1)[0]
    ax.vlines(idx, 0, 1, color=col, lw=0.7, alpha=0.8)
    ax.set_yticks([])
    ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=9, color=INK2)
    ax.text(0.995, 0.72, f"{k['exceptions']} exceptions / {k['n']} ticks "
            f"(expected {k['expected']:.0f})   Kupiec p = {k['p_value']:.3f}",
            transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
    ax.grid(False)
axes[1].set_xlabel("tick (6-year desk run)")
axes[0].set_title("5% VaR exceptions: correct coverage vs clustered blow-through")
save(fig, "fig8_var_exceptions")

print("All figures written to results/figures/")
