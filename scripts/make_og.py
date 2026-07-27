"""make_og.py — generate the 1200x630 social-preview card (docs/assets/og.png)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "docs", "assets", "og.png")

BG, SURF, INK, INK2, MUTED = "#0b0f14", "#11161d", "#e8edf2", "#9fb0bf", "#5f6d7a"
BLUE, GREEN, RED, GRID = "#3987e5", "#27c93f", "#e66767", "#1c232c"

fig = plt.figure(figsize=(12, 6.3), dpi=100)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 12); ax.set_ylim(0, 6.3)
ax.axis("off")

# faint grid
for gx in np.arange(0, 12.5, 0.44):
    ax.plot([gx, gx], [0, 6.3], color=BLUE, alpha=0.035, lw=0.8)
for gy in np.arange(0, 6.5, 0.44):
    ax.plot([0, 12], [gy, gy], color=BLUE, alpha=0.035, lw=0.8)

MONO = {"family": "monospace"}

# kicker
ax.text(0.75, 5.55, "● LIVE — QUANT FINANCE · MODEL RISK · REAL MARKET DATA",
        color=GREEN, fontsize=13, **MONO)
# headline
ax.text(0.72, 4.72, "The desk thinks it's", color=INK, fontsize=34, weight="bold")
ax.text(0.72, 4.02, "printing money.", color=INK, fontsize=34, weight="bold")
ax.text(0.72, 3.18, "The market knows better.", color=BLUE, fontsize=34, weight="bold")
# sub
ax.text(0.75, 2.55, "Options market-making under model risk — calibrated to real",
        color=INK2, fontsize=15)
ax.text(0.75, 2.20, "SPY option chains (2008–2025) and 17y of SPY & NIFTY 50 returns.",
        color=INK2, fontsize=15)
# stats strip
ax.text(0.75, 1.45, "LR=1627 vs GBM", color=BLUE, fontsize=15, weight="bold", **MONO)
ax.text(3.30, 1.45, "believed +0.15", color=GREEN, fontsize=15, weight="bold", **MONO)
ax.text(5.85, 1.45, "realised -10.20", color=RED, fontsize=15, weight="bold", **MONO)
# url
ax.text(0.75, 0.72, "arijitray2.github.io/model-risk-lab", color=MUTED, fontsize=14, **MONO)
ax.text(0.75, 0.34, "built by Arijit Ray", color=MUTED, fontsize=12, **MONO)

# mini terminal card with diverging curves (right side)
card = FancyBboxPatch((8.35, 0.55), 3.15, 4.5,
                      boxstyle="round,pad=0.06,rounding_size=0.12",
                      fc=SURF, ec="#2a3441", lw=1.5)
ax.add_patch(card)
for i, c in enumerate(["#ff5f57", "#ffbd2e", "#28c840"]):
    ax.add_patch(plt.Circle((8.62 + i * 0.28, 4.78), 0.07, color=c))
ax.text(9.55, 4.72, "desk.log", color=MUTED, fontsize=10, **MONO)

rng = np.random.default_rng(6)
t = np.linspace(8.6, 11.25, 160)
belief = 3.15 + np.linspace(0, 1.15, 160) + 0.02 * rng.standard_normal(160).cumsum() * 0.08
real = 3.15 + 0.30 * np.linspace(0, 1, 160)
for j, drop in [(46, 0.65), (85, 0.9), (127, 0.75)]:
    real[j:] -= drop
real += 0.05 * rng.standard_normal(160).cumsum() * 0.25
real = np.clip(real, 0.85, None)
ax.plot(t, belief, color=GREEN, lw=2.2, ls="--")
ax.plot(t, real, color=RED, lw=2.4)
ax.text(10.15, belief[-1] + 0.14, "believed", color=GREEN, fontsize=11, **MONO)
ax.text(10.35, real[-1] - 0.30, "realised", color=RED, fontsize=11, **MONO)

fig.savefig(OUT, facecolor=BG)
print("wrote", OUT)
