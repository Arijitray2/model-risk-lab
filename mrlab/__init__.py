"""
mrlab — Model Risk Lab
======================

A research codebase that measures, in dollars, what happens when a
derivatives desk prices and hedges with the wrong model — and provides the
statistical toolkit a model validator would use to detect it.

Modules
-------
pricing    : Black–Scholes and Merton jump-diffusion pricers + greeks + implied vol
simulate   : options market-making desk simulator under true/assumed model pairs
calibrate  : maximum-likelihood (physical) and chain-implied (risk-neutral) calibration
risk       : VaR / Expected Shortfall / drawdowns / model-risk premium
validate   : Kupiec & Christoffersen VaR backtests, bootstrap edge inference, monitoring
"""

__version__ = "1.0.0"
