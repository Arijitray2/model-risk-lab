# Model Risk Lab

**What a wrong pricing model costs an options market-making desk — in dollars.**
Calibrated to real SPY & NIFTY 50 data, with a full statistical model-validation suite.

**Live site:** https://arijitray2.github.io/model-risk-lab/ · **Report:** [`model_risk_lab_report.pdf`](model_risk_lab_report.pdf)

---

## The idea

A market maker earns the bid–ask spread by quoting around *fair value* — but fair
value comes from a model, and the model can be wrong. This project makes that danger
measurable end to end:

1. **Calibrate** — MLE of GBM and Merton jump-diffusion on 17+ years of SPY and
   NIFTY 50 daily returns (with standard errors, LR tests, AIC/BIC, QQ plots), and
   risk-neutral Merton calibration to four **real SPY option chains** including
   16 March 2020 (VIX 82), where the market priced λ = 3.7 crashes/year of mean −31%.
2. **Simulate** — a rolling constant-maturity options desk quotes bid/ask around its
   *believed* fair value; every fill's delta-hedged life is realised under the *true*
   model. Client flow is **price-elastic**: mispriced quotes attract informed flow
   (adverse selection) instead of being ignored.
3. **Measure** — VaR / Expected Shortfall / drawdowns of desk P/L across hundreds of
   seeds, and a **model-risk premium** via break-even spread search: the matched desk
   is 95%-safe from an 8.8¢ half-spread; the mispriced desk isn't safe below $15.
4. **Validate** — Kupiec & Christoffersen VaR backtests (mispriced desk: 811
   exceptions vs 75 expected, p < 1e−16), bootstrap CIs on realised edge, and a CUSUM
   monitor that alarms at trade 83 of 1,467 with zero false alarms on the matched desk.

Headline numbers: the correctly-specified desk realises **+0.153 per contract**
(promised 0.15, bootstrap 95% CI [0.146, 0.159]); the same desk pricing
Black–Scholes in the jump world *believes* +0.15 while realising **−10.20**.

## Repository layout

```
mrlab/                  core library
  pricing.py            Black–Scholes & Merton pricers, greeks, implied vol
  simulate.py           the market-making desk under model risk
  calibrate.py          physical-measure MLE + risk-neutral smile fitting
  risk.py               VaR, ES, drawdowns, break-even spread
  validate.py           Kupiec, Christoffersen, bootstrap, CUSUM
data/                   real market data + provenance (see data/README.md)
scripts/
  run_experiments.py    reproduces every number (→ results/*.json)
  make_figures.py       reproduces every figure (→ results/figures/)
  fetch_data.py         refresh data from your own machine
tests/test_all.py       15 unit tests
results/                JSON outputs + figures
docs/                   the website (GitHub Pages) incl. interactive JS simulator
report/                 LaTeX source of the PDF report
```

## Reproduce everything

```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # 15 tests
python scripts/run_experiments.py   # ~15 min: calibrations, experiments, validation
python scripts/make_figures.py      # all figures
```

Everything is seeded — results reproduce exactly.

## Run the website locally

```bash
cd docs && python -m http.server 8000
# open http://localhost:8000
```

The live demo ports the full engine (seeded RNG, Merton series pricer, hedged-fill
realisation, elastic flow) to vanilla JavaScript — no build step, no server.

## Acknowledgements

The seed of this project is the market-making teaching simulator by
[Roman Paolucci (Quant Guild)](https://www.youtube.com/watch?v=swPOLhSIBHo) — the
code here is an independent rewrite that adds real-data calibration, elastic flow,
rolling-moneyness quoting, multi-seed risk measurement and the validation layer.
Data sources and licenses are documented in [`data/README.md`](data/README.md).

Educational research project — not trading or investment advice.

## Author

**Arijit Ray** — MSc Statistics, IIT Bombay.
