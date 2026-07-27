# Data provenance

All files are real market data obtained from public mirrors; nothing is synthetic.

| File | Contents | Source |
|---|---|---|
| `spy_underlying.csv` | SPY daily OHLC + adjusted close, 1999-11 → 2025-12 (6,570 rows) | [lambdaclass/options_backtester `data-v1`](https://github.com/lambdaclass/options_backtester/releases/tag/data-v1) release (MIT-licensed repo; mirror of the philippdubach options dataset), `SPY_underlying.parquet` |
| `spy_chain_panel_monthend.csv` | End-of-day SPY option quotes, last trading day of each month 2008-01 → 2025-12, expiries 55–125 DTE, liquidity-filtered (bid > 0, OI or volume > 0). 145,043 rows: bid/ask/mark/last, volume, OI, implied vol, delta, spot | Extracted from `SPY_options.parquet` (24.7M rows, SHA-256 verified `a715...41f0a`) of the same `data-v1` release |
| `spy_chain_2017-01-03.csv` `spy_chain_2020-03-16.csv` `spy_chain_2022-06-13.csv` `spy_chain_2025-06-30.csv` | Full-surface single-day chain snapshots (all expiries), bid > 0 | Same source |
| `nifty50_history.csv` | NIFTY 50 index daily OHLC, 2007-09 → 2026-04 (4,554 rows) | [kalilurrahman/NIFTY_50_STOCK_DATA](https://github.com/kalilurrahman/NIFTY_50_STOCK_DATA) (`NIFTY50_stock_history.csv`, index history) |
| `us_risk_free_rate.csv` | US risk-free rate (annualised %), 2007 → 2025 | [SteelCerberus/us-market-data](https://github.com/SteelCerberus/us-market-data) (MIT) |

Refresh or extend with `python scripts/fetch_data.py` (uses yfinance; run from your
own machine). NSE publishes NIFTY option chains at nseindia.com if you want to add
an Indian risk-neutral calibration leg.

Data is redistributed here for reproducibility of an educational research project.
