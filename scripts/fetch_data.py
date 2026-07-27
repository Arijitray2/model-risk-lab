"""
fetch_data.py — Refresh the datasets from your own machine.

The repository ships ready-to-use CSVs in data/ (documented in data/README.md),
so you only need this script to EXTEND the data — e.g. pull returns up to
today, or add NIFTY option chains from NSE.

Run:  python scripts/fetch_data.py            # refresh underlying histories
      python scripts/fetch_data.py --chains   # also refresh SPY option chain snapshot

Requires:  pip install yfinance pandas
"""

import argparse
import os
import sys

import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def refresh_histories():
    import yfinance as yf
    for ticker, fname in [("SPY", "spy_underlying.csv"),
                          ("^NSEI", "nifty50_history.csv")]:
        h = yf.Ticker(ticker).history(period="max", interval="1d",
                                      auto_adjust=False)
        if h.empty:
            print(f"WARNING: no data for {ticker}"); continue
        h.index = h.index.tz_localize(None)
        df = h.reset_index().rename(columns=str.lower)[
            ["date", "open", "high", "low", "close", "volume"]]
        if ticker == "SPY":
            df["adjusted_close"] = h["Adj Close"].values
        df.to_csv(os.path.join(DATA, fname), index=False)
        print(f"{ticker}: {len(df)} rows -> data/{fname}")


def refresh_spy_chain():
    """Snapshot today's SPY option chain (expiry nearest 90 days)."""
    import yfinance as yf
    t = yf.Ticker("SPY")
    spot = t.history(period="1d")["Close"].iloc[-1]
    today = pd.Timestamp.today().normalize()
    expiries = pd.to_datetime(t.options)
    target = expiries[(expiries - today).days.argsort()]
    exp = min(expiries, key=lambda e: abs((e - today).days - 90))
    ch = t.option_chain(exp.strftime("%Y-%m-%d"))
    rows = []
    for typ, df in (("call", ch.calls), ("put", ch.puts)):
        d = df[["strike", "bid", "ask", "volume", "openInterest",
                "impliedVolatility"]].copy()
        d.columns = ["strike", "bid", "ask", "volume", "open_interest",
                     "implied_volatility"]
        d["type"] = typ
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    out["date"] = today
    out["expiration"] = exp
    out["spot"] = float(spot)
    fname = f"spy_chain_{today.date()}.csv"
    out.to_csv(os.path.join(DATA, fname), index=False)
    print(f"SPY chain {today.date()} exp {exp.date()}: {len(out)} rows -> data/{fname}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", action="store_true",
                    help="also snapshot today's SPY option chain")
    args = ap.parse_args()
    refresh_histories()
    if args.chains:
        refresh_spy_chain()
    print("Done. Re-run scripts/run_experiments.py to regenerate results.")
