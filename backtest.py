"""Walk-forward backtesting CLI.

Example: python backtest.py --ticker AAPL --model xgboost --folds 4
"""
import argparse

import yfinance as yf

from src.backtest import MODELS, walk_forward


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--model", choices=MODELS + ["all"], default="lstm")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    a = p.parse_args()

    close = yf.download(a.ticker, start=a.start, end=a.end, auto_adjust=True, progress=False)["Close"].dropna().values
    for name in (MODELS if a.model == "all" else [a.model]):
        _, folds, overall = walk_forward(close, name, a.lookback, a.folds, a.epochs)
        print(f"\n== {name} on {a.ticker} ==")
        print(folds.round(3).to_string(index=False))
        print("Overall:", {k: round(v, 3) for k, v in overall.items()})


if __name__ == "__main__":
    main()
