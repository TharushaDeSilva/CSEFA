#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from typing import List, Tuple
import os
import pandas as pd
import numpy as np

import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

DEFAULT_TICKERS = [
    "JKH.N0000.CM",
    "COMB.N0000.CM",
    "DIAL.N0000.CM"
]


START_DATE = "2018-01-01"
END_DATE = None
MIN_ROWS = 400

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret_1"] = df["Close"].pct_change()
    for lag in [1, 2, 3, 4, 5]:
        df[f"ret_lag{lag}"] = df["ret_1"].shift(lag)
    df["sma_5"] = df["Close"].rolling(5).mean()
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()
    df["sma_5_20_diff"] = (df["sma_5"] - df["sma_20"]) / df["sma_20"]
    df["sma_20_50_diff"] = (df["sma_20"] - df["sma_50"]) / df["sma_50"]
    df["rsi_14"] = compute_rsi(df["Close"], 14)
    macd, sig = compute_macd(df["Close"])
    df["macd"] = macd
    df["macd_signal"] = sig
    df["macd_hist"] = macd - sig
    df["target_up"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df.dropna()

@dataclass
class TickerResult:
    ticker: str
    last_date: str
    last_close: float
    pred_prob_up: float
    pred_dir: str
    in_sample_acc: float
    auc: float
    n_rows: int

def fit_and_predict_for_ticker(ticker):
    data = yf.download(ticker, period="1y")

    # Check if there's data
    if data.empty or "Close" not in data:
        return type("Result", (), {
            "ticker": ticker,
            "last_date": None,
            "last_close": None,
            "pred_dir": "NO DATA",
            "pred_prob_up": None
        })()

    df = make_features(data)

    features = [
        "ret_lag1","ret_lag2","ret_lag3","ret_lag4","ret_lag5",
        "sma_5_20_diff","sma_20_50_diff","rsi_14","macd","macd_signal","macd_hist"
    ]
    X = df[features].values
    y = df["target_up"].values

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, roc_auc_score
    import numpy as np

    tscv = TimeSeriesSplit(n_splits=5)
    probs = np.zeros(len(y))
    accs, aucs = [], []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model = LogisticRegression(max_iter=200, class_weight="balanced")
        model.fit(X_train, y_train)
        p = model.predict_proba(X_test)[:, 1]
        probs[test_idx] = p
        accs.append(accuracy_score(y_test, (p >= 0.5).astype(int)))
        try:
            aucs.append(roc_auc_score(y_test, p))
        except:
            aucs.append(np.nan)

    final_model = LogisticRegression(max_iter=200, class_weight="balanced")
    final_model.fit(X, y)
    last_row = X[-1].reshape(1, -1)
    next_prob = float(final_model.predict_proba(last_row)[:, 1][0])

    pred_dir = "UP" if next_prob >= 0.5 else "DOWN"
    last_date = str(df.index[-1].date())
    last_close = float(data["Close"].iloc[-1]) if not data.empty else float("nan")

    return TickerResult(
        ticker=ticker,
        last_date=last_date,
        last_close=last_close,
        pred_prob_up=next_prob,
        pred_dir=pred_dir,
        in_sample_acc=float(np.nanmean(accs)) if accs else float("nan"),
        auc=float(np.nanmean(aucs)) if aucs else float("nan"),
        n_rows=len(df)
    )

if __name__ == "__main__":
    # Optional: allow running as a script to generate a CSV quickly
    import pandas as pd
    results = []
    for t in DEFAULT_TICKERS:
        res = fit_and_predict_for_ticker(t)
        results.append({
            "ticker": res.ticker,
            "last_data_date": res.last_date,
            "last_close": res.last_close,
            "pred_prob_up": round(res.pred_prob_up, 4),
            "prediction": res.pred_dir,
            "cv_accuracy": round(res.in_sample_acc, 3),
            "cv_auc": round(res.auc, 3),
            "n_rows": res.n_rows
        })
    out = pd.DataFrame(results).sort_values(["prediction", "pred_prob_up"], ascending=[False, False])
    out.to_csv("cse_signals.csv", index=False)
    print(out)
