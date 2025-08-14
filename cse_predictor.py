import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DEFAULT_TICKERS = ["CARG.N0000.CM", "JKH.N0000.CM", "COMB.N0000.CM", "DIAL.N0000.CM"]

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)

    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/period, adjust=False).mean()

    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def make_features(df):
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["RSI_14"] = compute_rsi(df["Close"], 14)
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)
    return df

def fit_and_predict_for_ticker(ticker):
    data = yf.download(ticker, period="1y")

    # No data handling
    if data.empty or "Close" not in data:
        return type("Result", (), {
            "ticker": ticker,
            "last_date": None,
            "last_close": None,
            "pred_dir": "NO DATA",
            "pred_prob_up": None
        })()

    df = make_features(data)
    X = df[["SMA_5", "SMA_20", "RSI_14"]]
    y = df["target"]

    if len(df) < 30:
        return type("Result", (), {
            "ticker": ticker,
            "last_date": df.index[-1],
            "last_close": df["Close"].iloc[-1],
            "pred_dir": "NOT ENOUGH DATA",
            "pred_prob_up": None
        })()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    last_features = X.iloc[[-1]]
    pred_prob = model.predict_proba(last_features)[0][1]
    pred_dir = "UP" if pred_prob > 0.5 else "DOWN"

    return type("Result", (), {
        "ticker": ticker,
        "last_date": df.index[-1],
        "last_close": df["Close"].iloc[-1],
        "pred_dir": pred_dir,
        "pred_prob_up": pred_prob
    })()
