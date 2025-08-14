import streamlit as st
import pandas as pd
from cse_predictor import fit_and_predict_for_ticker, DEFAULT_TICKERS

st.set_page_config(page_title="CSE Stock Predictor", layout="wide")

st.title("Colombo Stock Exchange Stock Predictor 📈📉")
st.write("This app predicts whether selected CSE shares will go **UP** or **DOWN** tomorrow based on historical trends.")

tickers = st.multiselect("Select CSE Tickers", DEFAULT_TICKERS, DEFAULT_TICKERS)

if st.button("Predict"):
    results = []
    for t in tickers:
        res = fit_and_predict_for_ticker(t)
        results.append({
            "Ticker": res.ticker,
            "Last Date": res.last_date,
            "Last Close": res.last_close,
            "Prediction": res.pred_dir,
            "Probability Up": round(res.pred_prob_up, 4) if res.pred_prob_up is not None else None
        })
    df_results = pd.DataFrame(results)
    st.dataframe(df_results)
