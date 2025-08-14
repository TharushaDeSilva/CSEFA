
import streamlit as st
import pandas as pd
from cse_predictor import fit_and_predict_for_ticker, DEFAULT_TICKERS

st.set_page_config(page_title="CSE Market Predictor", page_icon="📈", layout="wide")
st.title("📈 Colombo Stock Exchange Direction Predictor")

st.markdown(
    "Predicts **next-day direction** (UP/DOWN) for selected CSE stocks using a simple "
    "logistic regression on technical indicators (RSI, SMAs, MACD). Data via Yahoo Finance."
)

with st.expander("How to use"):
    st.write(
        "- Pick the tickers (defaults to a starter CSE list).\n"
        "- Click **Run Prediction** to fetch data and build predictions.\n"
        "- Download the results as CSV."
    )

tickers = st.multiselect(
    "Select CSE Tickers:",
    DEFAULT_TICKERS,
    default=DEFAULT_TICKERS,
    help="Edit DEFAULT_TICKERS in cse_predictor.py to change the default list."
)

if st.button("Run Prediction"):
    rows = []
    for t in tickers:
        with st.spinner(f"Processing {t} ..."):
            res = fit_and_predict_for_ticker(t)
            rows.append({
                "Ticker": res.ticker,
                "Last Data Date": res.last_date,
                "Last Close": res.last_close,
                "Prediction": res.pred_dir,
                "Probability Up": round(res.pred_prob_up, 4),
                "Confidence": round(abs(res.pred_prob_up - 0.5), 4),
                "CV Accuracy": round(res.in_sample_acc, 3),
                "CV AUC": round(res.auc, 3),
                "Rows Used": res.n_rows
            })
    df = pd.DataFrame(rows).sort_values(["Prediction", "Confidence"], ascending=[False, False]).reset_index(drop=True)
    st.subheader("Results")
    st.dataframe(df, use_container_width=True)
    st.download_button("📥 Download CSV", df.to_csv(index=False), "cse_predictions.csv")

st.caption("Educational tool only. Not financial advice.")
