import streamlit as st
import pandas as pd

from afrikana_risk.credit import CreditScorer
from afrikana_risk.fraud import FraudDetector
from afrikana_risk.monitoring import ModelMonitor

st.title("Afrikana Risk Intelligence Platform")

# Upload data
uploaded_file = st.file_uploader("Upload portfolio data", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview")
    st.dataframe(df.head())

    if st.button("Run Credit Scoring"):
        scorer = CreditScorer()
        scorer.fit(df)
        scored = scorer.score(df)

        st.subheader("Credit Scores")
        st.dataframe(scored.head())

    if st.button("Run Fraud Detection"):
        detector = FraudDetector()
        detector.fit(df)
        fraud = detector.score(df)

        st.subheader("Fraud Results")
        st.dataframe(fraud.head())