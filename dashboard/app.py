"""Minimal Streamlit app for processed transaction data."""

from __future__ import annotations

import streamlit as st


st.set_page_config(page_title="Credit Card Transactions Dashboard", layout="wide")
st.title("Credit Card Transactions - Quick Dashboard")

default_path = "data/processed/transactions_parquet"
parquet_path = st.text_input("Parquet path", value=default_path)

if st.button("Load data"):
    try:
        import pandas as pd

        df = pd.read_parquet(parquet_path)
        st.success(f"Loaded {len(df):,} rows")
        st.dataframe(df.head(50), use_container_width=True)
        if "is_fraud" in df.columns:
            st.subheader("Fraud label distribution")
            st.bar_chart(df["is_fraud"].value_counts().sort_index())
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load data: {exc}")
