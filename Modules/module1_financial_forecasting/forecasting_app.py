# Streamlit UI for Module 1: Financial Forecasting
from __future__ import annotations

# Import necessary libraries
import pandas as pd
import numpy as np
import streamlit as st

# Import Models
from forecasting_models import (
    load_data_from_yahoo,
    load_data_from_csv,
    train_test_split,
    fit_arima, forecast_arima,
    fit_holtwinters, forecast_holtwinters,
    fit_prophet, forecast_prophet,
    fit_xgboost, forecast_xgboost,
    mape, rmse,
    manual_cfo_naive_last_value, manual_cfo_growth_rate,
)

# Function to run the forecasting module
# This function sets up the Streamlit app for financial forecasting
def run_forecasting_module():
    st.header("📈 Financial Forecasting")
    st.subheader("Forecast revenues, costs, financials and key ratios.")
    st.caption("Upload Data or use Demo mode, then generate forecasts.")

    ticker = st.text_input("Enter desired stock ticker (e.g., AAPL)", "AAPL")
    months = st.slider("Forecast horizon (months)", 1, 12, 3)

    if st.button("Run Forecast"):
        with st.spinner("Fetching data and running model..."):
            df = load_data(ticker)
            if df is not None:
                model, forecast = run_prophet_model(df, months)
                st.subheader("Forecast Chart")
                fig1 = plot_forecast(model, forecast)
                st.pyplot(fig1)

                st.subheader("Forecast Components")
                fig2 = model.plot_components(forecast)
                st.pyplot(fig2)
            else:
                st.error("Failed to fetch data.")
