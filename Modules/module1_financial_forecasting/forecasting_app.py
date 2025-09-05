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
    st.write("Train simple models (ARIMA / Holt-Winters / Prophet / XGBoost*) on your data or Yahoo Finance.\n" \
    "*Prophet and XGBoost are optional; the app falls back gracefully if not installed.")


    # Data Source Selection
    data_source = st.radio("Select Data Source", ["Yahoo Finance (ticker)", "Upload CSV"], horizontal=True)

    df = None
    # Yahoo Finance Data
    if data_source == "Yahoo Finance (ticker)":
        col1, col2 = st.columns([2, 1])
        with col1:
            ticker = st.text_input("Enter stock ticker (e.g., AAPL, MSFT, ^GSPC)", value="AAPL")
        with col2:
            period = st.selectbox("Select period", ["1y", "2y", "5y", "10y", "max"], index=2)
        if st.button("Load Data", type="primary"):
            with st.spinner("Fetching data..."):
                df = load_data_from_yahoo(ticker, period=period, interval="1d")
                # Error handling
                if df is None or df.empty:
                    st.error("Failed to fetch data. Please check the ticker symbol.") 
                # Display Data
                else:
                    st.success(f"Data for {ticker} fetched successfully!")
                    st.dataframe(df.tail())
    
    # CSV Upload Data
    else:
        uploaded_file = st.file_uploader("Upload your CSV file (columns: date, value)", type=["csv"])
        # If data is uploaded
        if uploaded_file is not None:
            df = load_data_from_csv(uploaded_file)
            # Error handling
            if df is None or df.empty:
                st.error("Could not parse CSV. Expected two columns: date & value. Please check the file format and contents.")
            # Display Data
            else:
                st.success("Data uploaded successfully!")
                st.dataframe(df.head())


    
    # When data is loaded, show options + run models
    if df is not None and not df.empty:
        st.success("Loaded {len(df)} rows from {df['ds'].min().date()} to {df['ds].max().date()}")
        st.line_chart(df.set_index('ds')['y'])

        # Model & Horizon Controls
        with st.expander("⚙️ Model & Forecasting Options", expanded=True):
            horizon_months = st.slider("Forecast horizon (months)", min_value=1, max_value=24, value=6)
            horizon_days = horizon_months * 30
            st.write(f"Forecasting {horizon_days} days into the future.")
            
            test_days = st.slider("Holdout size (days) to compare accuracy", min_value=0, max_value=365, value=30)
            model_choice = st.multiselect(
                "Select forecasting models to run",
                options=["ARIMA(1,1,)", "Holt-Winters", "Prophet*", "XGBoost*"],
                default=["ARIMA(1,1,1)", "Holt-Winters"]    
            )
            
            seasonal_periods = st.number_input("Seasonal periods (for Holt-Winters)", min_value=0, max_value=365, value=0, step=1, 
                                               help="Set to 0 for no seasonality. Set >0 to enable additive seasonality.")
            


        # CFO Manual Scenario for Comparison
        with st.expander("👤 CFO Manual Forecast", expanded=False):
            cfo_mode = st.selectbox("Select CFO Forecasting Assumption", ["Naive flat (last value)", "Monthly Growth Rate (%)"], index=0)
            
            if cfo_mode == "Monthly Growth Rate (%)":
                cfo_growth = st.number_input("Enter expected monthly growth rate (%)", min_value=-100.0, max_value=100.0, value=2.0, step=0.1)            


        # Train / Test Split (chronological)
        train, test = train_test_split(df, test_size=test_days)

        # Fit & Forecast
        results = {}
        # If user selected Holdout>0, compare on that holdout length
        # Else, plot horizon_days
        steps = horizon_days if test_days == 0 else len(test_days)

        with st.spinner("Training models..."):
            # ARIMA
            if "ARIMA(1,1,1)" in model_choice:
                try:
                    arima_model = fit_arima(train, order=(1,1,1))
                    if arima_model is not None:
                        arima_pred = forecast_arima(arima_model, steps=steps)
                        results['ARIMA(1,1,1)'] = arima_pred
                except Exception:
                    st.warning("ARIMA model failed to fit. Check data stationarity and try again.")

            # Holt-Winters
            if "Holt-Winters" in model_choice:
                try:
                    seasonal = None if seasonal_periods == 0 else 'add'
                    hw_model = fit_holtwinters(train, seasonal=seasonal, seasonal_periods=(seasonal_periods or None))                    
                    if hw_model is not None:
                        hw_pred = forecast_holtwinters(hw_model, steps)
                        results['Holt-Winters'] = hw_pred
                except Exception:
                    st.warning("Holt-Winters model failed to fit. Check data and try again.")

            # Prophet (optional)
            if "Prophet*" in model_choice:
                try:
                    prophet_model = fit_prophet(train)
                    if prophet_model is not None:
                        forecast = forecast_prophet(prophet_model, steps, include_history=False)
                        prophet_pred = forecast['yhat'].tail(steps).values
                        results['Prophet*'] = prophet_pred
                except ImportError:
                    st.warning("Prophet model is not installed; skipping.")
                except Exception:
                    st.warning("Prophet model failed to fit. Check data and try again.")

            # XGBoost (optional)
            if "XGBoost*" in model_choice:
                try:
                    xgb_model = fit_xgboost(train, max_lag=7)
                    if xgb_model[0] is not None:
                        history = train['y'].values
                        xgb_pred = forecast_xgboost(xgb_model, steps, history)
                        results['XGBoost*'] = xgb_pred
                except ImportError:
                    st.warning("XGBoost model is not installed; skipping.")
                except Exception:
                    st.warning("XGBoost model failed to fit. Check data and try again.")

            # CFO Manual Baseline
            if cfo_mode == "Monthly Growth Rate (%)":
                cfo_forecast = manual_cfo_growth_rate(train, steps, monthly_growth_pct=float(cfo_growth))
            else:
                cfo_forecast = manual_cfo_naive_last_value(train, steps)
            results['CFO Manual'] = cfo_forecast
                      


        # Accuracy Table if Holdout
        if test_days > 0 and len(test) > 0:
            
            metrics = []
            for model_name, preds in results.items():
                # Compare when lengths match hard-coded holdout length
                if len(preds) != len(test):
                    continue
                
                metrics.append({
                    'Model': model_name,
                    'MAPE (%)': round(mape(test['y'].values, preds), 2),
                    'RMSE': round(rmse(test['y'].values, preds), 4)
                })
            
            if metrics:
                st.subheader("📊 Model Accuracy on Holdout Set")
                st.dataframe(pd.DataFrame(metrics).sort_values('MAPE (%)'))


        # Build Index for Plotting Forecasts
        st.subheader("📅 Forecasts")

        if test_days > 0 and len(test) > 0:
            idx = test['ds'].values
        else:
            # Create future date index by daily increments
            last_date = df['ds'].iloc[-1]
            idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq='D')

        plot_df = pd.DataFrame(index=idx)
        for name, preds in results.items():
            if len(preds) == len(plot_df):
                plot_df[name] = preds

        # Show recent history + forecasts
        history_tail = df.set_index('ds')['y'].tail(180)
        st.line_chart(pd.concat([history_tail.rename("History"), plot_df], axis=1))


        # Allow download of forecast results CSV
        #csv = plot_df.reset_index().rename(columns={'index': 'ds'}).to_csv(index=False).encode('utf-8')
        csv = plot_df.reset_index().rename(columns={'index': 'ds'}).to_csv(index=False)
        st.download_button(
            label="📥 Download Forecasts (CSV)",
            data=csv,
            file_name='forecasts.csv',
            mime='text/csv',
        )

