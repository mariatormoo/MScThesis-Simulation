# forecasting_app.py
import streamlit as st
from .forecasting_models import load_data, run_prophet_model, plot_forecast

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
