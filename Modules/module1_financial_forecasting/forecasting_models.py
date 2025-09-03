# MODEL UTILS
# Utilities for data loading and forecasting models.
# Robust to missing libraries; Prophet and XGBoost are optional.

from __future__ import annotations

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Optional Imports
# Yahoo Finance
try:
    import yfinance as yf
except Exception:
    yf = None
# Prophet
try:
    from prophet import Prophet
except Exception:
    Prophet = None
# XGBoost
try:
    import xgboost as xgb
except Exception:
    xgb = None
#ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:
    ARIMA = None




# Load Financial Data from Yahoo Finance
# This function fetches historical stock data for a given ticker
def load_data(ticker):
    """ 
    Load historical stock data from Yahoo Finance if yfinance is available.
    Returns a DataFrame with historical stock data with columns ['ds','y'] suitable for Prophet-style models."""
    try:
        df = yf.download(ticker, period="2y", interval="1d").reset_index()
        df = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Run Prophet Model
# This function fits a Prophet model to the data and returns the model and forecast for a specified number of months
def run_prophet_model(df, months):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=months * 30)
    forecast = model.predict(future)
    return model, forecast

# Plot Forecast
# This function plots the forecast using the Prophet model
def plot_forecast(model, forecast):
    return model.plot(forecast)
