# forecasting_models.py
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt

def load_data(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d").reset_index()
        df = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def run_prophet_model(df, months):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=months * 30)
    forecast = model.predict(future)
    return model, forecast

def plot_forecast(model, forecast):
    return model.plot(forecast)
