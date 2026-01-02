import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="Universal Indian Stock Predictor", layout="wide")
st.title("📈 Indian Stock Price Predictor (LSTM)")

# 1. Sidebar for Custom Inputs
st.sidebar.header("Stock Selection")

# Option to choose from list or enter any custom Indian ticker
stock_option = st.sidebar.radio("Input Type", ["Select from List", "Enter Custom Ticker"])

if stock_option == "Select from List":
    stock_dict = {"SBI": "SBIN.NS", "Indian Oil": "IOC.NS", "Adani Power": "ADANIPOWER.NS"}
    selected_stock = st.sidebar.selectbox("Choose Stock", list(stock_dict.keys()))
    ticker = stock_dict[selected_stock]
else:
    # Feature (a): User can enter any stock of their choice
    custom_ticker = st.sidebar.text_input("Enter Ticker (e.g., RELIANCE, TATAMOTORS, HDFCBANK)", "RELIANCE").upper()
    # Auto-append .NS (National Stock Exchange) if not provided by the user
    ticker = custom_ticker if "." in custom_ticker else f"{custom_ticker}.NS"

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# 2. Data Fetching
@st.cache_data
def load_data(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end)
        if df.empty: return None
        return df
    except Exception:
        return None

data = load_data(ticker, start_date, end_date)

if data is not None:
    st.subheader(f"Data Overview: {ticker}")
    st.write(data.tail())

    # 3. Model Preparation &
