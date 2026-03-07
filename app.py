import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.spatial.distance import sqeuclidean
import plotly.graph_objects as go

st.set_page_config(page_title="NSE Pairs Trading Detector", layout="wide")

st.title("🏹 NSE Pairs Trading: Buy/Sell Signal Detector")

# --- 1. CONFIGURATION ---
# Using 2026 tickers for the Tata Motors demerger (TMPV and TMCV)
AUTO_SECTOR = ['TMPV.NS', 'TMCV.NS', 'MARUTI.NS', 'M&M.NS', 'ASHOKLEY.NS']
STEEL_SECTOR = ['TATASTEEL.NS', 'JSWSTEEL.NS', 'SAIL.NS', 'JINDALSTEL.NS']
TICKERS = list(set(AUTO_SECTOR + STEEL_SECTOR))

# --- 2. DATA ENGINE ---
@st.cache_data
def get_data(symbols):
    # auto_adjust=True (default) ensures we get dividend-adjusted 'Close'
    raw_data = yf.download(symbols, period="2y", progress=False)
    # Handle MultiIndex returned by yfinance
    if 'Close' in raw_data.columns.get_level_values(0):
        data = raw_data['Close']
    else:
        data = raw_data
    return data.ffill().dropna(axis=1)

df = get_data(TICKERS)

if not df.empty:
    # 12m Formation / 6m Trading
    formation_df = df.iloc[-504:-126]
    trading_df = df.iloc[-126:]
    
    norm_form = formation_df / formation_df.iloc[0]
    norm_trade = trading_df / trading_df.iloc[0]
