import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.spatial.distance import sqeuclidean
import plotly.graph_objects as go

st.set_page_config(page_title="NSE Pairs Trading Strategy", layout="wide")

# --- HEADER ---
st.title("🛡️ NSE Pairs Trading Strategy Dashboard")
st.markdown("""
Based on the **Gatev, Goetzmann, and Rouwenhorst (2006)** methodology. 
Scanning the Indian market for **Auto** and **Steel** pairs with significant price divergence.
""")

# --- 1. CONFIGURATION & 2026 TICKERS ---
# Using 2026 post-demerger tickers for accuracy
AUTO_SECTOR = ['TMPV.NS', 'TMCV.NS', 'MARUTI.NS', 'M&M.NS', 'ASHOKLEY.NS']
STEEL_SECTOR = ['TATASTEEL.NS', 'JSWSTEEL.NS', 'SAIL.NS', 'JINDALSTEL.NS']
TICKERS = list(set(AUTO_SECTOR + STEEL_SECTOR))

# --- 2. DATA ENGINE ---
@st.cache_data
def get_clean_data(symbols):
    # auto_adjust=True handles 'Close' as the adjusted price
    raw = yf.download(symbols, period="2y", progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        data = raw['Close']
    else:
        data = raw
    return data.ffill().dropna(axis=1)

df = get_clean_data(TICKERS)

# --- 3. PAIR SCANNING & SIGNAL ENGINE ---
if not df.empty:
    formation_df = df.iloc[-504:-126] # 12 months formation
    trading_df = df.iloc[-126:]      # 6 months trading
    
    norm_form = formation_df / formation_df.iloc[0]
    norm_trade = trading_df / trading_df.iloc[0]
    
    all_pairs_results = []
    
    # Process all combinations
    for s1 in AUTO_SECTOR:
        for s2 in STEEL_SECTOR:
            if s1 in df.columns and s2 in df.columns:
                # Minimum Distance (SSD)
                ssd = sqeuclidean(norm_form[s1], norm_form[s2])
                
                # Historical Spread Statistics
                hist_spread = norm_form[s1] - norm_
