import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.spatial.distance import sqeuclidean
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import plotly.graph_objects as go

# --- APP CONFIGURATION ---
st.set_page_config(page_title="NSE Pro Pairs Trader 2026", layout="wide")

# --- TICKER CONFIG (2026 DEMERGER) ---
AUTO_SECTOR = ['TMPV.NS', 'TMCV.NS', 'MARUTI.NS', 'M&M.NS', 'ASHOKLEY.NS']
STEEL_SECTOR = ['TATASTEEL.NS', 'JSWSTEEL.NS', 'SAIL.NS', 'JINDALSTEL.NS']
ALL_TICKERS = list(set(AUTO_SECTOR + STEEL_SECTOR))

@st.cache_data
def get_clean_data(symbols, period="2y"):
    raw = yf.download(symbols, period=period, progress=False)
    # yfinance 0.2.50+ returns 'Close' as the adjusted price by default
    data = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
    return data.ffill().dropna(axis=1)

# --- UI TABS ---
tab_ssd, tab_coint, tab_instr = st.tabs(["🔍 SSD Discovery (Industry)", "📊 Cointegration (Statistical)", "📖 How to Use"])

# --- TAB 1: SSD DISTANCE MODEL ---
with tab_ssd:
    st.subheader("Industry Discovery: Auto vs Steel")
    df_ssd = get_clean_data(ALL_TICKERS)
    
    if not df_ssd.empty:
        # 12m Formation / 6m Trading
        form_df = df_ssd.iloc[-504:-126]
        trade_df = df_ssd.iloc[-126:]
        
        norm_form = form_df / form_df.iloc[0]
        norm_trade = trade_df / trade_df.iloc[0]
        
        ssd_results = []
        for s1 in AUTO_SECTOR:
            for s2 in STEEL_SECTOR:
                if s1 in df_ssd.columns and s2 in df_ssd.columns:
                    ssd = sqeuclidean(norm_form[s1], norm_form[s2])
                    h_std = (norm_form[s1] - norm_form[s2]).std()
                    curr_spread = norm_trade[s1].iloc[-1] - norm_trade[s2].iloc[-1]
                    
                    # Entry Logic (2 Sigma)
                    action_s1, action_s2 = "HOLD", "HOLD"
                    color_s1, color_s2 = "gray", "gray"
