import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.spatial.distance import sqeuclidean
import plotly.graph_objects as go

st.set_page_config(page_title="NSE Pairs Trading Dashboard", layout="wide")

st.title("📈 Indian Stock Market: Pairs Trading Strategy")
st.markdown("Replicating the *Gatev, Goetzmann, and Rouwenhorst (2006)* strategy for the NSE.")

# --- 1. CONFIGURATION ---
# Using updated 2026 tickers for the Tata Motors demerger
AUTO_SECTOR = ['TMPV.NS', 'TMCV.NS', 'MARUTI.NS', 'M&M.NS', 'ASHOKLEY.NS']
STEEL_SECTOR = ['TATASTEEL.NS', 'JSWSTEEL.NS', 'SAIL.NS', 'JINDALSTEL.NS']
TICKERS = list(set(AUTO_SECTOR + STEEL_SECTOR))

# --- 2. SIDEBAR PARAMETERS ---
st.sidebar.header("Strategy Settings")
formation_period = st.sidebar.slider("Formation Period (Months)", 6, 24, 12)
trading_period = st.sidebar.slider("Trading Period (Months)", 1, 12, 6)
entry_threshold = st.sidebar.slider("Entry Threshold (Std Dev)", 1.0, 3.0, 2.0)

# --- 3. DATA FETCHING ---
@st.cache_data
def get_market_data(symbols):
    # Fetching enough data for both periods
    data = yf.download(symbols, period="3y", progress=False)['Close']
    return data.ffill().dropna(axis=1)

raw_data = get_market_data(TICKERS)

if raw_data.empty:
    st.error("Could not fetch data. Check your internet or tickers.")
else:
    # --- 4. PAIR FORMATION (Minimum Distance) ---
    st.subheader("🔍 Step 1: Pair Formation")
    
    # Calculate indices for splits
    end_idx = len(raw_data)
    start_idx = end_idx - (formation_period + trading_period) * 21 # approx trading days
    mid_idx = end_idx - trading_period * 21
    
    formation_df = raw_data.iloc[start_idx:mid_idx]
    trading_df = raw_data.iloc[mid_idx:]
    
    # Normalization (Starting at 1.0)
    norm_form = formation_df / formation_df.iloc[0]
    
    pairs_ssd = []
    for s1 in AUTO_SECTOR:
        if s1 in norm_form.columns:
            for s2 in STEEL_SECTOR:
                if s2 in norm_form.columns:
                    ssd = sqeuclidean(norm_form[s1], norm_form[s2])
                    pairs_ssd.append((s1, s2, ssd))
    
    # Identify the best pair
    best_pair = sorted(pairs_ssd, key=lambda x: x[2])[0]
    s1, s2, dist = best_pair
    
    col1, col2 = st.columns(2)
    col1.metric("Top Pair", f"{s1} & {s2}")
    col2.metric("SSD (Min Distance)", f"{dist:.4f}")

    # --- 5. SIGNAL CALCULATION ---
    st.subheader("📊 Step 2: Trading Signals & Visualization")
    
    hist_spread = (formation_df[s1]/formation_df[s1].iloc[0]) - (formation_df[s2]/formation_df[s2].iloc[0])
    st_dev = hist_spread.std()
    
    trade_spread = (trading_df[s1]/trading_df[s1].iloc[0]) - (trading_df[s2]/trading_df[s2].iloc[0])
    upper = entry_threshold * st_dev
    lower = -entry_threshold * st_dev
    
    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trade_spread.index, y=trade_spread, name="Spread", line=dict(color='royalblue')))
    fig.add_hline(y=upper, line_dash="dash", line_color="red", annotation_text=f"+{entry_threshold}σ")
    fig.add_hline(y=lower, line_dash="dash", line_color="green", annotation_text=f"-{entry_threshold}σ")
    fig.add_hline(y=0, line_color="white", opacity=0.5)
    
    fig.update_layout(title=f"Normalized Spread: {s1} vs {s2}", template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # --- 6. ACTION SUMMARY ---
    last_val = trade_spread.iloc[-1]
    
    st.subheader("📝 Trading Summary")
    if abs(last_val) > upper:
        status = "🚨 DIVERGENCE DETECTED"
        if last_val > 0:
            instr = f"Action: SELL {s1} | BUY {s2} (Wait 1 Day for bid-ask bounce)."
        else:
            instr = f"Action: BUY {s1} | SELL {s2} (Wait 1 Day for bid-ask bounce)."
    elif abs(last_val) < 0.1 * st_dev:
        status = "✅ CONVERGENCE"
        instr = "Action: Close existing positions. Prices have returned to parity."
    else:
        status = "⚖️ NEUTRAL"
        instr = "Action: No active signal. Maintain monitoring."

    st.info(f"**Current Status:** {status}\n\n**{instr}**")
