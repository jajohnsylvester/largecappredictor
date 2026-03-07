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
# Fixed the missing quote in JINDALSTEL.NS
AUTO_SECTOR = ['TMPV.NS', 'TMCV.NS', 'MARUTI.NS', 'M&M.NS', 'ASHOKLEY.NS']
STEEL_SECTOR = ['TATASTEEL.NS', 'JSWSTEEL.NS', 'SAIL.NS', 'JINDALSTEL.NS']
TICKERS = list(set(AUTO_SECTOR + STEEL_SECTOR))

# --- 2. DATA ENGINE ---
@st.cache_data
def get_clean_data(symbols):
    # auto_adjust=True (default) ensures we get dividend-adjusted 'Close'
    raw = yf.download(symbols, period="2y", progress=False)
    
    # Handle MultiIndex returned by yfinance
    if isinstance(raw.columns, pd.MultiIndex):
        data = raw['Close']
    else:
        data = raw
    
    return data.ffill().dropna(axis=1)

df = get_clean_data(TICKERS)

# --- 3. PAIR SCANNING & SIGNAL LOGIC ---
if not df.empty:
    formation_df = df.iloc[-504:-126] # 12 months
    trading_df = df.iloc[-126:]      # 6 months
    
    norm_form = formation_df / formation_df.iloc[0]
    norm_trade = trading_df / trading_df.iloc[0]
    
    all_results = []
    
    for s1 in AUTO_SECTOR:
        for s2 in STEEL_SECTOR:
            if s1 in df.columns and s2 in df.columns:
                ssd = sqeuclidean(norm_form[s1], norm_form[s2])
                hist_spread = norm_form[s1] - norm_form[s2]
                h_std = hist_spread.std()
                threshold = 2 * h_std
                
                t_series = norm_trade[s1] - norm_trade[s2]
                current_val = t_series.iloc[-1]
                
                # Determine specific Buy/Sell Action
                if current_val > threshold:
                    action = f"SELL {s1} / BUY {s2}"
                    status = "DIVERGED (High)"
                elif current_val < -threshold:
                    action = f"BUY {s1} / SELL {s2}"
                    status = "DIVERGED (Low)"
                elif abs(current_val) < (0.1 * h_std):
                    action = "EXIT (Convergence)"
                    status = "CONVERGED"
                else:
                    action = "Wait/Neutral"
                    status = "STABLE"

                all_results.append({
                    'Pair': f"{s1} vs {s2}",
                    'SSD': ssd,
                    'Current Spread': round(current_val, 4),
                    'Threshold': round(threshold, 4),
                    'Action': action,
                    'Status': status,
                    'S1': s1, 'S2': s2, 'limit': threshold,
                    'series': t_series
                })

    sorted_pairs = sorted(all_results, key=lambda x: x['SSD'])
    top_pair = sorted_pairs[0]

    # --- 4. VISUALIZATION WITH LABELS ---
    st.subheader(f"📈 Top Pair Analysis: {top_pair['Pair']}")
    
    fig = go.Figure()
    s_data = top_pair['series']
    lim = top_pair['limit']
    
    fig.add_trace(go.Scatter(x=s_data.index, y=s_data, name="Spread", line=dict(color='#00CC96')))
    
    # Label Divergence Points on Chart
    div_points = s_data[abs(s_data) > lim]
    if not div_points.empty:
        fig.add_trace(go.Scatter(x=div_points.index, y=div_points, mode='markers', 
                                 marker=dict(color='yellow', size=10, symbol='triangle-up'), 
                                 name='Divergence Detected'))

    fig.add_hline(y=lim, line_dash="dash", line_color="red", annotation_text="+2σ")
    fig.add_hline(y=-lim, line_dash="dash", line_color="red", annotation_text="-2σ")
    fig.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig, use_container_width=True)

    # --- 5. BUY/SELL SIGNAL DIRECTORY ---
    st.subheader("📋 Active Trade Signals")
    active_signals = [p for p in sorted_pairs if p['Status'] != "STABLE"]
    
    if not active_signals:
        st.success("All monitored pairs are currently within normal range.")
    else:
        cols = st.columns(3)
        for i, signal in enumerate(active_signals[:6]):
            with cols[i % 3]:
                st.info(f"**{signal['Pair']}**")
                st.error(f"**Action:** {signal['Action']}")
                st.caption(f"Spread: {signal['Current Spread']} | Threshold: {signal['Threshold']}")

    #
