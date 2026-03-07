import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.spatial.distance import sqeuclidean
import plotly.graph_objects as go

st.set_page_config(page_title="NSE Pairs Trading Detector", layout="wide")

st.title("📊 NSE Pairs Trading: Detection & Visualization")
st.markdown("Automated identification of stock pairs using the **Gatev et al. (2006)** minimum distance method.")

# --- 1. CONFIGURATION ---
# Using 2026 post-demerger tickers
AUTO_SECTOR = ['TMPV.NS', 'TMCV.NS', 'MARUTI.NS', 'M&M.NS', 'ASHOKLEY.NS']
STEEL_SECTOR = ['TATASTEEL.NS', 'JSWSTEEL.NS', 'SAIL.NS', 'JINDALSTEL.NS']
TICKERS = list(set(AUTO_SECTOR + STEEL_SECTOR))

# --- 2. DATA ENGINE ---
@st.cache_data
def get_data(symbols):
    # Fetching 2 years for 12m formation + 6m trading
    data = yf.download(symbols, period="2y", progress=False)['Close']
    return data.ffill().dropna(axis=1)

df = get_data(TICKERS)

# --- 3. PAIR SELECTION & SIGNAL ENGINE ---
if not df.empty:
    formation_days = 252 # 12 months
    trading_days = 126   # 6 months
    
    # Split Data
    formation_df = df.iloc[-(formation_days + trading_days):-trading_days]
    trading_df = df.iloc[-trading_days:]
    
    norm_form = formation_df / formation_df.iloc[0]
    norm_trade = trading_df / trading_df.iloc[0]
    
    detected_pairs = []
    
    # Analyze all combinations
    for s1 in AUTO_SECTOR:
        for s2 in STEEL_SECTOR:
            if s1 in df.columns and s2 in df.columns:
                # Calculate Historical Distance & Std Dev
                ssd = sqeuclidean(norm_form[s1], norm_form[s2])
                spread_hist = norm_form[s1] - norm_form[s2]
                h_std = spread_hist.std()
                
                # Current Spread
                current_spread = norm_trade[s1].iloc[-1] - norm_trade[s2].iloc[-1]
                threshold = 2 * h_std
                
                # Divergence Check
                is_diverged = abs(current_spread) > threshold
                detected_pairs.append({
                    'Pair': f"{s1} vs {s2}",
                    'SSD': ssd,
                    'Current Spread': current_spread,
                    'Threshold': threshold,
                    'Status': "DIVERGED" if is_diverged else "Stable",
                    'S1': s1, 'S2': s2, 'STD': h_std
                })

    # Sort to find the "Top Pair" for visualization
    all_pairs_df = pd.DataFrame(detected_pairs).sort_values('SSD')
    top_pair_meta = all_pairs_df.iloc[0]
    
    # --- 4. VISUALIZATION WITH LABELS ---
    st.subheader(f"📈 Real-time Visualization: {top_pair_meta['Pair']}")
    
    s1, s2 = top_pair_meta['S1'], top_pair_meta['S2']
    t_spread = norm_trade[s1] - norm_trade[s2]
    limit = top_pair_meta['Threshold']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_spread.index, y=t_spread, name="Spread", line=dict(color='#00CC96', width=2)))
    
    # Boundary Lines
    fig.add_hline(y=limit, line_dash="dash", line_color="red", annotation_text="Upper 2σ (Sell S1 / Buy S2)")
    fig.add_hline(y=-limit, line_dash="dash", line_color="red", annotation_text="Lower 2σ (Buy S1 / Sell S2)")
    fig.add_hline(y=0, line_color="gray", opacity=0.5)

    # Labeling Divergence Points
    divergence_points = t_spread[abs(t_spread) > limit]
    if not divergence_points.empty:
        fig.add_trace(go.Scatter(
            x=divergence_points.index, y=divergence_points,
            mode='markers', marker=dict(color='yellow', size=10, symbol='triangle-up'),
            name='Divergence Detected'
        ))
        st.warning(f"**Signal Alert:** Divergence detected in {top_pair_meta['Pair']}. Apply 'Wait One Day' rule.")

    fig.update_layout(template="plotly_dark", height=500, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # --- 5. INDIAN MARKET PAIRS DETECTION LIST ---
    st.subheader("📋 Detected Pairs Inventory (NSE)")
    
    # Filter for active signals
    signals_only = all_pairs_df[all_pairs_df['Status'] == "DIVERGED"]
    
    if signals_only.empty:
        st.write("No active divergences detected currently. The market is in a state of relative convergence.")
    else:
        st.dataframe(signals_only[['Pair', 'Current Spread', 'Threshold', 'Status']], use_container_width=True)
