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
    # auto_adjust=True handles 'Close' as the adjusted price in newer yfinance versions
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
    
    # Normalization Function
    def normalize_df(data_frame):
        return data_frame / data_frame.iloc[0]

    norm_form = normalize_df(formation_df)
    norm_trade = normalize_df(trading_df)
    
    all_pairs_results = []
    
    # Process all combinations between sectors
    for s1 in AUTO_SECTOR:
        for s2 in STEEL_SECTOR:
            if s1 in norm_form.columns and s2 in norm_form.columns:
                # Minimum Distance (SSD)
                ssd = sqeuclidean(norm_form[s1], norm_form[s2])
                
                # Historical Spread Statistics
                hist_spread = norm_form[s1] - norm_form[s2]
                h_std = hist_spread.std()
                threshold = 2 * h_std
                
                # Current Trading Spread
                t_series = norm_trade[s1] - norm_trade[s2]
                current_val = t_series.iloc[-1]
                
                # Determine Buy/Sell Action
                if current_val > threshold:
                    action = f"SELL {s1} / BUY {s2}"
                    status = "🚨 DIVERGED (High)"
                elif current_val < -threshold:
                    action = f"BUY {s1} / SELL {s2}"
                    status = "🚨 DIVERGED (Low)"
                elif abs(current_val) < (0.1 * h_std):
                    action = "✅ EXIT (Convergence)"
                    status = "CONVERGED"
                else:
                    action = "Neutral"
                    status = "STABLE"

                # Dictionary append fixed with proper closing
                all_pairs_results.append({
                    'PairName': f"{s1} vs {s2}",
                    'SSD': ssd,
                    'Spread': round(current_val, 4),
                    'Threshold': round(threshold, 4),
                    'Action': action,
                    'Status': status,
                    'S1': s1, 'S2': s2, 'limit': threshold,
                    'series': t_series
                })

    # --- 4. SIDEBAR SELECTION ---
    if all_pairs_results:
        sorted_pairs = sorted(all_pairs_results, key=lambda x: x['SSD'])
        pair_names = [p['PairName'] for p in sorted_pairs]
        
        st.sidebar.header("Pair Selection")
        selected_name = st.sidebar.selectbox("Select a Pair to Analyze", options=pair_names)
        
        # Retrieve data for the selected pair
        selected_pair = next(p for p in sorted_pairs if p['PairName'] == selected_name)

        # --- 5. VISUALIZATION ---
        st.subheader(f"📈 Chart: {selected_pair['PairName']}")
        
        fig = go.Figure()
        s_data = selected_pair['series']
        lim = selected_pair['limit']
        
        fig.add_trace(go.Scatter(x=s_data.index, y=s_data, name="Spread", line=dict(color='#00CC96')))
        
        # Highlight Divergence Points
        div_points = s_data[abs(s_data) > lim]
        if not div_points.empty:
            fig.add_trace(go.Scatter(x=div_points.index, y=div_points, mode='markers', 
                                     marker=dict(color='yellow', size=10, symbol='triangle-up'), 
                                     name='Divergence'))

        fig.add_hline(y=lim, line_dash="dash", line_color="red", annotation_text="+2σ Boundary")
        fig.add_hline(y=-lim, line_dash="dash", line_color="red", annotation_text="-2σ Boundary")
        fig.add_hline(y=0, line_color="white", opacity=0.3)
        
        fig.update_layout(template="plotly_dark", height=500, xaxis_title="Trading Days", yaxis_title="Normalized Price Difference")
        st.plotly_chart(fig, use_container_width=True)

        # --- 6. SIGNAL DISPLAY & SUMMARY ---
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"### Current Status: {selected_pair['Status']}")
            st.info(f"**Action:** {selected_pair['Action']}")
        
        with c2:
            st.write(f"**Historical Distance (SSD):** {selected_pair['SSD']:.6f}")
            st.write(f"**Current Spread:** {selected_pair['Spread']}")
            st.write(f"**Target Threshold:** ±{selected_pair['Threshold']}")

        # Table of Active Signals
        st.subheader("📋 Active Signals Across NSE")
        active_df = pd.DataFrame([p for p in sorted_pairs if p['Status'] != "STABLE"])
        if not active_df.empty:
            st.dataframe(active_df[['PairName', 'Action', 'Spread', 'Status']], hide_index=True, use_container_width=True)
        else:
            st.success("No active divergences detected in the monitored sectors.")
    else:
        st.error("No valid pairs could be formed with the available data.")

else:
    st.warning("Fetching market data... please wait.")
