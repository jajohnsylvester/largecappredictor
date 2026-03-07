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
# TMPV = Passenger Vehicles, TMCV = Commercial Vehicles
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
    
    def normalize_df(data_frame):
        return data_frame / data_frame.iloc[0]

    norm_form = normalize_df(formation_df)
    norm_trade = normalize_df(trading_df)
    
    all_pairs_results = []
    
    for s1 in AUTO_SECTOR:
        for s2 in STEEL_SECTOR:
            if s1 in norm_form.columns and s2 in norm_form.columns:
                ssd = sqeuclidean(norm_form[s1], norm_form[s2])
                hist_spread = norm_form[s1] - norm_form[s2]
                h_std = hist_spread.std()
                threshold = 2 * h_std
                t_series = norm_trade[s1] - norm_trade[s2]
                current_val = t_series.iloc[-1]
                
                # Signal Logic with Color Coding
                if current_val > threshold:
                    # Stock 1 is high (Red/Sell), Stock 2 is low (Green/Buy)
                    action_s1, color_s1 = "SELL", "red"
                    action_s2, color_s2 = "BUY", "green"
                    status = "🚨 DIVERGED (High)"
                elif current_val < -threshold:
                    # Stock 1 is low (Green/Buy), Stock 2 is high (Red/Sell)
                    action_s1, color_s1 = "BUY", "green"
                    action_s2, color_s2 = "SELL", "red"
                    status = "🚨 DIVERGED (Low)"
                elif abs(current_val) < (0.1 * h_std):
                    action_s1, action_s2, color_s1, color_s2 = "EXIT", "EXIT", "#00b4d8", "#00b4d8"
                    status = "✅ CONVERGED"
                else:
                    action_s1, action_s2, color_s1, color_s2 = "HOLD", "HOLD", "gray", "gray"
                    status = "STABLE"

                all_pairs_results.append({
                    'PairName': f"{s1} vs {s2}",
                    'SSD': ssd,
                    'Spread': round(current_val, 4),
                    'Threshold': round(threshold, 4),
                    'Status': status,
                    'S1': s1, 'S2': s2,
                    'Action_S1': action_s1, 'Color_S1': color_s1,
                    'Action_S2': action_s2, 'Color_S2': color_s2,
                    'limit': threshold, 'series': t_series
                })

    # --- 4. SIDEBAR SELECTION & HOW TO USE ---
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "How to Use (NSE)"])

    if page == "Dashboard":
        if all_pairs_results:
            sorted_pairs = sorted(all_pairs_results, key=lambda x: x['SSD'])
            pair_names = [p['PairName'] for p in sorted_pairs]
            st.sidebar.header("Pair Selection")
            selected_name = st.sidebar.selectbox("Select Pair", options=pair_names)
            selected_pair = next(p for p in sorted_pairs if p['PairName'] == selected_name)

            # --- 5. VISUALIZATION ---
            st.subheader(f"📈 Chart: {selected_pair['PairName']}")
            fig = go.Figure()
            s_data = selected_pair['series']
            lim = selected_pair['limit']
            fig.add_trace(go.Scatter(x=s_data.index, y=s_data, name="Spread", line=dict(color='#00CC96')))
            
            # Divergence Points
            div_points = s_data[abs(s_data) > lim]
            if not div_points.empty:
                fig.add_trace(go.Scatter(x=div_points.index, y=div_points, mode='markers', 
                                         marker=dict(color='yellow', size=10, symbol='triangle-up'), name='Divergence'))

            fig.add_hline(y=lim, line_dash="dash", line_color="red", annotation_text="+2σ")
            fig.add_hline(y=-lim, line_dash="dash", line_color="red", annotation_text="-2σ")
            fig.update_layout(template="plotly_dark", height=450)
            st.plotly_chart(fig, use_container_width=True)

            # --- 6. COLOR-CODED ACTION PANEL ---
            st.subheader("🎯 Trading Instructions")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"<p style='text-align: center;'>{selected_pair['S1']}</p>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='text-align: center; color:{selected_pair['Color_S1']};'>{selected_pair['Action_S1']}</h2>", unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"<p style='text-align: center;'>{selected_pair['S2']}</p>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='text-align: center; color:{selected_pair['Color_S2']};'>{selected_pair['Action_S2']}</h2>", unsafe_allow_html=True)
                
            st.write(f"**Current Status:** {selected_pair['Status']}")
            st.write(f"**Current Spread:** {selected_pair['Spread']} (Threshold: ±{selected_pair['Threshold']})")

        else:
            st.error("No valid pairs found.")
    
    else:
        # --- HOW TO USE PAGE ---
        st.subheader("📖 How to use this for the Indian Market")
        st.write("""
        This strategy leverages the economic link between the **Automobile industry** (Main) and the **Steel industry** (Related). 
        Since steel is a primary raw material for vehicles, these stocks should move in tandem.
        """)
        
        st.info("**Step 1: Check the Pair Signal**")
        st.write("Look for pairs with a **DIVERGED** status. This means the historical price relationship has temporarily broken.")
        
        st.error("**Step 2: Apply the 'Wait One Day' Rule**")
        st.write("When the yellow triangle marker appears on the chart, **do not trade immediately**. Wait for the next market day. If the divergence persists, the signal is valid.")
        
        st.success("**Step 3: Execution (Market Neutral)**")
        st.write("""
        Execute both the **BUY** and **SELL** orders simultaneously with equal capital (e.g., ₹50,000 each). 
        This ensures you are protected from broad market crashes.
        """)
        
        st.warning("**Step 4: The Exit**")
        st.write("Exit both positions when the chart shows **CONVERGED** (the spread crosses the zero line).")

else:
    st.warning("Fetching market data... please wait.")
