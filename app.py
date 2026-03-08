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
                    
                    if curr_spread > 2*h_std:
                        action_s1, color_s1 = "SELL", "red"
                        action_s2, color_s2 = "BUY", "green"
                    elif curr_spread < -2*h_std:
                        action_s1, color_s1 = "BUY", "green"
                        action_s2, color_s2 = "SELL", "red"
                    
                    ssd_results.append({
                        'Pair': f"{s1} vs {s2}", 'SSD': ssd, 'S1': s1, 'S2': s2, 
                        'Act1': action_s1, 'Col1': color_s1, 'Act2': action_s2, 'Col2': color_s2,
                        'Spread': curr_spread, 'Limit': 2*h_std
                    })

        top_pairs = sorted(ssd_results, key=lambda x: x['SSD'])
        selected_pair_name = st.selectbox("Select Scanned Pair", [p['Pair'] for p in top_pairs])
        p_data = next(p for p in top_pairs if p['Pair'] == selected_pair_name)
        
        # Color-Coded Signal Header
        st.markdown("### 🎯 Live Execution Signal")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write(f"**{p_data['S1']}**")
            st.markdown(f"<h1 style='color:{p_data['Col1']};'>{p_data['Act1']}</h1>", unsafe_allow_html=True)
        with c2:
            st.write(f"**{p_data['S2']}**")
            st.markdown(f"<h1 style='color:{p_data['Col2']};'>{p_data['Act2']}</h1>", unsafe_allow_html=True)
        with c3:
            price_s1 = df_ssd[p_data['S1']].iloc[-1]
            tax_unit = (min(20, 0.0003 * price_s1) + (0.00025 * price_s1) + (0.0000345 * price_s1)) * 1.18
            st.metric("SSD Distance", f"{p_data['SSD']:.5f}")
            st.metric("Est. Taxes/Unit", f"₹{tax_unit:.2f}")

        # Plotting SSD
        fig_ssd = go.Figure()
        spread_ser = norm_trade[p_data['S1']] - norm_trade[p_data['S2']]
        fig_ssd.add_trace(go.Scatter(y=spread_ser, name="Spread", line=dict(color='#00CC96')))
        fig_ssd.add_hline(y=p_data['Limit'], line_dash="dash", line_color="red", annotation_text="+2σ")
        fig_ssd.add_hline(y=-p_data['Limit'], line_dash="dash", line_color="red", annotation_text="-2σ")
        fig_ssd.update_layout(template="plotly_dark", height=400, title="Normalized Price Spread (Trading Period)")
        st.plotly_chart(fig_ssd, use_container_width=True)

# --- TAB 2: COINTEGRATION MODEL ---
with tab_coint:
    st.sidebar.header("Coint Settings")
    t1 = st.sidebar.text_input("Stock 1 (Hedge)", "HDFCBANK.NS")
    t2 = st.sidebar.text_input("Stock 2 (Target)", "ICICIBANK.NS")
    z_thresh = st.sidebar.slider("Z-Score Threshold", 1.5, 3.0, 2.0)
    
    try:
        df_c = get_clean_data([t1, t2])
        S1, S2 = df_c[t1], df_c[t2]
        
        # Stats
        _, pvalue, _ = coint(S1, S2)
        model = sm.OLS(S2, sm.add_constant(S1)).fit()
        beta = model.params[t1]
        spread = S2 - (beta * S1)
        z_score = (spread - spread.mean()) / spread.std()
        
        curr_z = z_score.iloc[-1]
        
        st.subheader(f"Statistical Arbitrage: {t1} vs {t2}")
        c_a, c_b = st.columns(2)
        
        if curr_z < -z_thresh:
            c_a.markdown(f"**{t2}**: <span style='color:green; font-size:30px;'>BUY</span> (100 units)", unsafe_allow_html=True)
            c_b.markdown(f"**{t1}**: <span style='color:red; font-size:30px;'>SELL</span> ({round(100*beta)} units)", unsafe_allow_html=True)
        elif curr_z > z_thresh:
            c_a.markdown(f"**{t2}**: <span style='color:red; font-size:30px;'>SELL</span> (100 units)", unsafe_allow_html=True)
            c_b.markdown(f"**{t1}**: <span style='color:green; font-size:30px;'>BUY</span> ({round(100*beta)} units)", unsafe_allow_html=True)
        else:
            st.info("⌛ **SIGNAL: NEUTRAL** - Spread is within historical bounds.")
            
        fig_z = go.Figure()
        fig_z.add_trace(go.Scatter(y=z_score, name="Z-Score", line=dict(color='orange')))
        fig_z.add_hline(y=z_thresh, line_dash="dot", line_color="red")
        fig_z.add_hline(y=-z_thresh, line_dash="dot", line_color="green")
        fig_z.update_layout(template="plotly_dark", height=400, title="Z-Score Spread Evolution")
        st.plotly_chart(fig_z, use_container_width=True)
        
    except Exception:
        st.warning("Please enter valid NSE tickers with .NS suffix.")

# --- TAB 3: INSTRUCTIONS ---
with tab_instr:
    st.header("📖 Instructions for Indian Market Execution")
    
    col_x, col_y = st.columns(2)
    with col_x:
        st.info("### 1. SSD Model (Discovery)")
        st.write("- **Best for:** Intersectoral trades (e.g., Auto vs Steel).")
        st.write("- **Goal:** Find stocks that are fundamentally linked but price-diverged.")
        st.write("- **Trade:** Sell the 'winner' and Buy the 'loser' simultaneously.")
        
    with col_y:
        st.success("### 2. Cointegration Model (Precision)")
        st.write("- **Best for:** Intrasectoral trades (e.g., HDFC vs ICICI).")
        st.write("- **Goal:** Profit from mean-reversion in the statistical spread.")
        st.write("- **Trade:** Use the Hedge Ratio (Beta) to balance your quantities.")
    
    st.divider()
    st.warning("**The Wait One Day Rule:** In the Indian market, slippage and bid-ask noise can trigger false signals. Always wait 24 hours after a signal appears. If the divergence persists the next morning, execute your trade.")
