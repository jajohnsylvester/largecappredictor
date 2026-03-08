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

# --- TICKER CONFIG (MARCH 2026 READY) ---
# Tickers reflect the Oct 2025 demerger: TMPV (Passenger) and TMCV (Commercial)
AUTO_SECTOR = ['TMPV.NS', 'TMCV.NS', 'MARUTI.NS', 'M&M.NS', 'ASHOKLEY.NS']
STEEL_SECTOR = ['TATASTEEL.NS', 'JSWSTEEL.NS', 'SAIL.NS', 'JINDALSTEL.NS']
ALL_TICKERS = list(set(AUTO_SECTOR + STEEL_SECTOR))

@st.cache_data
def get_clean_data(symbols, period="2y"):
    # yfinance 0.2.50+ returns 'Close' as adjusted price; auto_adjust=True is default
    raw = yf.download(symbols, period=period, progress=False)
    data = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
    return data.ffill().dropna(axis=1)

# --- UI TABS ---
tab_ssd, tab_coint, tab_instr = st.tabs(["🔍 SSD Discovery (Industry)", "📊 Cointegration (Statistical)", "📖 How to Use"])

# --- TAB 1: SSD DISTANCE MODEL ---
with tab_ssd:
    st.subheader("Industry Discovery: Auto vs Steel")
    df_ssd = get_clean_data(ALL_TICKERS)
    
    if not df_ssd.empty:
        form_df = df_ssd.iloc[-504:-126] # 12m Formation
        trade_df = df_ssd.iloc[-126:]    # 6m Trading
        
        norm_form = form_df / form_df.iloc[0]
        norm_trade = trade_df / trade_df.iloc[0]
        
        ssd_results = []
        for s1 in AUTO_SECTOR:
            for s2 in STEEL_SECTOR:
                if s1 in df_ssd.columns and s2 in df_ssd.columns:
                    ssd = sqeuclidean(norm_form[s1], norm_form[s2])
                    h_std = (norm_form[s1] - norm_form[s2]).std()
                    curr_spread = norm_trade[s1].iloc[-1] - norm_trade[s2].iloc[-1]
                    
                    act1, act2, col1, col2, stat = "HOLD", "HOLD", "gray", "gray", "STABLE"
                    if curr_spread > 2*h_std: act1, act2, col1, col2, stat = "SELL", "BUY", "red", "green", "🚨 DIVERGED"
                    elif curr_spread < -2*h_std: act1, act2, col1, col2, stat = "BUY", "SELL", "green", "red", "🚨 DIVERGED"
                    
                    ssd_results.append({
                        'Pair': f"{s1} vs {s2}", 'SSD': ssd, 'S1': s1, 'S2': s2, 
                        'Act1': act1, 'Col1': col1, 'Act2': act2, 'Col2': col2, 'Stat': stat,
                        'Spread': curr_spread, 'Limit': 2*h_std
                    })

        top_pairs = sorted(ssd_results, key=lambda x: x['SSD'])
        selected_pair_name = st.selectbox("Select Scanned Pair", [p['Pair'] for p in top_pairs])
        p_data = next(p for p in top_pairs if p['Pair'] == selected_pair_name)
        
        st.markdown("### 🎯 Live SSD Execution Signal")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write(f"**{p_data['S1']}**")
            st.markdown(f"<h1 style='color:{p_data['Col1']}; text-align:center;'>{p_data['Act1']}</h1>", unsafe_allow_html=True)
        with c2:
            st.write(f"**{p_data['S2']}**")
            st.markdown(f"<h1 style='color:{p_data['Col2']}; text-align:center;'>{p_data['Act2']}</h1>", unsafe_allow_html=True)
        with c3:
            # 2026 NSE Tax Logic
            price_ref = df_ssd[p_data['S1']].iloc[-1]
            stt_delivery = (price_ref * 0.001) * 2 
            stt_intraday = (price_ref * 0.00025)
            st.metric("Status", p_data['Stat'])
            st.metric("Est. STT (Delivery)", f"₹{stt_delivery:.2f}")

        fig_ssd = go.Figure()
        spread_ser = norm_trade[p_data['S1']] - norm_trade[p_data['S2']]
        fig_ssd.add_trace(go.Scatter(y=spread_ser, name="Spread", line=dict(color='#00CC96')))
        fig_ssd.add_hline(y=p_data['Limit'], line_dash="dash", line_color="red", annotation_text="+2σ")
        fig_ssd.add_hline(y=-p_data['Limit'], line_dash="dash", line_color="red", annotation_text="-2σ")
        fig_ssd.update_layout(template="plotly_dark", height=400)
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
        _, pvalue, _ = coint(S1, S2)
        model = sm.OLS(S2, sm.add_constant(S1)).fit()
        beta = model.params[t1]
        spread = S2 - (beta * S1)
        z_score = (spread - spread.mean()) / spread.std()
        curr_z = z_score.iloc[-1]
        
        c_act1, c_act2, c_col1, c_col2, c_stat = "HOLD", "HOLD", "gray", "gray", "NEUTRAL"
        if curr_z > z_thresh: c_act1, c_act2, c_col1, c_col2, c_stat = "BUY", "SELL", "green", "red", "🚨 OVERBOUGHT"
        elif curr_z < -z_thresh: c_act1, c_act2, c_col1, c_col2, c_stat = "SELL", "BUY", "red", "green", "🚨 OVERSOLD"
        elif abs(curr_z) < 0.5: c_act1, c_act2, c_col1, c_col2, c_stat = "EXIT", "EXIT", "#00b4d8", "#00b4d8", "✅ CONVERGED"

        st.subheader(f"Statistical Arbitrage: {t1} vs {t2}")
        st.markdown("### 🎯 Live Coint Execution Signal")
        cx1, cx2, cx3 = st.columns(3)
        with cx1:
            st.write(f"**{t1} (Hedge)**")
            st.markdown(f"<h1 style='color:{c_col1}; text-align:center;'>{c_act1}</h1>", unsafe_allow_html=True)
            st.caption(f"Qty: {round(100*beta)} (per 100 of {t2})")
        with cx2:
            st.write(f"**{t2} (Target)**")
            st.markdown(f"<h1 style='color:{c_col2}; text-align:center;'>{c_act2}</h1>", unsafe_allow_html=True)
            st.caption("Qty: 100")
        with cx3:
            st.metric("P-Value", f"{pvalue:.4f}")
            st.metric("Z-Score", f"{curr_z:.2f}")

        fig_z = go.Figure()
        fig_z.add_trace(go.Scatter(y=z_score, name="Z-Score", line=dict(color='orange')))
        fig_z.add_hline(y=z_thresh, line_dash="dot", line_color="red")
        fig_z.add_hline(y=-z_thresh, line_dash="dot", line_color="green")
        fig_z.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_z, use_container_width=True)
    except Exception:
        st.warning("Ensure tickers use .NS suffix.")

# --- TAB 3: MERGED COMPREHENSIVE INSTRUCTIONS ---
with tab_instr:
    st.header("📖 Professional Guide to NSE Pairs Trading")
    
    st.info("### 1. Strategy Selection (SSD vs Cointegration)")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**SSD Discovery Model**")
        st.write("- **Best for:** Cross-industry pairs (e.g., Auto vs Steel).")
        st.write("- **Logic:** Finds stocks with strong fundamental links (Supply Chain). Uses SSD (Sum of Squared Deviations).")
    with col_b:
        st.markdown("**Cointegration Model**")
        st.write("- **Best for:** Identical stocks/competitors (e.g., HDFC vs ICICI).")
        st.write("- **Logic:** Uses statistical regression (Z-Score) to determine mean-reversion with a specific hedge ratio.")

    st.error("### 2. Merged Execution Guide (Critical)")
    st.markdown("""
    1. **Wait One Day Rule:** In the Indian market, overnight gaps can trigger false signals. When a signal (Yellow marker/status change) appears, **wait 24 hours**. Execute only if the divergence persists the next morning.
    2. **Self-Financing (Market Neutral):** Always buy and sell equal rupee values (SSD) or Beta-adjusted quantities (Coint). This protects you from broad market crashes.
    3. **The Tata Demerger:** `TATAMOTORS` is retired. Use **TMPV** (Passenger/EV/JLR) and **TMCV** (Commercial) to avoid data gaps.
    4. **2026 Taxation:** Delivery STT is **0.1%** on both sides. Intraday is **0.025%** on the sell side.
    """)

    st.success("### 3. Handling Signal States")
    col_1, col_2, col_3 = st.columns(3)
    with col_1:
        st.markdown("<span style='color:gray; font-weight:bold;'>HOLD / NEUTRAL</span>", unsafe_allow_html=True)
        st.write("Wait on the sidelines. The risk-to-reward ratio is not favorable for entry.")
    with col_2:
        st.markdown("<span style='color:green; font-weight:bold;'>BUY</span> / <span style='color:red; font-weight:bold;'>SELL</span>", unsafe_allow_html=True)
        st.write("Arbitrage opportunity active. Enter both legs simultaneously.")
    with col_3:
        st.markdown("<span style='color:#00b4d8; font-weight:bold;'>EXIT / CONVERGED</span>", unsafe_allow_html=True)
        st.write("Square off both positions immediately. Parity has been restored.")
