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
    data = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
    return data.ffill().dropna(axis=1)

# --- UI TABS ---
tab_ssd, tab_coint, tab_instr = st.tabs(["🔍 SSD Discovery (Industry)", "📊 Cointegration (Statistical)", "📖 How to Use"])

# --- TAB 1: SSD DISTANCE MODEL ---
with tab_ssd:
    st.subheader("Industry Discovery: Auto vs Steel")
    df_ssd = get_clean_data(ALL_TICKERS)
    
    if not df_ssd.empty:
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
                    
                    # Logic
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
            st.markdown(f"<h1 style='color:{p_data['Col1']};'>{p_data['Act1']}</h1>", unsafe_allow_html=True)
        with c2:
            st.write(f"**{p_data['S2']}**")
            st.markdown(f"<h1 style='color:{p_data['Col2']};'>{p_data['Act2']}</h1>", unsafe_allow_html=True)
        with c3:
            st.metric("Status", p_data['Stat'])
            st.metric("SSD Distance", f"{p_data['SSD']:.5f}")

        fig_ssd = go.Figure()
        spread_ser = norm_trade[p_data['S1']] - norm_trade[p_data['S2']]
        fig_ssd.add_trace(go.Scatter(y=spread_ser, name="Spread", line=dict(color='#00CC96')))
        fig_ssd.add_hline(y=p_data['Limit'], line_dash="dash", line_color="red")
        fig_ssd.add_hline(y=-p_data['Limit'], line_dash="dash", line_color="red")
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
        
        # Stats
        _, pvalue, _ = coint(S1, S2)
        model = sm.OLS(S2, sm.add_constant(S1)).fit()
        beta = model.params[t1]
        spread = S2 - (beta * S1)
        z_score = (spread - spread.mean()) / spread.std()
        curr_z = z_score.iloc[-1]
        
        # Cointegration Signal Logic
        c_act1, c_act2, c_col1, c_col2, c_stat = "HOLD", "HOLD", "gray", "gray", "NEUTRAL"
        if curr_z > z_thresh: c_act1, c_act2, c_col1, c_col2, c_stat = "BUY", "SELL", "green", "red", "🚨 OVERBOUGHT"
        elif curr_z < -z_thresh: c_act1, c_act2, c_col1, c_col2, c_stat = "SELL", "BUY", "red", "green", "🚨 OVERSOLD"
        elif abs(curr_z) < 0.5: c_act1, c_act2, c_col1, c_col2, c_stat = "EXIT", "EXIT", "#00b4d8", "#00b4d8", "✅ CONVERGED"

        st.subheader(f"Statistical Arbitrage: {t1} vs {t2}")
        st.markdown("### 🎯 Live Coint Execution Signal")
        cx1, cx2, cx3 = st.columns(3)
        with cx1:
            st.write(f"**{t1} (Hedge)**")
            st.markdown(f"<h1 style='color:{c_col1};'>{c_act1}</h1>", unsafe_allow_html=True)
            st.caption(f"Qty: {round(100*beta)} (per 100 of {t2})")
        with cx2:
            st.write(f"**{t2} (Target)**")
            st.markdown(f"<h1 style='color:{c_col2};'>{c_act2}</h1>", unsafe_allow_html=True)
            st.caption("Qty: 100")
        with cx3:
            st.metric("Status", c_stat)
            st.metric("P-Value", f"{pvalue:.4f}")

        fig_z = go.Figure()
        fig_z.add_trace(go.Scatter(y=z_score, name="Z-Score", line=dict(color='orange')))
        fig_z.add_hline(y=z_thresh, line_dash="dot", line_color="red")
        fig_z.add_hline(y=-z_thresh, line_dash="dot", line_color="green")
        fig_z.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_z, use_container_width=True)
        
    except Exception:
        st.warning("Enter valid tickers (e.g. RELIANCE.NS)")

# --- TAB 3: INSTRUCTIONS ---
with tab_instr:
    st.header("📖 Operational Guide for Indian Markets")
    st.info("### How to handle HOLD / NEUTRAL / EXIT")
    st.write("""
    1.  **HOLD (Gray):** The spread is currently in 'No Man's Land.' No new trades should be initiated. If you have an open position, continue to hold until the EXIT signal appears.
    2.  **NEUTRAL (Info):** The stocks are moving perfectly in sync. There is no arbitrage opportunity. Keep the pair on your watchlist.
    3.  **EXIT (Blue):** The spread has returned to its historical mean. Close **both** the Buy and Sell legs immediately to lock in your profit.
    4.  **Wait One Day Rule:** In India, market opening gaps are common. If a signal appears at 3:20 PM, wait for the next day's opening. If the signal remains valid after 10:00 AM, execute the trade.
    """)
