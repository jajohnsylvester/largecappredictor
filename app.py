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
This app scans the Indian market for pairs in the **Auto** and **Steel** industries that show significant price divergence.
""")

# --- 1. CONFIGURATION & 2026 TICKERS ---
# Updated for the 2026 Tata Motors demerger
AUTO_SECTOR = ['TMPV.NS', 'TMCV.NS', 'MARUTI.NS', 'M&M.NS', 'ASHOKLEY.NS']
STEEL_SECTOR = ['TATASTEEL.NS', 'JSWSTEEL.NS', 'SAIL.NS', 'JINDALSTEL.NS
