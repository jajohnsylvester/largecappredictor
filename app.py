import pandas as pd
import numpy as np
import yfinance as yf
from scipy.spatial.distance import sqeuclidean
import matplotlib.pyplot as plt

# --- CONFIGURATION & UNIVERSE ---
# Top Nifty 100 stocks for high liquidity (Representative list)
tickers = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'BHARTIARTL.NS',
    'SBIN.NS', 'LICI.NS', 'ITC.NS', 'HINDUNILVR.NS', 'LT.NS', 'BAJFINANCE.NS',
    'MARUTI.NS', 'SUNPHARMA.NS', 'ADANIENT.NS', 'TATAMOTORS.NS', 'AXISBANK.NS',
    'ONGC.NS', 'NTPC.NS', 'KOTAKBANK.NS', 'TATASTEEL.NS', 'M&M.NS', 'JSWSTEEL.NS'
]

# 1. FETCH DATA (12M Formation + 6M Trading)
print("Fetching Nifty 100 data...")
df = yf.download(tickers, period="2y")['Adj Close'].dropna(axis=1)
formation_df = df.iloc[-504:-126]  # 12-month formation period
trading_df = df.iloc[-126:]        # 6-month trading period

# 2. NORMALIZATION & PAIRING
def normalize(data): return data / data.iloc[0]

norm_form = normalize(formation_df)
pairs_ssd = []

# Calculate SSD for all possible combinations
cols = norm_form.columns
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        s1, s2 = cols[i], cols[j]
        ssd = sqeuclidean(norm_form[s1], norm_form[s2])
        pairs_ssd.append((s1, s2, ssd))

# Pick the pair with the smallest historical distance
s1, s2, min_ssd = sorted(pairs_ssd, key=lambda x: x[2])[0]
print(f"\nTop Pair Identified: {s1} & {s2} (SSD: {min_ssd:.4f})")

# 3. TRADING LOGIC
# Calculate historical standard deviation of the spread
hist_spread = normalize(formation_df[s1]) - normalize(formation_df[s2])
hist_std = hist_spread.std()

# Trading period data
norm_trade = normalize(trading_df[[s1, s2]])
spread = norm_trade[s1] - norm_trade[s2]
threshold = 2 * hist_std

# SIGNAL SUMMARY LOGIC
last_spread = spread.iloc[-1]
prev_spread = spread.iloc[-2]
status = "NEUTRAL"
instruction = "Maintain watchlist; spread within normal range."

# RULE 1: Wait One Day Rule (Divergence > 2 Sigma)
if abs(last_spread) > threshold:
    status = "DIVERGENCE DETECTED"
    instruction = "WAIT ONE DAY: Do not enter today. Confirm the gap persists to avoid bid-ask bounce."

# RULE 2: Convergence Exit (Zero Crossing)
elif (prev_spread > 0 and last_spread <= 0) or (prev_spread < 0 and last_spread >= 0):
    status = "CONVERGENCE REACHED"
    instruction = "EXIT NOW: Normalized prices have crossed. Close both positions for profit."

# RULE 3: Stop Loss (6-Month Time Limit or 4-Sigma Distance)
elif abs(last_spread) > (threshold * 2):
    status = "STOP LOSS TRIGGERED"
    instruction = "EXIT IMMEDIATELY: Structural break detected (4-Sigma divergence)."

# --- OUTPUT ---
print(f"{'='*50}")
print(f"PAIR: {s1} vs {s2}")
print(f"CURRENT SPREAD: {last_spread:.4f} (Limit: {threshold:.4f})")
print(f"STATUS: {status}")
print(f"ACTION: {instruction}")
print(f"{'='*50}")
