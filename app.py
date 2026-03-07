import pandas as pd
import numpy as np
import yfinance as yf
from scipy.spatial.distance import sqeuclidean
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION & UPDATED TICKERS ---
# We have replaced TATAMOTORS with the new post-demerger tickers: TMPV and TMCV
main_industry = ['TMPV.NS', 'TMCV.NS', 'MARUTI.NS', 'M&M.NS'] # Automobile
related_industry = ['TATASTEEL.NS', 'JSWSTEEL.NS', 'SAIL.NS']  # Steel
tickers = list(set(main_industry + related_industry))

# --- 2. DATA FETCHING ---
print("Fetching latest Indian market data...")
# Fetching 2 years of data to cover 12m formation + 6m trading + buffer
data = yf.download(tickers, period="2y")['Adj Close']

# Data Cleaning: forward fill minor gaps and drop stocks with no data
data = data.ffill().dropna(axis=1)

if data.empty:
    print("Error: No data found. Please check your internet connection or tickers.")
else:
    # Split into Formation (12 months) and Trading (latest 6 months)
    formation_df = data.iloc[-504:-126] 
    trading_df = data.iloc[-126:]

    # Normalization Function (Paper Rule: Start at 1.0)
    def normalize(df): return df / df.iloc[0]
    
    norm_form = normalize(formation_df)
    
    # --- 3. PAIRING LOGIC (MINIMUM DISTANCE) ---
    pairs_ssd = []
    # Ensure we only iterate over tickers that successfully downloaded
    valid_main = [s for s in main_industry if s in norm_form.columns]
    valid_related = [s for s in related_industry if s in norm_form.columns]

    for s1 in valid_main:
        for s2 in valid_related:
            ssd = sqeuclidean(norm_form[s1], norm_form[s2])
            pairs_ssd.append((s1, s2, ssd))

    # Error Check: Prevent IndexError if no pairs are found
    if not pairs_ssd:
        print("Error: Could not form any pairs. Check if tickers are active.")
    else:
        # Sort by SSD and pick the best pair
        s1, s2, min_ssd = sorted(pairs_ssd, key=lambda x: x[2])[0]
        
        # --- 4. SIGNAL CALCULATION ---
        # Historical Std Dev of the spread during formation
        hist_spread = normalize(formation_df[s1]) - normalize(formation_df[s2])
        hist_std = hist_spread.std()
        
        # Current spread in the trading period
        norm_trade = normalize(trading_df[[s1, s2]])
        current_spread = norm_trade[s1].iloc[-1] - norm_trade[s2].iloc[-1]
        prev_spread = norm_trade[s1].iloc[-2] - norm_trade[s2].iloc[-2]
        threshold = 2 * hist_std

        # --- 5. RULE-BASED SUMMARY ---
        status = "NEUTRAL"
        instruction = "Monitor; spread is within the 2-Sigma range."

        if abs(current_spread) > threshold:
            status = "DIVERGENCE DETECTED"
            instruction = "WAIT ONE DAY: Confirm the gap persists to avoid bid-ask bounce."
        elif (prev_spread > 0 and current_spread <= 0) or (prev_spread < 0 and current_spread >= 0):
            status = "CONVERGENCE REACHED"
            instruction = "EXIT POSITION: Prices have crossed. Realize profits now."
        elif abs(current_spread) > (threshold * 2):
            status = "STOP LOSS"
            instruction = "EXIT IMMEDIATELY: Spread has exceeded 4-Sigma; fundamental link broken."

        print(f"\n{'='*50}")
        print(f"LATEST TOP PAIR: {s1} & {s2}")
        print(f"SSD (Distance): {min_ssd:.6f}")
        print(f"Current Spread: {current_spread:.4f} (Limit: {threshold:.4f})")
        print(f"STATUS: {status}")
        print(f"ACTION: {instruction}")
        print(f"{'='*50}")
