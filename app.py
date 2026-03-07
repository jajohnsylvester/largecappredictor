import pandas as pd
import numpy as np
import yfinance as yf
from scipy.spatial.distance import sqeuclidean

# 1. Configuration with 2026 Tickers
# Replacing TATAMOTORS.NS with post-demerger entities TMPV and TMCV
main_industry = ['TMPV.NS', 'TMCV.NS', 'MARUTI.NS', 'M&M.NS']
related_industry = ['TATASTEEL.NS', 'JSWSTEEL.NS', 'SAIL.NS']
tickers = list(set(main_industry + related_industry))

# 2. Fetch Data (Fix for KeyError and MultiIndex)
print("Fetching March 2026 data from NSE...")
# auto_adjust=True (default) makes 'Close' the adjusted price and removes 'Adj Close'
raw_data = yf.download(tickers, period="2y", progress=False)

# Select 'Close' column (which is now adjusted)
# yfinance multi-ticker download results in columns: (PriceType, Ticker)
if 'Close' in raw_data.columns.get_level_values(0):
    data = raw_data['Close'].ffill().dropna(axis=1)
else:
    # Fallback for single ticker or different structure
    data = raw_data.ffill().dropna(axis=1)

if data.empty:
    print("Error: No data found. Verify your internet and ticker symbols.")
else:
    # 3. Formation and Trading Splits
    formation_df = data.iloc[-504:-126] 
    trading_df = data.iloc[-126:]

    def normalize(df): return df / df.iloc[0]
    norm_form = normalize(formation_df)
    
    # 4. Find Best Intersectoral Pair
    pairs_ssd = []
    valid_main = [s for s in main_industry if s in norm_form.columns]
    valid_related = [s for s in related_industry if s in norm_form.columns]

    for s1 in valid_main:
        for s2 in valid_related:
            ssd = sqeuclidean(norm_form[s1], norm_form[s2])
            pairs_ssd.append((s1, s2, ssd))

    if not pairs_ssd:
        print("Error: No valid pairs formed. Ensure tickers are correct.")
    else:
        # Sort and Pick Best
        s1, s2, min_ssd = sorted(pairs_ssd, key=lambda x: x[2])[0]
        
        # 5. Signal Calculation
        hist_spread = normalize(formation_df[s1]) - normalize(formation_df[s2])
        hist_std = hist_spread.std()
        
        norm_trade = normalize(trading_df[[s1, s2]])
        current_spread = norm_trade[s1].iloc[-1] - norm_trade[s2].iloc[-1]
        prev_spread = norm_trade[s1].iloc[-2] - norm_trade[s2].iloc[-2]
        threshold = 2 * hist_std

        # 6. Rule-Based Summary
        status = "NEUTRAL"
        instruction = "Spread within 2-Sigma range. Maintain monitor."

        if abs(current_spread) > threshold:
            status = "DIVERGENCE DETECTED"
            instruction = "WAIT ONE DAY: Confirm the gap persists to avoid bid-ask bounce."
        elif (prev_spread > 0 and current_spread <= 0) or (prev_spread < 0 and current_spread >= 0):
            status = "CONVERGENCE"
            instruction = "EXIT POSITION: Prices have crossed. Realize profit/loss now."
        elif abs(current_spread) > (threshold * 2):
            status = "STOP LOSS"
            instruction = "EXIT IMMEDIATELY: Spread exceeded 4-Sigma. Link broken."

        print(f"\n{'='*50}")
        print(f"TOP PAIR: {s1} & {s2}")
        print(f"STATUS: {status}")
        print(f"ACTION: {instruction}")
        print(f"Current Spread: {current_spread:.4f} (Threshold: {threshold:.4f})")
        print(f"{'='*50}")
