import pandas as pd
import numpy as np
import yfinance as yf
from scipy.spatial.distance import sqeuclidean

# 1. Define Universe (Ensure these tickers are active on NSE)
main_industry = ['TATAMOTORS.NS', 'MARUTI.NS', 'M&M.NS']
related_industry = ['TATASTEEL.NS', 'JSWSTEEL.NS', 'SAIL.NS']
tickers = list(set(main_industry + related_industry))

# 2. Fetch Data with Error Handling
print("Fetching data...")
data = yf.download(tickers, period="2y")['Adj Close']

# Fix for IndexError: fill minor gaps instead of dropping everything
data = data.ffill().dropna(axis=1) 

if data.empty:
    print("Error: No data found for the specified tickers.")
else:
    formation_df = data.iloc[-504:-126] 
    trading_df = data.iloc[-126:]

    def normalize(df): return df / df.iloc[0]
    norm_form = normalize(formation_df)
    
    pairs_ssd = []
    # Ensure we only iterate over columns that actually exist in the fetched data
    valid_main = [s for s in main_industry if s in norm_form.columns]
    valid_related = [s for s in related_industry if s in norm_form.columns]

    for s1 in valid_main:
        for s2 in valid_related:
            ssd = sqeuclidean(norm_form[s1], norm_form[s2])
            pairs_ssd.append((s1, s2, ssd))

    # 3. Check if pairs_ssd is empty before sorting to prevent IndexError
    if not pairs_ssd:
        print("Error: No valid pairs could be formed. Check ticker symbols.")
    else:
        # Now safe to sort
        s1, s2, min_ssd = sorted(pairs_ssd, key=lambda x: x[2])[0]
        
        # Calculate Signals
        hist_std = (normalize(formation_df[s1]) - normalize(formation_df[s2])).std()
        current_spread = (trading_df[s1].iloc[-1]/trading_df[s1].iloc[0]) - \
                         (trading_df[s2].iloc[-1]/trading_df[s2].iloc[0])
        
        print(f"\nTop Pair: {s1} & {s2}")
        print(f"Status: {'DIVERGENCE' if abs(current_spread) > 2*hist_std else 'NEUTRAL'}")
