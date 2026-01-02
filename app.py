import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(page_title="Indian Stock Predictor", layout="wide")

st.title("📈 Indian Stock Price Predictor (LSTM)")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("Configuration")

# 1. Ticker Input
ticker_input = st.sidebar.text_input("Enter NSE Ticker Symbol", "SBIN")
stock_symbol = ticker_input.upper() + ".NS"

# 2. Start Date Selection
default_start = datetime.now() - timedelta(days=365*5) # Default 5 years ago
start_date = st.sidebar.date_input("Select Start Date", default_start)

# 3. Time Step Selection
time_step = st.sidebar.slider("Select Time Step (Look-back days)", min_value=10, max_value=120, value=60)

# Fetch Data
@st.cache_data
def load_data(ticker, start):
    data = yf.download(ticker, start=start, end=datetime.now().strftime('%Y-%m-%d'))
    return data

data = load_data(stock_symbol, start_date)

if data.empty:
    st.error("No data found. Please check the ticker symbol and date range.")
else:
    # --- TOP METRICS LAYOUT ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Historical Data for {ticker_input.upper()}")
        st.write(data.tail(3))

    # Preprocessing
    # Ensure we use 'Close' column. Handle Multi-index if yfinance returns it.
    if isinstance(data.columns, pd.MultiIndex):
        close_prices = data['Close'][stock_symbol].values.reshape(-1, 1)
    else:
        close_prices = data['Close'].values.reshape(-1, 1)
        
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Prepare Training/Testing sets
    training_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[0:training_size, :]
    
    def create_dataset(dataset, time_step):
        X, y = [], []
        for i in range(len(dataset) - time_step - 1):
            X.append(dataset[i:(i + time_step), 0])
            y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_dataset(train_data, time_step)
    
    # Check if we have enough data for the chosen time_step
    if len(X_train) == 0:
        st.warning("Not enough data for the selected Time Step. Please choose a longer date range or smaller time step.")
    else:
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        if st.button('Train and Predict'):
            with st.spinner('Training model...'):
                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
                    LSTM(50, return_sequences=False),
                    Dense(25),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(X_train, y_train, batch_size=32, epochs=5, verbose=0)

                # Testing set
                test_data = scaled_data[training_size - time_step:, :]
                X_test, y_test = create_dataset(test_data, time_step)
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

                # Predictions
                predictions = model.predict(X_test)
                predictions = scaler.inverse_transform(predictions)
                actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

                # Display Predicted Value in Top Right
                latest_pred = predictions[-1][0]
                latest_date = data.index[-1].strftime('%Y-%m-%d')
                
                with col2:
                    st.metric(label=f"Latest Predicted Close ({latest_date})", 
                              value=f"₹{latest_pred:.2f}")

                # Interactive Plot
                st.subheader("Prediction vs Actual Prices")
                fig = go.Figure()
                
                test_dates = data.index[training_size + time_step + 1:]

                fig.add_trace(go.Scatter(
                    x=test_dates, 
                    y=actual_prices.flatten(), 
                    mode='lines', 
                    name='Actual Price',
                    line=dict(color='#1f77b4')
                ))
                
                fig.add_trace(go.Scatter(
                    x=test_dates, 
                    y=predictions.flatten(), 
                    mode='lines', 
                    name='Predicted Price',
                    line=dict(color='#ff7f0e')
                ))

                fig.update_layout(
                    hovermode="x unified",
                    xaxis_title="Date",
                    yaxis_title="Price (INR)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=0, r=0, t=50, b=0)
                )

                st.plotly_chart(fig, use_container_width=True)
