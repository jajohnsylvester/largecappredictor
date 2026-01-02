import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Indian Stock Predictor", layout="wide")

st.title("📈 Indian Stock Price Predictor (LSTM)")
st.write("Predicting closing prices for Indian stocks using Long Short-Term Memory (LSTM).")

# a) User input for any Indian Stock
# Note: For Indian stocks on Yahoo Finance, append '.NS' (e.g., SBIN.NS)
ticker_input = st.sidebar.text_input("Enter NSE Ticker Symbol (e.g., SBIN, IOC, ADANIPOWER)", "SBIN")
stock_symbol = ticker_input.upper() + ".NS"

# Fetch Data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start="2015-01-01", end=datetime.now().strftime('%Y-%m-%d'))
    return data

data = load_data(stock_symbol)

if data.empty:
    st.error("No data found. Please check the ticker symbol (e.g., use 'SBIN' for SBI).")
else:
    st.subheader(f"Historical Data for {ticker_input.upper()}")
    st.write(data.tail())

    # Preprocessing
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Prepare Training/Testing sets (80% training)
    training_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[0:training_size, :]
    
    def create_dataset(dataset, time_step=60):
        X, y = [], []
        for i in range(len(dataset) - time_step - 1):
            X.append(dataset[i:(i + time_step), 0])
            y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 60
    X_train, y_train = create_dataset(train_data, time_step)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    # Simple LSTM Model
    if st.button('Train and Predict'):
        with st.spinner('Training model... please wait.'):
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

            # b) Interactive Plot with Hover display
            st.subheader("Prediction vs Actual Prices")
            fig = go.Figure()
            
            # Actual Price Trace
            fig.add_trace(go.Scatter(
                x=data.index[training_size:], 
                y=actual_prices.flatten(), 
                mode='lines', 
                name='Actual Price',
                line=dict(color='blue')
            ))
            
            # Predicted Price Trace
            fig.add_trace(go.Scatter(
                x=data.index[training_size:], 
                y=predictions.flatten(), 
                mode='lines', 
                name='Predicted Price',
                line=dict(color='orange')
            ))

            fig.update_layout(
                hovermode="x unified",
                xaxis_title="Date",
                yaxis_title="Price (INR)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig, use_container_width=True)
