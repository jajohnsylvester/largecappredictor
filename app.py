import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="Universal Stock Predictor", layout="wide")
st.title("📈 Stock Price Predictor (LSTM)")

# 1. Sidebar for Custom Inputs
st.sidebar.header("Stock Selection")

# Option to choose from list or enter custom
stock_option = st.sidebar.radio("Input Type", ["Select from List", "Enter Custom Ticker"])

if stock_option == "Select from List":
    stock_dict = {"SBI": "SBIN.NS", "Indian Oil": "IOC.NS", "Adani Power": "ADANIPOWER.NS"}
    selected_stock = st.sidebar.selectbox("Choose Stock", list(stock_dict.keys()))
    ticker = stock_dict[selected_stock]
else:
    custom_ticker = st.sidebar.text_input("Enter Ticker (e.g., RELIANCE, TATAMOTORS)", "RELIANCE").upper()
    # Auto-append .NS for Indian stocks if not provided
    ticker = custom_ticker if "." in custom_ticker else f"{custom_ticker}.NS"
    selected_stock = custom_ticker

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# 2. Fetch Data
@st.cache_data
def load_data(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end)
        if df.empty:
            return None
        return df
    except Exception:
        return None

data = load_data(ticker, start_date, end_date)

if data is not None:
    st.subheader(f"Historical Data for {ticker}")
    st.write(data.tail())

    # Interactive Historical Chart
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=data.index, y=data['Close'].values.flatten(), name="Close Price"))
    fig_hist.update_layout(title=f"Price History: {ticker}", hovermode="x unified")
    st.plotly_chart(fig_hist, use_container_width=True)

    # 3. Model Preparation
    st.subheader("LSTM Prediction Model")
    
    if st.button("Train & Predict"):
        with st.spinner(f"Fetching data and training LSTM for {ticker}..."):
            # Data scaling
            close_prices = data[['Close']].values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(close_prices)

            train_size = int(len(scaled_data) * 0.8)
            train_data = scaled_data[:train_size]
            test_data = scaled_data[train_size:]

            def create_sequences(dataset, time_step=60):
                X, y = [], []
                for i in range(len(dataset) - time_step):
                    X.append(dataset[i:(i + time_step), 0])
                    y.append(dataset[i + time_step, 0])
                return np.array(X), np.array(y)

            time_step = 60
            if len(test_data) <= time_step:
                st.error("Not enough data for the selected date range. Try a wider range.")
            else:
                X_train, y_train = create_sequences(train_data, time_step)
                X_test, y_test = create_sequences(test_data, time_step)

                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

                # Define LSTM Model
                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
                    Dropout(0.2),
                    LSTM(50, return_sequences=False),
                    Dropout(0.2),
                    Dense(25),
                    Dense(1)
                ])

                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(X_train, y_train, batch_size=32, epochs=2, verbose=0) # Epochs reduced for speed

                # Predictions
                predictions = model.predict(X_test)
                predictions = scaler.inverse_transform(predictions)
                actual_prices = close_prices[train_size + time_step:]
                test_dates = data.index[train_size + time_step:]

                # Interactive Plot
                fig_res = go.Figure()
                fig_res.add_trace(go.Scatter(x=test_dates, y=actual_prices.flatten(), name="Actual", line=dict(color='blue')))
                fig_res.add_trace(go.Scatter(x=test_dates, y=predictions.flatten(), name="Predicted", line=dict(color='red', dash='dash')))
                fig_res.update_layout(title="Prediction vs Actual (Hover to see prices)", hovermode="x unified")
                st.plotly_chart(fig_res, use_container_width=True)
                st.success("Analysis Complete!")
else:
    st.error("Invalid Ticker or No Data Found. Please check the symbol (e.g., TCS, INFEY).")
