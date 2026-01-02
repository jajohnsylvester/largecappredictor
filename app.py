import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Page Configuration
st.set_page_config(page_title="Indian Stock Prediction", layout="wide")
st.title("📈 Indian Stock Price Predictor (LSTM)")

# 1. Sidebar for Inputs
st.sidebar.header("User Input Parameters")
stock_dict = {
    "SBI": "SBIN.NS",
    "Indian Oil Corp": "IOC.NS",
    "Adani Power": "ADANIPOWER.NS"
}
selected_stock = st.sidebar.selectbox("Select Stock", list(stock_dict.keys()))
ticker = stock_dict[selected_stock]

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# 2. Fetch Data
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

data = load_data(ticker, start_date, end_date)

st.subheader(f"Historical Data for {selected_stock}")
st.write(data.tail())

# Plot Closing Price
st.line_chart(data['Close'])

# 3. Data Preprocessing
st.subheader("Model Training & Prediction")

# Using 'Close' price for prediction
df = data[['Close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Prepare Training Data
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
X_train, y_train = create_sequences(train_data, time_step)
X_test, y_test = create_sequences(test_data, time_step)

# Reshape for LSTM [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# 4. Build & Train LSTM Model
if st.button("Train Model & Predict"):
    with st.spinner("Training the LSTM model... please wait."):
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=32, epochs=5, verbose=0)

        # 5. Predictions
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        
        # Actual values for comparison
        actual_prices = df[train_size + time_step:]

        # 6. Visualization
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(actual_prices, color='blue', label="Actual Price")
        ax.plot(predictions, color='red', label="Predicted Price")
        ax.set_title(f"{selected_stock} Price Prediction")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)
        
        st.success("Prediction Complete!")
else:
    st.info("Click the button in the sidebar to start training and see predictions.")
