import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import yfinance as yf
import time  # To simulate processing time

# Title of the web app
st.title('Stock Price Prediction using LSTM')

# Sidebar for user input
st.sidebar.header('User Input')
start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2012-01-01'))
end_date = st.sidebar.date_input('End Date', value=pd.to_datetime('2022-12-21'))
stock = st.sidebar.text_input('Stock Ticker', 'GOOG')
n_days_predict = st.sidebar.number_input('Days to Predict Ahead', min_value=1, max_value=30, value=7)

# Fetch the stock data
@st.cache
def load_stock_data(ticker, start, end):
    return yf.download(ticker, start, end)

df = load_stock_data(stock, start_date, end_date)

# Check if 'Close' column exists and contains data
if 'Close' not in df.columns or df['Close'].isnull().all():
    st.error("No 'Close' price data available for this stock.")
else:
    # Display the raw data
    st.subheader(f'Raw data of {stock}')
    st.write(df.tail())

    # Plotting the closing price
    def plot_stock_data():
        plt.figure(figsize=(10, 6))
        plt.plot(df['Close'], label='Closing Price')
        plt.title(f'{stock} Closing Price History')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(plt)

    plot_stock_data()

    # Prepare the data for training the LSTM
    data = df.filter(['Close'])
    dataset = data.values

    # Check if dataset has any data before scaling
    if dataset.size == 0:
        st.error("Insufficient data to perform scaling and prediction.")
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        train_data_len = int(len(scaled_data) * 0.8)
        train_data = scaled_data[0:train_data_len]
        test_data = scaled_data[train_data_len - 100:]

        # Create training datasets
        def create_dataset(data, time_step=100):
            x, y = [], []
            for i in range(time_step, len(data)):
                x.append(data[i-time_step:i])
                y.append(data[i, 0])
            return np.array(x), np.array(y)

        X_train, Y_train = create_dataset(train_data)
        X_test, Y_test = create_dataset(test_data)

        # Reshaping the data to 3D for LSTM
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(60, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(80, return_sequences=True))
        model.add(Dropout(0.4))
        model.add(LSTM(120))
        model.add(Dropout(0.5))
        model.add(Dense(1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model (epochs fixed at 1 for reduced computing time)
        with st.spinner('Training the model...'):
            model.fit(X_train, Y_train, epochs=1, batch_size=32)
            time.sleep(1)  # Simulating delay for better UX

        st.success('Model training completed!')

        # Predicting stock prices
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)

        # Plotting predicted vs actual prices
        st.subheader('Predicted vs Actual Closing Prices')
        def plot_predictions():
            plt.figure(figsize=(10, 6))
            plt.plot(scaler.inverse_transform(Y_test.reshape(-1, 1)), 'g', label='Actual Prices')
            plt.plot(predictions, 'r', label='Predicted Prices')
            plt.title(f'{stock} Price Prediction')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(plt)

        plot_predictions()

        # Predicting future stock prices
        last_100_days_data = scaled_data[-100:]
        last_100_days_data = np.reshape(last_100_days_data, (1, last_100_days_data.shape[0], 1))

        predicted_future = []
        with st.spinner('Predicting future stock prices...'):
            for _ in range(n_days_predict):
                prediction = model.predict(last_100_days_data)
                predicted_future.append(prediction[0][0])

                # Update the input for the next prediction
                next_input = np.concatenate([last_100_days_data[:, 1:, :], np.reshape(prediction, (1, 1, 1))], axis=1)
                last_100_days_data = next_input
            time.sleep(1)  # Simulating delay

        # Scale back the future predictions
        predicted_future_prices = scaler.inverse_transform(np.array(predicted_future).reshape(-1, 1))

        # Display the predicted future prices
        st.subheader('Future Stock Price Predictions')
        st.write(predicted_future_prices)

        # Plot future stock prices
        def plot_future_predictions():
            future_dates = pd.date_range(end_date, periods=n_days_predict+1, closed='right')
            plt.figure(figsize=(10, 6))
            plt.plot(future_dates, predicted_future_prices, 'r', label='Predicted Future Prices')
            plt.title(f'{stock} Future Stock Price Prediction')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(plt)

        plot_future_predictions()
