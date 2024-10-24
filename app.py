import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import yfinance as yf
import time  # To simulate processing time

# Title of the web app
st.title('Simple Stock Price Prediction')

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

# Check if 'Close' column exists and is not all NaN
if 'Close' not in df.columns or df['Close'].isnull().all():
    st.error("No 'Close' price data available for this stock or data is entirely null.")
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

    # Prepare the data for Linear Regression
    data = df.filter(['Close']).copy()
    data['Date'] = np.arange(0, len(data))  # Converting the Date to numerical format

    X = np.array(data['Date']).reshape(-1, 1)  # Feature
    Y = np.array(data['Close']).reshape(-1, 1)  # Target

    # Train-test split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    # Linear Regression model
    model = LinearRegression()

    # Train the model
    with st.spinner('Training the model...'):
        model.fit(X_train, Y_train)
        time.sleep(1)

    st.success('Model training completed!')

    # Predict stock prices on test set
    predictions = model.predict(X_test)

    # Plotting predicted vs actual prices
    st.subheader('Predicted vs Actual Closing Prices')
    def plot_predictions():
        plt.figure(figsize=(10, 6))
        plt.plot(Y_test, 'g', label='Actual Prices')
        plt.plot(predictions, 'r', label='Predicted Prices')
        plt.title(f'{stock} Price Prediction (Test Data)')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(plt)

    plot_predictions()

    # Predicting future stock prices
    last_day = X[-1][0]
    future_dates = np.arange(last_day + 1, last_day + n_days_predict + 1).reshape(-1, 1)
    future_predictions = model.predict(future_dates)

    # Display the predicted future prices
    st.subheader('Future Stock Price Predictions')
    st.write(future_predictions)

    # Plot future stock prices
    def plot_future_predictions():
        future_days = pd.date_range(end_date, periods=n_days_predict+1, closed='right')
        plt.figure(figsize=(10, 6))
        plt.plot(future_days, future_predictions, 'r', label='Predicted Future Prices')
        plt.title(f'{stock} Future Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(plt)

    plot_future_predictions()
