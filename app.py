import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import time  # To simulate processing time

# Title of the web app
st.title('Simple Stock Price Prediction with Predefined Dataset')

# Sidebar for user input
st.sidebar.header('User Input')
n_days_predict = st.sidebar.number_input('Days to Predict Ahead', min_value=1, max_value=30, value=7)

# Load predefined dataset
@st.cache
def load_sample_data():
    # Here I'm using a sample dataset that mimics stock prices
    dates = pd.date_range(start="2020-01-01", periods=100)
    prices = np.sin(np.linspace(0, 10, 100)) * 100 + 500  # Simulated stock prices
    return pd.DataFrame({'Date': dates, 'Close': prices})

df = load_sample_data()

# Ensure no empty or null data
if df.empty:
    st.error("No data available.")
else:
    st.subheader('Sample Stock Data')
    st.write(df.tail())  # Show last 5 rows

    # Plotting the closing price
    def plot_stock_data():
        plt.figure(figsize=(10, 6))
        plt.plot(df['Date'], df['Close'], label='Closing Price')
        plt.title('Sample Stock Closing Price History')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(plt)

    plot_stock_data()

    # Prepare the data for Linear Regression
    df['Date'] = pd.to_datetime(df['Date']).map(pd.Timestamp.toordinal)  # Convert date to numerical format
    X = df['Date'].values.reshape(-1, 1)  # Feature (Date)
    Y = df['Close'].values.reshape(-1, 1)  # Target (Price)

    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

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
        plt.scatter(X_test, Y_test, color='green', label='Actual Prices')
        plt.plot(X_test, predictions, color='red', label='Predicted Prices')
        plt.title('Price Prediction (Test Data)')
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
    future_df = pd.DataFrame(future_predictions, columns=['Predicted Prices'], index=pd.date_range(df['Date'].max(), periods=n_days_predict+1, closed='right'))
    st.write(future_df)

    # Plot future stock prices
    def plot_future_predictions():
        future_days = pd.date_range(df['Date'].max(), periods=n_days_predict+1, closed='right')
        plt.figure(figsize=(10, 6))
        plt.plot(future_days, future_predictions, 'r', label='Predicted Future Prices')
        plt.title('Future Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(plt)

    plot_future_predictions()

