import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Set up the Streamlit app title
st.title('Stock Price Prediction App')

# Sidebar for user inputs
st.sidebar.header('User Input')
start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2012-01-01'))
end_date = st.sidebar.date_input('End Date', value=pd.to_datetime('2022-12-21'))
stock = st.sidebar.text_input('Stock Ticker', 'GOOG')

# Load data
st.subheader(f'Displaying data for {stock} from {start_date} to {end_date}')
try:
    df = yf.download(stock, start=start_date, end=end_date)

    # Show the first few rows of data
    st.write("Data preview:")
    st.dataframe(df.head())

    # Basic EDA: Display summary statistics
    st.subheader('Basic Statistics')
    st.write(df.describe())

    # Plot stock closing price
    st.subheader('Closing Price over Time')
    plt.figure(figsize=(10, 6))
    plt.plot(df['Close'], label=f'{stock} Close Price')
    plt.title(f'{stock} Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

except Exception as e:
    st.error(f"Error: {e}")

