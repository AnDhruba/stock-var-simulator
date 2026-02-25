import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# 1
st.set_page_config(page_title="Stock VaR Simulator", layout="centered")
st.title("📈 Monte Carlo Stock Simulator")
st.write(
    "Project by [Dhrubajyoti Rajak](https://www.linkedin.com/in/dhrubajyoti-rajak-3649a6195/)")
st.write("Calculate the 95% Value at Risk (VaR) using Geometric Brownian Motion.")

# 2
col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("Enter Stock Ticker:", value="RELIANCE.NS").strip().upper()
with col2:
    initial_investment = st.number_input(
        "Initial Investment (₹):", min_value=1000, value=100000, step=1000)

# 3
if st.button("Run Simulation"):
    with st.spinner(f"Fetching 10-year data and simulating 1,000 paths for {ticker}..."):
        try:
            # 3a
            stock = yf.Ticker(ticker)
            data = stock.history(period="10y")

            if data.empty:
                st.error(
                    "No data found. Please check the ticker symbol (e.g., 'AAPL', 'TCS.NS').")
            else:
                # 3b
                data['Log_Returns'] = np.log(
                    data['Close'] / data['Close'].shift(1))
                data = data.dropna()

                u = data['Log_Returns'].mean()
                var = data['Log_Returns'].var()
                drift = u - (0.5 * var)
                stdev = data['Log_Returns'].std()

                # 3c
                t_intervals = 252
                iterations = 1000
                np.random.seed(42)
                Z = np.random.standard_normal((t_intervals, iterations))
                daily_returns = np.exp(drift + stdev * Z)

                # 4
                S0 = data['Close'].iloc[-1]
                price_list = np.zeros_like(daily_returns)
                price_list[0] = S0

                for t in range(1, t_intervals):
                    price_list[t] = price_list[t - 1] * daily_returns[t]

                # 5
                final_prices = price_list[-1]
                portfolio_final_values = (
                    final_prices / S0) * initial_investment
                var_95_value = np.percentile(portfolio_final_values, 5)
                var_amount = initial_investment - var_95_value

                # 6
                st.subheader(f"Simulation Results for {ticker}")
                st.write(f"**Current Price:** ₹{S0:,.2f}")

                # 7
                st.metric(label="95% Confidence VaR (1 Year)",
                          value=f"₹{var_amount:,.2f}", delta="Worst Case Loss", delta_color="inverse")

                # 8
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(price_list, linewidth=1, alpha=0.8)
                ax.set_title(f"1,000 Simulated Price Paths for {ticker}")
                ax.set_xlabel("Trading Days")
                ax.set_ylabel("Simulated Price (₹)")

                # 9
                st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred: {e}")
