import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st
import matplotlib.pyplot as plt
from trading_methods import (
    calculate_obv,
    calculate_atr,
    calculate_moving_averages,
    identify_golden_cross,
    implement_strategy,
    calculate_bollinger_bands,
    implement_bollinger_strategy,
    obv_strategy,
    atr_strategy,
)

# Download stock data for a given ticker and period
def get_stock_data(ticker, period="5y", interval="1d"):
    stock = yf.Ticker(ticker)
    # Pass interval to history so users can choose weekly/monthly views
    data = stock.history(period=period, interval=interval)
    return data



 

# Analyze the results
def analyze_results(positions):
    if positions.empty:
        return "No trading signals detected"

    # Summary statistics
    total_trades = len(positions)
    win_trades = len(positions[positions['ProfitPct'] > 0])
    loss_trades = total_trades - win_trades
    win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0

    avg_profit = positions['ProfitPct'].mean()
    avg_win = positions[positions['ProfitPct'] > 0]['ProfitPct'].mean() if win_trades > 0 else 0
    avg_loss = positions[positions['ProfitPct'] <= 0]['ProfitPct'].mean() if loss_trades > 0 else 0

    avg_holding = positions['HoldingDays'].mean()

    target_reached = len(positions[positions['SellReason'] == 'Target reached'])
    max_period = len(positions[positions['SellReason'] == 'Max holding period'])

    print("\n===== Trading Strategy Results (Golden Cross)=====")
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {win_trades} ({win_rate:.2f}%)")
    print(f"Losing Trades: {loss_trades}")
    print(f"Average Profit: {avg_profit:.2f}%")

    return positions

# Load data
def load_data(ticker="MSFT", period="5y", interval="1d"):
    try:
        price_data = get_stock_data(ticker, period=period, interval=interval)
        price_data = calculate_moving_averages(price_data)
        price_data = identify_golden_cross(price_data)
    except Exception:
        price_data = pd.DataFrame()
    return price_data

def get_statistics(positions):
    if positions.empty:
        return {}
    total_trades = len(positions)
    win_trades = len(positions[positions['ProfitPct'] > 0])
    loss_trades = total_trades - win_trades
    win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0
    avg_profit = positions['ProfitPct'].mean()
    avg_win = positions[positions['ProfitPct'] > 0]['ProfitPct'].mean() if win_trades > 0 else 0
    avg_loss = positions[positions['ProfitPct'] <= 0]['ProfitPct'].mean() if loss_trades > 0 else 0
    avg_holding = positions['HoldingDays'].mean()
    return {
        'Total Trades': total_trades,
        'Winning Trades': win_trades,
        'Losing Trades': loss_trades,
        'Win Rate (%)': win_rate,
        'Average Profit (%)': avg_profit,
        'Average Win (%)': avg_win,
        'Average Loss (%)': avg_loss,
        'Average Holding Days': avg_holding
    }



# Streamlit app
st.set_page_config(page_title="Trading Strategy Dashboard", layout="wide")

# Sidebar controls: choose ticker, period, and method
st.sidebar.header("Data & Strategy")
ticker = st.sidebar.text_input("Ticker (USA stocks, e.g. AAPL, MSFT, AMZN)", value="MSFT")
period = st.sidebar.selectbox("Period", options=["1wk", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=4)
interval = st.sidebar.selectbox("Interval", options=["1d", "1wk", "1mo"], index=0)
method = st.sidebar.selectbox("Strategy / Method", options=[
    "Golden Cross (MA50/MA200)",
    "Bollinger Bands",
    "OBV Strategy",
    "ATR Strategy",
], index=0)
run = st.sidebar.button("Run")

menu = ["Chart", "Trade Statistics", "Detailed Trades"]
choice = st.sidebar.radio("Select Page", menu)

# Load data for selected ticker/period
price_data = load_data(ticker=ticker, period=period, interval=interval)

# Always calculate Bollinger Bands for charting convenience
if not price_data.empty:
    price_data = calculate_bollinger_bands(price_data)
    price_data = calculate_moving_averages(price_data)
    price_data = identify_golden_cross(price_data)
    # Calculate OBV and ATR
    price_data = calculate_obv(price_data)
    price_data = calculate_atr(price_data, window=14)

# Run strategy when user clicks Run (or automatically if data already loaded)
positions = pd.DataFrame()
if run and not price_data.empty:
    if method.startswith("Golden Cross"):
        positions = implement_strategy(price_data)
    elif method == "Bollinger Bands":
        positions = implement_bollinger_strategy(price_data)
    elif method == "OBV Strategy":
        positions = obv_strategy(price_data)
    elif method == "ATR Strategy":
        positions = atr_strategy(price_data)
else:
    # Default behavior: if not run but data exists and method is Golden Cross, prepare positions so pages show something
    if not price_data.empty and method.startswith("Golden Cross"):
        positions = implement_strategy(price_data)
    elif not price_data.empty and method == "Bollinger Bands":
        positions = implement_bollinger_strategy(price_data)
    elif not price_data.empty and method == "OBV Strategy":
        positions = obv_strategy(price_data)
    elif not price_data.empty and method == "ATR Strategy":
        positions = atr_strategy(price_data)

if choice == "Chart":
    st.title(f"{method}: Price Chart")
    if not price_data.empty:
        # Create 3-row subplot: price, OBV, ATR
        fig, (ax_price, ax_obv, ax_atr) = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})

        # Price plot
        ax_price.plot(price_data.index, price_data['Close'], label='Close Price', color='blue')
        if method.startswith("Golden Cross"):
            if 'MA50' in price_data:
                ax_price.plot(price_data.index, price_data['MA50'], label='MA50', color='orange')
            if 'MA200' in price_data:
                ax_price.plot(price_data.index, price_data['MA200'], label='MA200', color='green')
        elif method == "Bollinger Bands":
            if 'BB_Middle' in price_data:
                ax_price.plot(price_data.index, price_data['BB_Middle'], label='BB Middle', color='orange')
            if 'BB_Upper' in price_data:
                ax_price.plot(price_data.index, price_data['BB_Upper'], label='BB Upper', color='green', linestyle='--')
            if 'BB_Lower' in price_data:
                ax_price.plot(price_data.index, price_data['BB_Lower'], label='BB Lower', color='red', linestyle='--')

        # Buy/sell on price axis
        if not positions.empty:
            ax_price.scatter(positions['BuyDate'], positions['BuyPrice'], color='red', marker='o', label='Buy', zorder=5)
            ax_price.scatter(positions['SellDate'], positions['SellPrice'], color='purple', marker='o', label='Sell', zorder=5)

        ax_price.set_ylabel('Price')
        ax_price.legend(loc='upper left')

        # OBV plot
        if 'OBV' in price_data:
            ax_obv.plot(price_data.index, price_data['OBV'], label='OBV', color='black')
            ax_obv.set_ylabel('OBV')
        else:
            ax_obv.text(0.5, 0.5, 'OBV not available', ha='center', va='center')

        # ATR plot
        if 'ATR' in price_data:
            ax_atr.plot(price_data.index, price_data['ATR'], label='ATR', color='brown')
            ax_atr.set_ylabel('ATR')
        else:
            ax_atr.text(0.5, 0.5, 'ATR not available', ha='center', va='center')

        ax_atr.set_xlabel('Date')
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.write("No price data available.")

elif choice == "Trade Statistics":
    st.title("Trade Statistics Summary")
    stats = get_statistics(positions)
    if stats:
        st.table(pd.DataFrame(stats, index=[0]))
    else:
        st.write("No trades to summarize.")

elif choice == "Detailed Trades":
    st.title("Detailed Trade Records")
    if not positions.empty:
        st.dataframe(positions)
    else:
        st.write("No trade records available.")