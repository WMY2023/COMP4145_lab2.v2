import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st
import matplotlib.pyplot as plt

# Download 5 years of MSFT data
def get_stock_data(ticker, period="5y"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data

# Calculate moving averages
def calculate_moving_averages(data):
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    return data

# Identify golden cross (buy signals)
def identify_golden_cross(data):
    data['Signal'] = 0  # Initialize signal column with 0
    # Golden Cross occurs when MA50 crosses above MA200
    data['GoldenCross'] = (data['MA50'] > data['MA200']) & (data['MA50'].shift(1) <= data['MA200'].shift(1))
    return data

# Implement trading strategy
def implement_strategy(data):
    positions = []

    # Need at least 200 days to calculate the 200-day MA
    data = data.iloc[200:].copy()

    buy_dates = data[data['GoldenCross'] == True].index.tolist()

    for buy_date in buy_dates:
        # Get buy price
        buy_price = data.loc[buy_date, 'Close']

        # Calculate target sell price (15% profit)
        target_price = buy_price * 1.15

        # Set maximum holding period
        max_sell_date = buy_date + pd.Timedelta(days=60)

        # Get data slice for potential sell period
        sell_period = data.loc[buy_date:max_sell_date].copy()

        # Check if target price is reached during the period
        target_reached = sell_period[sell_period['Close'] >= target_price]

        if not target_reached.empty:
            # Sell at first date target is reached
            sell_date = target_reached.index[0]
            sell_price = target_reached.loc[sell_date, 'Close']
            sell_reason = "Target reached"
        else:
            # Sell at end of maximum holding period
            sell_date_candidates = sell_period.index.tolist()
            if sell_date_candidates:
                sell_date = sell_date_candidates[-1]
                sell_price = data.loc[sell_date, 'Close']
                sell_reason = "Max holding period"
            else:
                # Skip if no valid sell date (should not happen in practice)
                continue

        # Calculate holding period in calendar days
        holding_days = (sell_date - buy_date).days

        # Calculate profit
        profit_pct = (sell_price / buy_price - 1) * 100

        positions.append({
            'BuyDate': buy_date,
            'BuyPrice': buy_price,
            'SellDate': sell_date,
            'SellPrice': sell_price,
            'HoldingDays': holding_days,
            'ProfitPct': profit_pct,
            'SellReason': sell_reason
        })

    return pd.DataFrame(positions)

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
def load_data():
    try:
        price_data = get_stock_data("MSFT")
        price_data = calculate_moving_averages(price_data)
        price_data = identify_golden_cross(price_data)
        positions = implement_strategy(price_data)
    except Exception:
        price_data = pd.DataFrame()
        positions = pd.DataFrame()
    return price_data, positions

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

menu = ["Price Chart", "Trade Statistics", "Detailed Trades"]
choice = st.sidebar.radio("Select Page", menu)

price_data, positions = load_data()

if choice == "Price Chart":
    st.title("Price Chart with Moving Averages and Trades")
    if not price_data.empty:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(price_data.index, price_data['Close'], label='Close Price', color='blue')
        if 'MA50' in price_data:
            ax.plot(price_data.index, price_data['MA50'], label='MA50', color='orange')
        if 'MA200' in price_data:
            ax.plot(price_data.index, price_data['MA200'], label='MA200', color='green')
        # Plot buy/sell points
        if not positions.empty:
            ax.scatter(positions['BuyDate'], positions['BuyPrice'], color='red', marker='o', label='Buy', zorder=5)
            ax.scatter(positions['SellDate'], positions['SellPrice'], color='purple', marker='o', label='Sell', zorder=5)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
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