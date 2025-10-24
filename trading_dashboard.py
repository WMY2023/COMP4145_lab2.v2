import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load data
price_data = pd.read_csv('amazon_5_years_data.csv', index_col=0, parse_dates=True)
try:
    from trading_strategy import calculate_moving_averages, implement_strategy, identify_golden_cross
    # Recalculate moving averages and signals for chart
    price_data = calculate_moving_averages(price_data)
    price_data = identify_golden_cross(price_data)
    positions = implement_strategy(price_data)
except Exception:
    positions = pd.DataFrame()

# Calculate statistics
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

if choice == "Price Chart":
    st.title("Price Chart with Moving Averages and Trades")
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
