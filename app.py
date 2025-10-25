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
    simple_forecast,
    generate_suggestion,
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


def run_selected_method(price_data, method):
    """Run the selected method on price_data and return positions DataFrame."""
    if method.startswith("Golden Cross"):
        return implement_strategy(price_data)
    elif method == "Bollinger Bands":
        return implement_bollinger_strategy(price_data)
    elif method == "OBV Strategy":
        return obv_strategy(price_data)
    elif method == "ATR Strategy":
        return atr_strategy(price_data)
    else:
        return pd.DataFrame()


def plot_price_obv_atr(price_data, positions, method=None, buy_color='red', sell_color='purple', forecast_df=None):
    """Return a matplotlib Figure showing price, OBV and ATR with buy/sell markers for the provided positions.

    The plot follows the same appearance as the main Chart page. If `method` is provided,
    MA or Bollinger lines are shown according to the method (Golden Cross shows MA50/MA200,
    Bollinger shows BB bands).
    """
    fig, (ax_price, ax_obv, ax_atr) = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})

    # Price plot
    ax_price.plot(price_data.index, price_data['Close'], label='Close Price', color='blue')
    # Render overlays conditionally to match Chart page
    if method and method.startswith("Golden Cross"):
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

    # Buy/sell on price axis (use same marker style as Chart page)
    if not positions.empty:
        ax_price.scatter(positions['BuyDate'], positions['BuyPrice'], color=buy_color, marker='o', label='Buy', zorder=5)
        ax_price.scatter(positions['SellDate'], positions['SellPrice'], color=sell_color, marker='o', label='Sell', zorder=5)

    # Forecast overlay
    if forecast_df is not None and not forecast_df.empty:
        try:
            ax_price.plot(forecast_df.index, forecast_df['Forecast'], linestyle='--', color='magenta', label='Forecast')
            ax_price.scatter(forecast_df.index, forecast_df['Forecast'], color='magenta', marker='D')
        except Exception:
            pass

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
    return fig



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

# Forecast controls
st.sidebar.markdown("---")
enable_forecast = st.sidebar.checkbox("Show forecast", value=False)
forecast_days = st.sidebar.number_input("Forecast days", min_value=1, max_value=60, value=5)
forecast_method = st.sidebar.selectbox("Forecast method", options=['linear', 'ma', 'ema'], index=0)
forecast_window = st.sidebar.number_input("Forecast window (history points)", min_value=5, max_value=365, value=60)

menu = ["Chart", "Trade Statistics", "Detailed Trades", "Compare Methods"]
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

# Always run the selected strategy for the current sidebar selections so graphs update reactively
positions = pd.DataFrame()
if not price_data.empty:
    positions = run_selected_method(price_data, method)

forecast_series = None
forecast_df = None
suggestion = None

# Diagnostics for troubleshooting why strategies may produce no trades
if not price_data.empty:
    ma_valid = price_data.dropna(subset=['MA50','MA200']) if {'MA50','MA200'}.issubset(price_data.columns) else pd.DataFrame()
    gc_count = int(price_data['GoldenCross'].sum()) if 'GoldenCross' in price_data else 0
    st.sidebar.markdown("**Data diagnostics**")
    st.sidebar.write(f"Rows fetched: {len(price_data)}")
    st.sidebar.write(f"Rows with MA50 & MA200: {len(ma_valid)}")
    st.sidebar.write(f"Golden Cross signals: {gc_count}")
    st.sidebar.write(f"Positions detected for selected method: {len(positions)}")
    if gc_count > 0:
        # show up to 5 sample buy dates
        sample_dates = price_data[price_data['GoldenCross']].index[:5].tolist()
        st.sidebar.write("Sample GoldenCross dates:")
        for d in sample_dates:
            st.sidebar.write(str(d))

if choice == "Chart":
    st.title(f"{method}: Price Chart")
    if not price_data.empty:
        # Use the shared plotting helper so Chart and Compare behave identically
        forecast_df = None
        if enable_forecast:
            # compute forecast (the helper may return a Series or a DataFrame)
            forecast_series = simple_forecast(price_data, days=int(forecast_days), method=forecast_method, window=int(forecast_window))
            # normalize to forecast_df (DataFrame with 'Forecast' column)
            if isinstance(forecast_series, pd.DataFrame):
                # already a DataFrame (expected with a 'Forecast' column)
                if 'Forecast' in forecast_series.columns:
                    forecast_df = forecast_series.copy()
                    suggestion = generate_suggestion(price_data, forecast_df['Forecast'])
                else:
                    # if DataFrame but different shape, try to take first column
                    first_col = forecast_series.columns[0]
                    forecast_df = forecast_series[[first_col]].rename(columns={first_col: 'Forecast'})
                    suggestion = generate_suggestion(price_data, forecast_df['Forecast'])
            elif isinstance(forecast_series, pd.Series):
                if not forecast_series.empty:
                    forecast_df = forecast_series.to_frame(name='Forecast')
                    suggestion = generate_suggestion(price_data, forecast_series)
            else:
                # unexpected type: ignore
                forecast_df = None
        fig = plot_price_obv_atr(price_data, positions, method=method, buy_color='red', sell_color='purple', forecast_df=forecast_df)
        st.pyplot(fig)
        # show suggestion if available
        if suggestion:
            st.markdown("**Prediction suggestion:**")
            st.info(f"{suggestion['recommendation']}: {suggestion['reason']}")
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

elif choice == "Compare Methods":
    st.title("Compare Two Methods")
    if price_data.empty:
        st.write("No price data available for the selected ticker/period.")
    else:
        # Allow user to pick two methods to compare
        col_a, col_b = st.columns(2)
        methods_list = [
            "Golden Cross (MA50/MA200)",
            "Bollinger Bands",
            "OBV Strategy",
            "ATR Strategy",
        ]
        with col_a:
            method1 = st.selectbox("Method 1", options=methods_list, index=0, key='method1')
        with col_b:
            method2 = st.selectbox("Method 2", options=methods_list, index=1, key='method2')

        run_compare = st.button("Run Comparison")

        # Compute both methods for the current selections (reactive). The button remains for UX but
        # we compute automatically so charts follow sidebar selections.
        pos1 = run_selected_method(price_data, method1)
        pos2 = run_selected_method(price_data, method2)

        # Show side-by-side charts
        left, right = st.columns(2)
        with left:
            st.header(method1)
            fig1 = plot_price_obv_atr(price_data, pos1, method=method1, buy_color='red', sell_color='purple')
            st.pyplot(fig1)
            stats1 = get_statistics(pos1)
            if stats1:
                st.table(pd.DataFrame(stats1, index=[0]))
            else:
                st.write("No trades for this method.")

        with right:
            st.header(method2)
            fig2 = plot_price_obv_atr(price_data, pos2, method=method2, buy_color='green', sell_color='black')
            st.pyplot(fig2)
            stats2 = get_statistics(pos2)
            if stats2:
                st.table(pd.DataFrame(stats2, index=[0]))
            else:
                st.write("No trades for this method.")

        # Combined summary
        st.subheader("Combined Comparison")
        df_compare = pd.DataFrame({
            'Metric': list((stats1 or {}).keys()) if (stats1 or {}) else [],
        })
        # Build a comparison table for common metrics
        if stats1 or stats2:
            all_metrics = set()
            if stats1:
                all_metrics.update(stats1.keys())
            if stats2:
                all_metrics.update(stats2.keys())
            rows = []
            for k in sorted(all_metrics):
                rows.append({
                    'Metric': k,
                    'Method 1': (stats1.get(k) if stats1 else None),
                    'Method 2': (stats2.get(k) if stats2 else None),
                })
            st.table(pd.DataFrame(rows))