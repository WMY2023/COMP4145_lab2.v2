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


def get_portfolio_data(tickers, period="1y", interval="1d"):
    """Get stock data for multiple tickers in portfolio analysis."""
    portfolio_data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=interval)
            if not data.empty:
                portfolio_data[ticker] = data
        except Exception as e:
            st.warning(f"Failed to fetch data for {ticker}: {str(e)}")
    return portfolio_data

def calculate_portfolio_metrics(portfolio_data):
    """Calculate portfolio performance metrics."""
    if not portfolio_data:
        return {}
    
    metrics = {}
    portfolio_returns = pd.DataFrame()
    
    for ticker, data in portfolio_data.items():
        if not data.empty and len(data) > 1:
            # Calculate daily returns
            returns = data['Close'].pct_change().dropna()
            portfolio_returns[ticker] = returns
            
            # Individual stock metrics
            total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
            sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative / rolling_max - 1) * 100
            max_drawdown = drawdown.min()
            
            metrics[ticker] = {
                'Total Return (%)': round(total_return, 2),
                'Volatility (%)': round(volatility, 2),
                'Sharpe Ratio': round(sharpe_ratio, 3),
                'Max Drawdown (%)': round(max_drawdown, 2),
                'Current Price': round(data['Close'].iloc[-1], 2),
                'Start Price': round(data['Close'].iloc[0], 2)
            }
    
    return metrics, portfolio_returns

def calculate_portfolio_correlation(portfolio_returns):
    """Calculate correlation matrix for portfolio stocks."""
    if portfolio_returns.empty:
        return pd.DataFrame()
    return portfolio_returns.corr()

def calculate_equal_weight_portfolio(portfolio_returns):
    """Calculate equal-weighted portfolio performance."""
    if portfolio_returns.empty:
        return {}
    
    # Equal weight portfolio returns
    portfolio_return = portfolio_returns.mean(axis=1)
    
    # Portfolio metrics
    total_return = ((1 + portfolio_return).cumprod().iloc[-1] - 1) * 100
    volatility = portfolio_return.std() * np.sqrt(252) * 100
    sharpe_ratio = (portfolio_return.mean() * 252) / (portfolio_return.std() * np.sqrt(252)) if portfolio_return.std() != 0 else 0
    
    # Portfolio drawdown
    cumulative = (1 + portfolio_return).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative / rolling_max - 1) * 100
    max_drawdown = drawdown.min()
    
    return {
        'Total Return (%)': round(total_return, 2),
        'Volatility (%)': round(volatility, 2),
        'Sharpe Ratio': round(sharpe_ratio, 3),
        'Max Drawdown (%)': round(max_drawdown, 2)
    }

def plot_portfolio_performance(portfolio_data):
    """Plot normalized performance of portfolio stocks."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for ticker, data in portfolio_data.items():
        if not data.empty:
            # Normalize to 100 at start
            normalized = (data['Close'] / data['Close'].iloc[0]) * 100
            ax.plot(data.index, normalized, label=ticker, linewidth=2)
    
    ax.set_title('Portfolio Performance (Normalized to 100)', fontsize=16)
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Price')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig

def plot_correlation_heatmap(correlation_matrix):
    """Plot correlation heatmap for portfolio stocks."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(correlation_matrix.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(range(len(correlation_matrix.columns)))
    ax.set_yticks(range(len(correlation_matrix.index)))
    ax.set_xticklabels(correlation_matrix.columns)
    ax.set_yticklabels(correlation_matrix.index)
    
    # Add correlation values as text
    for i in range(len(correlation_matrix.index)):
        for j in range(len(correlation_matrix.columns)):
            text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Portfolio Correlation Matrix', fontsize=16)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    return fig

def plot_portfolio_comparison(portfolio_data1, portfolio_data2, portfolio_name1="Portfolio 1", portfolio_name2="Portfolio 2"):
    """Plot side-by-side comparison of two portfolios."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Portfolio 1
    for ticker, data in portfolio_data1.items():
        if not data.empty:
            normalized = (data['Close'] / data['Close'].iloc[0]) * 100
            ax1.plot(data.index, normalized, label=ticker, linewidth=2)
    
    ax1.set_title(f'{portfolio_name1} Performance (Normalized to 100)', fontsize=14)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Normalized Price')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Portfolio 2
    for ticker, data in portfolio_data2.items():
        if not data.empty:
            normalized = (data['Close'] / data['Close'].iloc[0]) * 100
            ax2.plot(data.index, normalized, label=ticker, linewidth=2)
    
    ax2.set_title(f'{portfolio_name2} Performance (Normalized to 100)', fontsize=14)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Normalized Price')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_portfolio_overlay_comparison(portfolio_returns1, portfolio_returns2, portfolio_name1="Portfolio 1", portfolio_name2="Portfolio 2"):
    """Plot overlay comparison of two portfolios' cumulative returns."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Calculate equal-weighted portfolio returns
    if not portfolio_returns1.empty:
        portfolio_return1 = portfolio_returns1.mean(axis=1)
        cumulative1 = (1 + portfolio_return1).cumprod() * 100
        ax.plot(cumulative1.index, cumulative1, label=portfolio_name1, linewidth=3, color='blue')
    
    if not portfolio_returns2.empty:
        portfolio_return2 = portfolio_returns2.mean(axis=1)
        cumulative2 = (1 + portfolio_return2).cumprod() * 100
        ax.plot(cumulative2.index, cumulative2, label=portfolio_name2, linewidth=3, color='red')
    
    ax.set_title('Portfolio Performance Comparison (Cumulative Returns)', fontsize=16)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Value (Starting at 100)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig

def create_comparison_metrics_table(metrics1, metrics2, portfolio_name1="Portfolio 1", portfolio_name2="Portfolio 2"):
    """Create a comparison table of portfolio metrics."""
    if not metrics1 or not metrics2:
        return pd.DataFrame()
    
    comparison_data = []
    
    # Get all unique metrics
    all_metrics = set()
    all_metrics.update(metrics1.keys())
    all_metrics.update(metrics2.keys())
    
    for metric in ['Total Return (%)', 'Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)']:
        if metric in all_metrics:
            comparison_data.append({
                'Metric': metric,
                portfolio_name1: metrics1.get(metric, 'N/A'),
                portfolio_name2: metrics2.get(metric, 'N/A'),
                'Difference': (metrics1.get(metric, 0) - metrics2.get(metric, 0)) if metric in metrics1 and metric in metrics2 else 'N/A'
            })
    
    return pd.DataFrame(comparison_data)

def validate_ticker_symbol(ticker):
    """Validate if a ticker symbol exists and is tradeable."""
    try:
        stock = yf.Ticker(ticker.upper())
        # Try to get basic info to validate the ticker
        info = stock.info
        # Check if it's a valid stock (has market cap or other essential data)
        if info and ('marketCap' in info or 'totalCash' in info or 'regularMarketPrice' in info):
            return True, info.get('longName', ticker.upper())
    except Exception:
        pass
    return False, None

def search_stocks_by_name_or_ticker(query, max_results=10):
    """Search for stocks by company name or ticker symbol."""
    # This is a simplified search - in a production app, you'd use a proper stock search API
    # For now, we'll validate individual tickers and provide some common stocks that match
    
    query = query.upper().strip()
    results = []
    
    # If it looks like a direct ticker, validate it
    if len(query) <= 5 and query.isalpha():
        is_valid, company_name = validate_ticker_symbol(query)
        if is_valid:
            results.append({'ticker': query, 'name': company_name})
    
    # Extended list of popular US stocks for search matching
    extended_stock_list = {
        # Tech
        'AAPL': 'Apple Inc.', 'MSFT': 'Microsoft Corporation', 'GOOGL': 'Alphabet Inc.', 'GOOG': 'Alphabet Inc.',
        'AMZN': 'Amazon.com Inc.', 'META': 'Meta Platforms Inc.', 'NVDA': 'NVIDIA Corporation', 'TSLA': 'Tesla Inc.',
        'NFLX': 'Netflix Inc.', 'CRM': 'Salesforce Inc.', 'ORCL': 'Oracle Corporation', 'ADBE': 'Adobe Inc.',
        'INTC': 'Intel Corporation', 'AMD': 'Advanced Micro Devices', 'PYPL': 'PayPal Holdings', 'UBER': 'Uber Technologies',
        'SPOT': 'Spotify Technology', 'SNAP': 'Snap Inc.', 'TWTR': 'Twitter Inc.', 'ZOOM': 'Zoom Video Communications',
        
        # Healthcare
        'JNJ': 'Johnson & Johnson', 'UNH': 'UnitedHealth Group', 'PFE': 'Pfizer Inc.', 'ABBV': 'AbbVie Inc.',
        'TMO': 'Thermo Fisher Scientific', 'ABT': 'Abbott Laboratories', 'LLY': 'Eli Lilly and Company',
        'DHR': 'Danaher Corporation', 'BMY': 'Bristol-Myers Squibb', 'MRK': 'Merck & Co Inc.',
        'GILD': 'Gilead Sciences', 'AMGN': 'Amgen Inc.', 'CVS': 'CVS Health Corporation', 'ANTM': 'Anthem Inc.',
        
        # Finance
        'JPM': 'JPMorgan Chase & Co.', 'BAC': 'Bank of America Corp', 'WFC': 'Wells Fargo & Company',
        'GS': 'Goldman Sachs Group', 'MS': 'Morgan Stanley', 'C': 'Citigroup Inc.', 'AXP': 'American Express Company',
        'BLK': 'BlackRock Inc.', 'SPGI': 'S&P Global Inc.', 'V': 'Visa Inc.', 'MA': 'Mastercard Incorporated',
        'SCHW': 'Charles Schwab Corporation', 'USB': 'U.S. Bancorp', 'TFC': 'Truist Financial Corporation',
        
        # Consumer & Retail
        'PG': 'Procter & Gamble Company', 'KO': 'Coca-Cola Company', 'PEP': 'PepsiCo Inc.',
        'WMT': 'Walmart Inc.', 'HD': 'Home Depot Inc.', 'MCD': 'McDonald\'s Corporation', 'NKE': 'Nike Inc.',
        'SBUX': 'Starbucks Corporation', 'TGT': 'Target Corporation', 'COST': 'Costco Wholesale Corporation',
        'LOW': 'Lowe\'s Companies Inc.', 'DIS': 'Walt Disney Company', 'CMCSA': 'Comcast Corporation',
        
        # Energy
        'XOM': 'Exxon Mobil Corporation', 'CVX': 'Chevron Corporation', 'COP': 'ConocoPhillips',
        'EOG': 'EOG Resources Inc.', 'SLB': 'Schlumberger Limited', 'MPC': 'Marathon Petroleum Corporation',
        'PSX': 'Phillips 66', 'VLO': 'Valero Energy Corporation', 'OXY': 'Occidental Petroleum Corporation',
        'BKR': 'Baker Hughes Company', 'KMI': 'Kinder Morgan Inc.', 'WMB': 'Williams Companies Inc.',
        
        # Industrial
        'BA': 'Boeing Company', 'CAT': 'Caterpillar Inc.', 'GE': 'General Electric Company',
        'MMM': '3M Company', 'HON': 'Honeywell International', 'UPS': 'United Parcel Service',
        'RTX': 'Raytheon Technologies', 'LMT': 'Lockheed Martin Corporation', 'DE': 'Deere & Company',
        'FDX': 'FedEx Corporation', 'EMR': 'Emerson Electric Co.', 'ETN': 'Eaton Corporation',
        
        # Utilities & Real Estate
        'NEE': 'NextEra Energy Inc.', 'DUK': 'Duke Energy Corporation', 'SO': 'Southern Company',
        'D': 'Dominion Energy Inc.', 'AEP': 'American Electric Power', 'AMT': 'American Tower Corporation',
        'PLD': 'Prologis Inc.', 'CCI': 'Crown Castle International', 'EQIX': 'Equinix Inc.',
        
        # Additional Popular Stocks
        'BRK-A': 'Berkshire Hathaway Inc.', 'BRK-B': 'Berkshire Hathaway Inc.', 'SPY': 'SPDR S&P 500 ETF',
        'QQQ': 'Invesco QQQ Trust', 'IWM': 'iShares Russell 2000 ETF', 'VTI': 'Vanguard Total Stock Market ETF'
    }
    
    # Search by ticker or company name
    for ticker, name in extended_stock_list.items():
        if (query in ticker or 
            query in name.upper() or
            any(word in name.upper() for word in query.split())):
            if len(results) < max_results:
                results.append({'ticker': ticker, 'name': name})
    
    return results

def get_portfolio_template_stocks():
    """Get predefined portfolio templates for comparison."""
    return {
        "🚀 Tech Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX"],
        "🏛️ Dow Jones Leaders": ["AAPL", "MSFT", "JNJ", "V", "PG", "JPM", "UNH", "HD", "MCD", "DIS"],
        "📊 S&P 500 Top 10": ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "TSLA", "META", "BRK-B", "UNH", "JNJ"],
        "💰 Dividend Aristocrats": ["JNJ", "PG", "KO", "PEP", "WMT", "MCD", "HD", "V", "MA", "MSFT"],
        "🌱 ESG Leaders": ["MSFT", "GOOGL", "AAPL", "JNJ", "PG", "UNH", "V", "MA", "NVDA", "ADBE"],
        "🏦 Financial Sector": ["JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "V", "MA"],
        "⚡ Energy Sector": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "BKR"],
        "🏥 Healthcare Focus": ["JNJ", "UNH", "PFE", "ABBV", "TMO", "ABT", "LLY", "DHR", "BMY", "MRK"],
        "🏭 Industrial Leaders": ["BA", "CAT", "GE", "MMM", "HON", "UPS", "RTX", "LMT", "DE", "FDX"],
        "🛒 Consumer Staples": ["PG", "KO", "PEP", "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "COST"],
        "📡 Technology ETFs": ["QQQ", "XLK", "VGT", "FTEC"],
        "🏦 Financial ETFs": ["XLF", "VFH", "KBE", "FNCL"],
        "🏥 Healthcare ETFs": ["XLV", "VHT", "IHI", "FHLC"]
    }

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

menu = ["Chart", "Trade Statistics", "Detailed Trades", "Compare Methods", "Portfolio Analysis", "Portfolio Comparison"]
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

elif choice == "Portfolio Analysis":
    st.title("Portfolio Analysis (Max 20 Stocks)")
    
    # Portfolio input section
    st.subheader("Portfolio Configuration")
    
    # Popular stock suggestions organized by sectors
    stock_suggestions = {
        "Technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "CRM", "ORCL", "ADBE", "INTC"],
        "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "TMO", "ABT", "LLY", "DHR", "BMY", "MRK"],
        "Finance": ["JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SPGI", "V", "MA"],
        "Consumer": ["PG", "KO", "PEP", "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "COST"],
        "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "BKR"],
        "Industrial": ["BA", "CAT", "GE", "MMM", "HON", "UPS", "RTX", "LMT", "DE", "FDX"]
    }
    
    # Initialize session state for selected stocks
    if 'selected_stocks' not in st.session_state:
        st.session_state.selected_stocks = []
    
    # Stock selection method
    selection_method = st.radio(
        "How would you like to select stocks?",
        ["Browse by Sector", "Search & Add Individual Stocks", "Use Template Portfolio", "Manual Input"],
        horizontal=True
    )
    
    if selection_method == "Browse by Sector":
        st.write("**Select stocks by browsing different sectors:**")
        
        # Create expandable sections for each sector
        for sector, stocks in stock_suggestions.items():
            with st.expander(f"📈 {sector} Sector", expanded=False):
                selected_from_sector = st.multiselect(
                    f"Select {sector} stocks:",
                    options=stocks,
                    key=f"sector_{sector}"
                )
                if selected_from_sector:
                    for stock in selected_from_sector:
                        if stock not in st.session_state.selected_stocks and len(st.session_state.selected_stocks) < 20:
                            st.session_state.selected_stocks.append(stock)
    
    elif selection_method == "Search & Add Individual Stocks":
        st.write("**Search and add individual stocks (supports ALL US stocks):**")
        
        # Enhanced search functionality for any US stock
        search_term = st.text_input(
            "🔍 Search ANY US stock by ticker or company name:", 
            placeholder="e.g., AAPL, Apple, Microsoft, Tesla, AMD, etc.",
            help="Search by ticker symbol (AAPL) or company name (Apple). Supports all US-listed stocks!"
        )
        
        if search_term and len(search_term) >= 2:
            with st.spinner("Searching stocks..."):
                search_results = search_stocks_by_name_or_ticker(search_term, max_results=15)
                
                if search_results:
                    st.write(f"Found {len(search_results)} matches:")
                    for i in range(0, len(search_results), 3):  # Display in rows of 3 for better readability
                        cols = st.columns(3)
                        for j, result in enumerate(search_results[i:i+3]):
                            if j < len(cols):
                                ticker = result['ticker']
                                name = result['name']
                                # Truncate long company names
                                display_name = name[:30] + "..." if len(name) > 30 else name
                                
                                if cols[j].button(f"➕ {ticker}\n{display_name}", key=f"search_add_{ticker}_{i}_{j}"):
                                    if ticker not in st.session_state.selected_stocks and len(st.session_state.selected_stocks) < 20:
                                        st.session_state.selected_stocks.append(ticker)
                                        st.success(f"Added {ticker} ({name}) to portfolio!")
                                    elif len(st.session_state.selected_stocks) >= 20:
                                        st.warning("Maximum 20 stocks allowed!")
                                    else:
                                        st.info(f"{ticker} is already in your portfolio!")
                else:
                    st.info("No matches found. Try a different search term or enter the exact ticker symbol.")
        
        # Direct ticker entry
        st.markdown("---")
        st.write("**Or enter ticker symbol directly:**")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            direct_ticker = st.text_input(
                "Enter ticker symbol:", 
                placeholder="e.g., AAPL, MSFT, TSLA",
                key="direct_ticker_input"
            )
        
        with col2:
            if st.button("🔍 Validate & Add", key="validate_add"):
                if direct_ticker:
                    ticker = direct_ticker.upper().strip()
                    if ticker not in st.session_state.selected_stocks:
                        if len(st.session_state.selected_stocks) < 20:
                            with st.spinner(f"Validating {ticker}..."):
                                is_valid, company_name = validate_ticker_symbol(ticker)
                                if is_valid:
                                    st.session_state.selected_stocks.append(ticker)
                                    st.success(f"Added {ticker} ({company_name}) to portfolio!")
                                else:
                                    st.error(f"'{ticker}' is not a valid US stock ticker symbol. Please check the symbol and try again.")
                        else:
                            st.warning("Maximum 20 stocks allowed!")
                    else:
                        st.info(f"{ticker} is already in your portfolio!")
        
        # Popular stocks quick add
        st.markdown("---")
        st.write("**Quick Add Popular Stocks:**")
        popular_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
        cols = st.columns(4)
        for i, stock in enumerate(popular_stocks):
            if cols[i % 4].button(f"➕ {stock}", key=f"popular_{stock}"):
                if stock not in st.session_state.selected_stocks and len(st.session_state.selected_stocks) < 20:
                    st.session_state.selected_stocks.append(stock)
                    st.success(f"Added {stock} to portfolio!")
                elif stock in st.session_state.selected_stocks:
                    st.info(f"{stock} already in portfolio!")
        
        # Information about supported stocks
        st.info("💡 **Supports ALL US-listed stocks** including NYSE, NASDAQ, and major ETFs. Search by company name or ticker symbol!")
    
    elif selection_method == "Use Template Portfolio":
        st.write("**Choose from pre-built portfolio templates:**")
        
        template_portfolios = {
            "🚀 Tech Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX"],
            "🏛️ Dow Jones Leaders": ["AAPL", "MSFT", "JNJ", "V", "PG", "JPM", "UNH", "HD", "MCD", "DIS"],
            "📊 S&P 500 Top 10": ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "TSLA", "META", "BRK-B", "UNH", "JNJ"],
            "💰 Dividend Aristocrats": ["JNJ", "PG", "KO", "PEP", "WMT", "MCD", "HD", "V", "MA", "MSFT"],
            "🌱 ESG Leaders": ["MSFT", "GOOGL", "AAPL", "JNJ", "PG", "UNH", "V", "MA", "NVDA", "ADBE"],
            "🏦 Financial Sector": ["JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "V", "MA"]
        }
        
        selected_template = st.selectbox("Select a template portfolio:", options=list(template_portfolios.keys()))
        
        if st.button(f"📋 Load {selected_template}"):
            st.session_state.selected_stocks = template_portfolios[selected_template].copy()
            st.success(f"Loaded {len(st.session_state.selected_stocks)} stocks from {selected_template}!")
    
    elif selection_method == "Manual Input":
        st.write("**Enter stock tickers manually (with validation):**")
        ticker_input = st.text_area(
            "Enter stock tickers (comma-separated, max 20):",
            value=",".join(st.session_state.selected_stocks) if st.session_state.selected_stocks else "AAPL,MSFT,GOOGL,AMZN,TSLA",
            height=100,
            help="Enter valid US stock ticker symbols separated by commas. Examples: AAPL,MSFT,GOOGL,AMZN,TSLA,SPY,QQQ"
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("📝 Update Portfolio (with validation)", type="primary"):
                manual_tickers = [ticker.strip().upper() for ticker in ticker_input.split(',') if ticker.strip()]
                if len(manual_tickers) > 20:
                    st.warning("Maximum 20 stocks allowed. Using first 20 tickers.")
                    manual_tickers = manual_tickers[:20]
                
                if manual_tickers:
                    with st.spinner("Validating ticker symbols..."):
                        valid_tickers = []
                        invalid_tickers = []
                        
                        for ticker in manual_tickers:
                            is_valid, company_name = validate_ticker_symbol(ticker)
                            if is_valid:
                                valid_tickers.append(ticker)
                            else:
                                invalid_tickers.append(ticker)
                        
                        if valid_tickers:
                            st.session_state.selected_stocks = valid_tickers
                            st.success(f"✅ Updated portfolio with {len(valid_tickers)} valid stocks: {', '.join(valid_tickers)}")
                        
                        if invalid_tickers:
                            st.error(f"❌ Invalid ticker symbols: {', '.join(invalid_tickers)}")
                            st.info("Please check the spelling and ensure they are valid US stock symbols.")
        
        with col2:
            if st.button("⚡ Quick Update (no validation)"):
                manual_tickers = [ticker.strip().upper() for ticker in ticker_input.split(',') if ticker.strip()]
                if len(manual_tickers) > 20:
                    st.warning("Maximum 20 stocks allowed. Using first 20 tickers.")
                    manual_tickers = manual_tickers[:20]
                st.session_state.selected_stocks = manual_tickers
                st.success(f"Updated portfolio with {len(manual_tickers)} stocks (no validation performed)!")
        
        # Common ticker examples
        st.markdown("---")
        st.write("**💡 Common Stock Examples:**")
        example_categories = {
            "Mega Cap": "AAPL,MSFT,GOOGL,AMZN,TSLA,META,NVDA,BRK-B",
            "Finance": "JPM,BAC,WFC,GS,V,MA,PYPL,SQ",
            "Healthcare": "JNJ,UNH,PFE,MRNA,ABBV,TMO,ABT",
            "Tech Growth": "NVDA,AMD,CRM,ADBE,NFLX,UBER,SNAP",
            "ETFs": "SPY,QQQ,VTI,IWM,XLK,XLF,XLV,XLE"
        }
        
        cols = st.columns(len(example_categories))
        for i, (category, tickers) in enumerate(example_categories.items()):
            if cols[i].button(f"📋 {category}", key=f"example_{category}"):
                current_text = ticker_input if ticker_input.strip() else ""
                new_tickers = tickers
                updated_text = f"{current_text},{new_tickers}" if current_text else new_tickers
                st.session_state.manual_input_suggestion = updated_text
                st.info(f"Added {category} examples to input field!")
    
    # Display current portfolio
    st.markdown("---")
    st.subheader("📊 Current Portfolio")
    
    if st.session_state.selected_stocks:
        # Display selected stocks in a nice format
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**Selected Stocks ({len(st.session_state.selected_stocks)}/20):**")
            # Display stocks in rows of 5
            for i in range(0, len(st.session_state.selected_stocks), 5):
                cols = st.columns(5)
                for j, stock in enumerate(st.session_state.selected_stocks[i:i+5]):
                    if j < len(cols):
                        cols[j].write(f"• {stock}")
        
        with col2:
            if st.button("🗑️ Clear All"):
                st.session_state.selected_stocks = []
                st.success("Portfolio cleared!")
        
        # Individual stock removal
        if len(st.session_state.selected_stocks) > 0:
            stock_to_remove = st.selectbox("Remove a stock:", options=["Select..."] + st.session_state.selected_stocks)
            if stock_to_remove != "Select..." and st.button(f"❌ Remove {stock_to_remove}"):
                st.session_state.selected_stocks.remove(stock_to_remove)
                st.success(f"Removed {stock_to_remove} from portfolio!")
        
        portfolio_tickers = st.session_state.selected_stocks
    else:
        st.info("👆 Please select stocks using one of the methods above.")
        portfolio_tickers = []
    
    # Analysis configuration (only show if stocks are selected)
    if portfolio_tickers:
        st.markdown("---")
        st.subheader("⚙️ Analysis Configuration")
        
        # Portfolio period selection
        col1, col2, col3 = st.columns(3)
        with col1:
            portfolio_period = st.selectbox("Analysis Period", options=["3mo", "6mo", "1y", "2y", "5y"], index=2)
        with col2:
            portfolio_interval = st.selectbox("Data Interval", options=["1d", "1wk"], index=0)
        with col3:
            auto_analyze = st.checkbox("Auto-analyze on changes", value=False, help="Automatically run analysis when portfolio changes")
        
        # Analysis buttons
        col1, col2 = st.columns(2)
        with col1:
            run_portfolio = st.button("🚀 Analyze Portfolio", type="primary", use_container_width=True)
        with col2:
            if st.button("💾 Save Portfolio", use_container_width=True):
                st.session_state.saved_portfolio = portfolio_tickers.copy()
                st.success("Portfolio saved! You can restore it later.")
        
        # Show saved portfolio option
        if 'saved_portfolio' in st.session_state and st.session_state.saved_portfolio:
            if st.button("📂 Load Saved Portfolio"):
                st.session_state.selected_stocks = st.session_state.saved_portfolio.copy()
                st.success("Loaded saved portfolio!")
                portfolio_tickers = st.session_state.selected_stocks
    
    if portfolio_tickers and (run_portfolio or auto_analyze):
        with st.spinner("Fetching portfolio data..."):
            # Get portfolio data
            portfolio_data = get_portfolio_data(portfolio_tickers, period=portfolio_period, interval=portfolio_interval)
            
            if portfolio_data:
                # Calculate metrics
                metrics, portfolio_returns = calculate_portfolio_metrics(portfolio_data)
                correlation_matrix = calculate_portfolio_correlation(portfolio_returns)
                portfolio_metrics = calculate_equal_weight_portfolio(portfolio_returns)
                
                # Quick portfolio overview at the top
                if portfolio_metrics:
                    st.markdown("---")
                    st.subheader("📈 Portfolio Overview")
                    
                    # Key metrics in prominent display
                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        total_return = portfolio_metrics['Total Return (%)']
                        delta_color = "normal" if total_return >= 0 else "inverse"
                        st.metric("Portfolio Return", f"{total_return}%", delta=f"{total_return}%")
                    
                    with metric_cols[1]:
                        volatility = portfolio_metrics['Volatility (%)']
                        st.metric("Volatility (Risk)", f"{volatility}%")
                    
                    with metric_cols[2]:
                        sharpe = portfolio_metrics['Sharpe Ratio']
                        st.metric("Sharpe Ratio", f"{sharpe}")
                    
                    with metric_cols[3]:
                        max_dd = portfolio_metrics['Max Drawdown (%)']
                        st.metric("Max Drawdown", f"{max_dd}%")
                    
                    # Risk level indicator
                    if volatility < 15:
                        risk_level = "🟢 Low Risk"
                        risk_color = "success"
                    elif volatility < 25:
                        risk_level = "🟡 Medium Risk"
                        risk_color = "warning"
                    else:
                        risk_level = "🔴 High Risk"
                        risk_color = "error"
                    
                    st.markdown(f"**Risk Assessment:** :{risk_color}[{risk_level}]")
                
                # Display results in tabs
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance Chart", "📋 Individual Metrics", "🔗 Correlation Analysis", "💼 Portfolio Summary"])
                
                with tab1:
                    st.subheader("Normalized Performance Comparison")
                    if portfolio_data:
                        fig_perf = plot_portfolio_performance(portfolio_data)
                        st.pyplot(fig_perf)
                        
                        # Performance summary table
                        st.subheader("Performance Summary")
                        summary_data = []
                        for ticker, metric in metrics.items():
                            summary_data.append({
                                'Ticker': ticker,
                                'Total Return (%)': metric['Total Return (%)'],
                                'Current Price': metric['Current Price'],
                                'Volatility (%)': metric['Volatility (%)'],
                                'Sharpe Ratio': metric['Sharpe Ratio']
                            })
                        st.dataframe(pd.DataFrame(summary_data))
                
                with tab2:
                    st.subheader("Individual Stock Metrics")
                    if metrics:
                        # Convert to DataFrame for better display
                        metrics_df = pd.DataFrame(metrics).T
                        st.dataframe(metrics_df.style.format({
                            'Total Return (%)': '{:.2f}%',
                            'Volatility (%)': '{:.2f}%',
                            'Sharpe Ratio': '{:.3f}',
                            'Max Drawdown (%)': '{:.2f}%',
                            'Current Price': '${:.2f}',
                            'Start Price': '${:.2f}'
                        }))
                        
                        # Best and worst performers
                        col1, col2 = st.columns(2)
                        with col1:
                            best_performer = metrics_df['Total Return (%)'].idxmax()
                            best_return = metrics_df.loc[best_performer, 'Total Return (%)']
                            st.success(f"**Best Performer**: {best_performer} ({best_return:.2f}%)")
                        
                        with col2:
                            worst_performer = metrics_df['Total Return (%)'].idxmin()
                            worst_return = metrics_df.loc[worst_performer, 'Total Return (%)']
                            st.error(f"**Worst Performer**: {worst_performer} ({worst_return:.2f}%)")
                
                with tab3:
                    st.subheader("Correlation Analysis")
                    if not correlation_matrix.empty:
                        fig_corr = plot_correlation_heatmap(correlation_matrix)
                        st.pyplot(fig_corr)
                        
                        st.write("**Interpretation:**")
                        st.write("- Values close to 1.0 indicate strong positive correlation")
                        st.write("- Values close to -1.0 indicate strong negative correlation") 
                        st.write("- Values close to 0.0 indicate little to no correlation")
                        
                        # Show highest and lowest correlations
                        if len(correlation_matrix) > 1:
                            # Get upper triangle of correlation matrix (excluding diagonal)
                            mask = np.triu(np.ones_like(correlation_matrix), k=1).astype(bool)
                            corr_pairs = correlation_matrix.where(mask).stack().reset_index()
                            corr_pairs.columns = ['Stock1', 'Stock2', 'Correlation']
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                highest_corr = corr_pairs.loc[corr_pairs['Correlation'].idxmax()]
                                st.info(f"**Highest Correlation**: {highest_corr['Stock1']} & {highest_corr['Stock2']} ({highest_corr['Correlation']:.3f})")
                            
                            with col2:
                                lowest_corr = corr_pairs.loc[corr_pairs['Correlation'].idxmin()]
                                st.info(f"**Lowest Correlation**: {lowest_corr['Stock1']} & {lowest_corr['Stock2']} ({lowest_corr['Correlation']:.3f})")
                
                with tab4:
                    st.subheader("Equal-Weighted Portfolio Performance")
                    if portfolio_metrics:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Return", f"{portfolio_metrics['Total Return (%)']}%")
                        with col2:
                            st.metric("Volatility", f"{portfolio_metrics['Volatility (%)']}%")
                        with col3:
                            st.metric("Sharpe Ratio", f"{portfolio_metrics['Sharpe Ratio']}")
                        with col4:
                            st.metric("Max Drawdown", f"{portfolio_metrics['Max Drawdown (%)']}%")
                        
                        # Portfolio composition
                        st.subheader("Portfolio Composition")
                        if len(portfolio_tickers) > 0:
                            weight_per_stock = 100 / len(portfolio_tickers)
                            composition_data = []
                            for ticker in portfolio_tickers:
                                if ticker in metrics:
                                    composition_data.append({
                                        'Ticker': ticker,
                                        'Weight (%)': round(weight_per_stock, 2),
                                        'Current Price': metrics[ticker]['Current Price'],
                                        'Total Return (%)': metrics[ticker]['Total Return (%)']
                                    })
                            
                            if composition_data:
                                st.dataframe(pd.DataFrame(composition_data))
                        
                        # Risk analysis
                        st.subheader("Risk Analysis")
                        if len(metrics) > 1:
                            volatilities = [metric['Volatility (%)'] for metric in metrics.values()]
                            avg_volatility = np.mean(volatilities)
                            
                            st.write(f"**Average Individual Volatility**: {avg_volatility:.2f}%")
                            st.write(f"**Portfolio Volatility**: {portfolio_metrics['Volatility (%)']}%")
                            
                            diversification_benefit = avg_volatility - portfolio_metrics['Volatility (%)']
                            if diversification_benefit > 0:
                                st.success(f"**Diversification Benefit**: {diversification_benefit:.2f}% reduction in volatility")
                            else:
                                st.warning("Portfolio shows limited diversification benefit")
            else:
                st.error("No valid portfolio data could be retrieved. Please check ticker symbols and try again.")
    
    # Help and tips section
    if not portfolio_tickers:
        st.markdown("---")
        with st.expander("💡 Portfolio Analysis Tips & Help", expanded=True):
            st.markdown("""
            ### How to Use Portfolio Analysis:
            
            **1. Select Stocks (ALL US STOCKS SUPPORTED):**
            - **Browse by Sector**: Explore popular stocks organized by industry sectors
            - **Search & Add**: Search ANY US stock by ticker or company name
            - **Use Templates**: Choose from pre-built portfolio templates
            - **Manual Input**: Type ticker symbols directly (with validation)
            
            **🌟 Stock Coverage:**
            - ✅ **NYSE**: All New York Stock Exchange listed companies
            - ✅ **NASDAQ**: All NASDAQ listed companies  
            - ✅ **ETFs**: Exchange-traded funds (SPY, QQQ, VTI, etc.)
            - ✅ **Small Cap**: Russell 2000 and smaller companies
            - ✅ **Large Cap**: S&P 500 and Fortune 500 companies
            - ✅ **Sectors**: Technology, Healthcare, Finance, Energy, Consumer, Industrial, etc.
            
            **2. Key Metrics Explained:**
            - **Total Return**: Overall percentage gain/loss over the selected period
            - **Volatility**: Measure of price fluctuation (higher = more risky)
            - **Sharpe Ratio**: Risk-adjusted return (higher = better risk/reward)
            - **Max Drawdown**: Largest peak-to-trough decline (lower = better)
            - **Correlation**: How stocks move relative to each other (-1 to +1)
            
            **3. Portfolio Benefits:**
            - **Diversification**: Spreading risk across different stocks/sectors
            - **Risk Reduction**: Lower portfolio volatility than individual stocks
            - **Performance Comparison**: See which stocks are outperforming
            
            **4. Tips for Better Portfolios:**
            - Mix stocks from different sectors and market caps
            - Look for low correlations between stocks
            - Balance high-growth and stable dividend stocks
            - Include both individual stocks and ETFs
            - Consider international exposure via ADRs
            
            **🔍 Search Examples:**
            - By Ticker: AAPL, MSFT, TSLA, AMD, ROKU
            - By Company: Apple, Microsoft, Tesla, Netflix
            - ETFs: SPY, QQQ, VTI, IWM, XLK, XLF, XLV
            - Small Caps: Any Russell 2000 component
            - Crypto-related: COIN, MSTR, RIOT, MARA
            
            **Popular Stock Categories:**
            - 🖥️ **Technology**: AAPL, MSFT, GOOGL, NVDA, TSLA, AMD, CRM
            - 🏥 **Healthcare**: JNJ, UNH, PFE, ABBV, TMO, GILD, MRNA
            - 🏦 **Finance**: JPM, BAC, V, MA, GS, BRK-B, PYPL
            - 🛒 **Consumer**: PG, KO, WMT, HD, MCD, AMZN, DIS
            - ⚡ **Energy**: XOM, CVX, COP, EOG, SLB, NEE
            - 🏭 **Industrial**: BA, CAT, GE, UPS, FDX, MMM
            """)
            
            st.info("💡 **Pro Tip**: Start with 5-10 stocks from different sectors, then analyze their correlation to build a well-diversified portfolio!")

elif choice == "Portfolio Comparison":
    st.title("📊 Portfolio Comparison Tool")
    st.write("Compare the performance of two different portfolios side by side")
    
    # Initialize session states for both portfolios
    if 'portfolio1_stocks' not in st.session_state:
        st.session_state.portfolio1_stocks = []
    if 'portfolio2_stocks' not in st.session_state:
        st.session_state.portfolio2_stocks = []
    
    # Portfolio templates
    template_portfolios = get_portfolio_template_stocks()
    
    # Two columns for portfolio configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔵 Portfolio 1 Configuration")
        
        # Portfolio 1 name
        portfolio1_name = st.text_input("Portfolio 1 Name:", value="Portfolio A", key="p1_name")
        
        # Portfolio 1 selection method
        p1_method = st.selectbox(
            "Selection Method for Portfolio 1:",
            ["Template", "Custom Input", "Load from Current Session"],
            key="p1_method"
        )
        
        if p1_method == "Template":
            p1_template = st.selectbox("Choose Portfolio 1 Template:", list(template_portfolios.keys()), key="p1_template")
            if st.button("📋 Load Template to Portfolio 1", key="load_p1"):
                st.session_state.portfolio1_stocks = template_portfolios[p1_template].copy()
                st.success(f"Loaded {len(st.session_state.portfolio1_stocks)} stocks to Portfolio 1!")
        
        elif p1_method == "Custom Input":
            p1_input = st.text_area(
                "Enter Portfolio 1 tickers (comma-separated):",
                value=",".join(st.session_state.portfolio1_stocks),
                height=100,
                key="p1_input"
            )
            if st.button("📝 Update Portfolio 1", key="update_p1"):
                stocks = [ticker.strip().upper() for ticker in p1_input.split(',') if ticker.strip()]
                if len(stocks) > 20:
                    st.warning("Maximum 20 stocks allowed. Using first 20.")
                    stocks = stocks[:20]
                st.session_state.portfolio1_stocks = stocks
                st.success(f"Updated Portfolio 1 with {len(stocks)} stocks!")
        
        elif p1_method == "Load from Current Session":
            if 'selected_stocks' in st.session_state and st.session_state.selected_stocks:
                if st.button("📂 Load from Portfolio Analysis", key="load_session_p1"):
                    st.session_state.portfolio1_stocks = st.session_state.selected_stocks.copy()
                    st.success(f"Loaded {len(st.session_state.portfolio1_stocks)} stocks from current session!")
            else:
                st.info("No stocks in current session. Go to Portfolio Analysis first.")
        
        # Display Portfolio 1 stocks
        if st.session_state.portfolio1_stocks:
            st.write(f"**Portfolio 1 Stocks ({len(st.session_state.portfolio1_stocks)}):**")
            for i in range(0, len(st.session_state.portfolio1_stocks), 3):
                cols = st.columns(3)
                for j, stock in enumerate(st.session_state.portfolio1_stocks[i:i+3]):
                    if j < len(cols):
                        cols[j].write(f"• {stock}")
            
            if st.button("🗑️ Clear Portfolio 1", key="clear_p1"):
                st.session_state.portfolio1_stocks = []
                st.success("Portfolio 1 cleared!")
    
    with col2:
        st.subheader("🔴 Portfolio 2 Configuration")
        
        # Portfolio 2 name
        portfolio2_name = st.text_input("Portfolio 2 Name:", value="Portfolio B", key="p2_name")
        
        # Portfolio 2 selection method
        p2_method = st.selectbox(
            "Selection Method for Portfolio 2:",
            ["Template", "Custom Input", "Load from Current Session"],
            key="p2_method"
        )
        
        if p2_method == "Template":
            p2_template = st.selectbox("Choose Portfolio 2 Template:", list(template_portfolios.keys()), key="p2_template")
            if st.button("📋 Load Template to Portfolio 2", key="load_p2"):
                st.session_state.portfolio2_stocks = template_portfolios[p2_template].copy()
                st.success(f"Loaded {len(st.session_state.portfolio2_stocks)} stocks to Portfolio 2!")
        
        elif p2_method == "Custom Input":
            p2_input = st.text_area(
                "Enter Portfolio 2 tickers (comma-separated):",
                value=",".join(st.session_state.portfolio2_stocks),
                height=100,
                key="p2_input"
            )
            if st.button("📝 Update Portfolio 2", key="update_p2"):
                stocks = [ticker.strip().upper() for ticker in p2_input.split(',') if ticker.strip()]
                if len(stocks) > 20:
                    st.warning("Maximum 20 stocks allowed. Using first 20.")
                    stocks = stocks[:20]
                st.session_state.portfolio2_stocks = stocks
                st.success(f"Updated Portfolio 2 with {len(stocks)} stocks!")
        
        elif p2_method == "Load from Current Session":
            if 'selected_stocks' in st.session_state and st.session_state.selected_stocks:
                if st.button("📂 Load from Portfolio Analysis", key="load_session_p2"):
                    st.session_state.portfolio2_stocks = st.session_state.selected_stocks.copy()
                    st.success(f"Loaded {len(st.session_state.portfolio2_stocks)} stocks from current session!")
            else:
                st.info("No stocks in current session. Go to Portfolio Analysis first.")
        
        # Display Portfolio 2 stocks
        if st.session_state.portfolio2_stocks:
            st.write(f"**Portfolio 2 Stocks ({len(st.session_state.portfolio2_stocks)}):**")
            for i in range(0, len(st.session_state.portfolio2_stocks), 3):
                cols = st.columns(3)
                for j, stock in enumerate(st.session_state.portfolio2_stocks[i:i+3]):
                    if j < len(cols):
                        cols[j].write(f"• {stock}")
            
            if st.button("🗑️ Clear Portfolio 2", key="clear_p2"):
                st.session_state.portfolio2_stocks = []
                st.success("Portfolio 2 cleared!")
    
    # Comparison configuration
    if st.session_state.portfolio1_stocks and st.session_state.portfolio2_stocks:
        st.markdown("---")
        st.subheader("⚙️ Comparison Configuration")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            comp_period = st.selectbox("Comparison Period", options=["3mo", "6mo", "1y", "2y", "5y"], index=2, key="comp_period")
        with col2:
            comp_interval = st.selectbox("Data Interval", options=["1d", "1wk"], index=0, key="comp_interval")
        with col3:
            comparison_type = st.selectbox("Comparison Type", options=["Side by Side", "Overlay", "Both"], index=2, key="comp_type")
        
        # Run comparison
        if st.button("🚀 Compare Portfolios", type="primary"):
            with st.spinner("Fetching data and analyzing portfolios..."):
                # Get data for both portfolios
                portfolio1_data = get_portfolio_data(st.session_state.portfolio1_stocks, period=comp_period, interval=comp_interval)
                portfolio2_data = get_portfolio_data(st.session_state.portfolio2_stocks, period=comp_period, interval=comp_interval)
                
                if portfolio1_data and portfolio2_data:
                    # Calculate metrics for both portfolios
                    metrics1, returns1 = calculate_portfolio_metrics(portfolio1_data)
                    metrics2, returns2 = calculate_portfolio_metrics(portfolio2_data)
                    
                    portfolio_metrics1 = calculate_equal_weight_portfolio(returns1)
                    portfolio_metrics2 = calculate_equal_weight_portfolio(returns2)
                    
                    # Portfolio overview comparison
                    st.markdown("---")
                    st.subheader("📈 Portfolio Comparison Overview")
                    
                    # Side-by-side metrics
                    if portfolio_metrics1 and portfolio_metrics2:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"### 🔵 {portfolio1_name}")
                            subcol1, subcol2 = st.columns(2)
                            with subcol1:
                                total_return1 = portfolio_metrics1['Total Return (%)']
                                st.metric("Total Return", f"{total_return1}%")
                                sharpe1 = portfolio_metrics1['Sharpe Ratio']
                                st.metric("Sharpe Ratio", f"{sharpe1}")
                            with subcol2:
                                volatility1 = portfolio_metrics1['Volatility (%)']
                                st.metric("Volatility", f"{volatility1}%")
                                max_dd1 = portfolio_metrics1['Max Drawdown (%)']
                                st.metric("Max Drawdown", f"{max_dd1}%")
                        
                        with col2:
                            st.markdown(f"### 🔴 {portfolio2_name}")
                            subcol1, subcol2 = st.columns(2)
                            with subcol1:
                                total_return2 = portfolio_metrics2['Total Return (%)']
                                delta_return = total_return2 - total_return1
                                st.metric("Total Return", f"{total_return2}%", delta=f"{delta_return:.2f}%")
                                sharpe2 = portfolio_metrics2['Sharpe Ratio']
                                delta_sharpe = sharpe2 - sharpe1
                                st.metric("Sharpe Ratio", f"{sharpe2}", delta=f"{delta_sharpe:.3f}")
                            with subcol2:
                                volatility2 = portfolio_metrics2['Volatility (%)']
                                delta_vol = volatility2 - volatility1
                                st.metric("Volatility", f"{volatility2}%", delta=f"{delta_vol:.2f}%")
                                max_dd2 = portfolio_metrics2['Max Drawdown (%)']
                                delta_dd = max_dd2 - max_dd1
                                st.metric("Max Drawdown", f"{max_dd2}%", delta=f"{delta_dd:.2f}%")
                        
                        # Winner analysis
                        st.markdown("### 🏆 Performance Winner Analysis")
                        winners = []
                        if total_return1 > total_return2:
                            winners.append(f"📈 **Best Total Return**: {portfolio1_name} ({total_return1:.2f}% vs {total_return2:.2f}%)")
                        else:
                            winners.append(f"📈 **Best Total Return**: {portfolio2_name} ({total_return2:.2f}% vs {total_return1:.2f}%)")
                        
                        if volatility1 < volatility2:
                            winners.append(f"🛡️ **Lower Risk**: {portfolio1_name} ({volatility1:.2f}% vs {volatility2:.2f}%)")
                        else:
                            winners.append(f"🛡️ **Lower Risk**: {portfolio2_name} ({volatility2:.2f}% vs {volatility1:.2f}%)")
                        
                        if sharpe1 > sharpe2:
                            winners.append(f"⚖️ **Better Risk-Adjusted Return**: {portfolio1_name} (Sharpe: {sharpe1:.3f} vs {sharpe2:.3f})")
                        else:
                            winners.append(f"⚖️ **Better Risk-Adjusted Return**: {portfolio2_name} (Sharpe: {sharpe2:.3f} vs {sharpe1:.3f})")
                        
                        for winner in winners:
                            st.success(winner)
                    
                    # Detailed comparison tabs
                    tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance Charts", "📋 Detailed Metrics", "📈 Individual Stocks", "🔍 Portfolio Composition"])
                    
                    with tab1:
                        if comparison_type in ["Side by Side", "Both"]:
                            st.subheader("Side-by-Side Performance Comparison")
                            fig_side = plot_portfolio_comparison(
                                portfolio1_data, portfolio2_data, 
                                portfolio1_name, portfolio2_name
                            )
                            st.pyplot(fig_side)
                        
                        if comparison_type in ["Overlay", "Both"]:
                            st.subheader("Portfolio Overlay Comparison")
                            fig_overlay = plot_portfolio_overlay_comparison(
                                returns1, returns2, portfolio1_name, portfolio2_name
                            )
                            st.pyplot(fig_overlay)
                    
                    with tab2:
                        st.subheader("Detailed Metrics Comparison")
                        if portfolio_metrics1 and portfolio_metrics2:
                            comparison_table = create_comparison_metrics_table(
                                portfolio_metrics1, portfolio_metrics2, 
                                portfolio1_name, portfolio2_name
                            )
                            st.dataframe(comparison_table, use_container_width=True)
                    
                    with tab3:
                        st.subheader("Individual Stock Performance")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"#### {portfolio1_name} Individual Stocks")
                            if metrics1:
                                df1 = pd.DataFrame(metrics1).T[['Total Return (%)', 'Volatility (%)', 'Sharpe Ratio']]
                                st.dataframe(df1.style.format({
                                    'Total Return (%)': '{:.2f}%',
                                    'Volatility (%)': '{:.2f}%',
                                    'Sharpe Ratio': '{:.3f}'
                                }))
                        
                        with col2:
                            st.markdown(f"#### {portfolio2_name} Individual Stocks")
                            if metrics2:
                                df2 = pd.DataFrame(metrics2).T[['Total Return (%)', 'Volatility (%)', 'Sharpe Ratio']]
                                st.dataframe(df2.style.format({
                                    'Total Return (%)': '{:.2f}%',
                                    'Volatility (%)': '{:.2f}%',
                                    'Sharpe Ratio': '{:.3f}'
                                }))
                    
                    with tab4:
                        st.subheader("Portfolio Composition Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"#### {portfolio1_name} Composition")
                            st.write(f"**Number of stocks**: {len(st.session_state.portfolio1_stocks)}")
                            st.write(f"**Stocks**: {', '.join(st.session_state.portfolio1_stocks)}")
                            
                            # Sector analysis (simplified)
                            tech_stocks1 = [s for s in st.session_state.portfolio1_stocks if s in ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX"]]
                            if tech_stocks1:
                                st.write(f"**Tech exposure**: {len(tech_stocks1)} stocks ({len(tech_stocks1)/len(st.session_state.portfolio1_stocks)*100:.1f}%)")
                        
                        with col2:
                            st.markdown(f"#### {portfolio2_name} Composition")
                            st.write(f"**Number of stocks**: {len(st.session_state.portfolio2_stocks)}")
                            st.write(f"**Stocks**: {', '.join(st.session_state.portfolio2_stocks)}")
                            
                            # Sector analysis (simplified)
                            tech_stocks2 = [s for s in st.session_state.portfolio2_stocks if s in ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX"]]
                            if tech_stocks2:
                                st.write(f"**Tech exposure**: {len(tech_stocks2)} stocks ({len(tech_stocks2)/len(st.session_state.portfolio2_stocks)*100:.1f}%)")
                        
                        # Common stocks analysis
                        common_stocks = set(st.session_state.portfolio1_stocks) & set(st.session_state.portfolio2_stocks)
                        if common_stocks:
                            st.markdown("#### 🔄 Common Stocks")
                            st.write(f"**Overlapping stocks**: {', '.join(sorted(common_stocks))}")
                            overlap_pct = len(common_stocks) / max(len(st.session_state.portfolio1_stocks), len(st.session_state.portfolio2_stocks)) * 100
                            st.write(f"**Portfolio overlap**: {overlap_pct:.1f}%")
                        else:
                            st.info("No common stocks between portfolios - completely different strategies!")
                
                else:
                    st.error("Failed to fetch data for one or both portfolios. Please check ticker symbols.")
    
    # Quick comparison templates
    elif not st.session_state.portfolio1_stocks or not st.session_state.portfolio2_stocks:
        st.markdown("---")
        with st.expander("🚀 Quick Start: Popular Portfolio Comparisons", expanded=True):
            st.markdown("""
            ### Popular Comparison Ideas:
            
            **🔥 Growth vs Value:**
            - Portfolio 1: Tech Giants (AAPL, MSFT, GOOGL, NVDA, TSLA)
            - Portfolio 2: Dividend Aristocrats (JNJ, PG, KO, WMT, HD)
            
            **🌍 Sector Diversification:**
            - Portfolio 1: Technology Focus
            - Portfolio 2: Healthcare Focus
            
            **📊 Index Comparison:**
            - Portfolio 1: S&P 500 Top 10
            - Portfolio 2: Dow Jones Leaders
            
            **🎯 Risk Comparison:**
            - Portfolio 1: High Growth/High Risk
            - Portfolio 2: Stable/Low Risk
            """)
            
            # Quick load buttons
            st.markdown("#### Quick Load Examples:")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Load Tech vs Healthcare Example"):
                    st.session_state.portfolio1_stocks = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "META", "NFLX"]
                    st.session_state.portfolio2_stocks = ["JNJ", "UNH", "PFE", "ABBV", "TMO", "ABT", "LLY"]
                    st.success("Loaded Tech vs Healthcare example!")
            
            with col2:
                if st.button("Load Growth vs Dividend Example"):
                    st.session_state.portfolio1_stocks = ["TSLA", "NVDA", "META", "NFLX", "AMZN", "GOOGL"]
                    st.session_state.portfolio2_stocks = ["JNJ", "PG", "KO", "WMT", "HD", "V", "MA"]
                    st.success("Loaded Growth vs Dividend example!")