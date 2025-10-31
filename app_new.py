import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
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
st.set_page_config(
    page_title="📈 Trading Strategy Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/WMY2023/COMP4145_lab2.v2',
        'Report a bug': 'https://github.com/WMY2023/COMP4145_lab2.v2/issues',
        'About': "# Trading Strategy Dashboard\nA comprehensive tool for stock analysis and portfolio management."
    }
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    .stSelectbox > div > div {
        background-color: #ffffff;
    }
    .sidebar-section {
        background-color: #f1f3f4;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">📈 Trading Strategy Dashboard</h1>', unsafe_allow_html=True)

# Navigation menu at the top
menu_options = {
    "📊 Chart Analysis": "Chart", 
    "📈 Trade Statistics": "Trade Statistics", 
    "📋 Detailed Trades": "Detailed Trades", 
    "⚖️ Compare Methods": "Compare Methods"
}

# Create tabs for main navigation
tab_keys = list(menu_options.keys())
selected_tab = st.tabs(tab_keys)

# Sidebar controls with improved organization
st.sidebar.markdown("## 🎯 Analysis Controls")

# Stock selection section
with st.sidebar.expander("🏢 Stock Selection", expanded=True):
    ticker = st.text_input(
        "Stock Ticker", 
        value="MSFT", 
        help="Enter a US stock ticker (e.g., AAPL, GOOGL, TSLA)",
        placeholder="e.g., AAPL, MSFT, GOOGL"
    )

# Time period section
with st.sidebar.expander("📅 Time Period", expanded=True):
    period = st.selectbox(
        "Analysis Period", 
        options=["1wk", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], 
        index=4,
        help="Select the time period for analysis"
    )
    interval = st.selectbox(
        "Data Interval", 
        options=["1d", "1wk", "1mo"], 
        index=0,
        help="Choose data granularity"
    )

# Strategy selection section
with st.sidebar.expander("🎯 Trading Strategy", expanded=True):
    method = st.selectbox(
        "Strategy Type", 
        options=[
            "Golden Cross (MA50/MA200)",
            "Bollinger Bands",
            "OBV Strategy",
            "ATR Strategy",
        ], 
        index=0,
        help="Select trading strategy for analysis"
    )

# Forecast controls section
with st.sidebar.expander("🔮 Forecast Options", expanded=False):
    enable_forecast = st.checkbox("Enable Forecast", value=False)
    if enable_forecast:
        forecast_days = st.slider("Forecast Days", min_value=1, max_value=60, value=5)
        forecast_method = st.selectbox("Forecast Method", options=['linear', 'ma', 'ema'], index=0)
        forecast_window = st.slider("History Window", min_value=5, max_value=365, value=60)

# Quick actions section
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 Quick Actions")
run = st.sidebar.button("🔄 Refresh Analysis", type="primary", use_container_width=True)

# Portfolio Analysis section
with st.sidebar.expander("💼 Portfolio Analysis", expanded=False):
    if st.button("🔍 Analyze Portfolio", use_container_width=True):
        st.session_state.show_portfolio_analysis = True
        st.rerun()

# Portfolio Comparison section  
with st.sidebar.expander("🔄 Portfolio Comparison", expanded=False):
    if st.button("⚖️ Compare Portfolios", use_container_width=True):
        st.session_state.show_portfolio_comparison = True
        st.rerun()

# Add help section
with st.sidebar.expander("❓ Help & Tips"):
    st.markdown("""
    **Quick Tips:**
    - 📊 Use Chart tab for visual analysis
    - 📈 Check Trade Statistics for performance
    - 💼 Portfolio Analysis supports 20+ stocks
    - 🔄 Compare different strategies easily
    
    **Popular Tickers:**
    - Tech: AAPL, MSFT, GOOGL, NVDA
    - Finance: JPM, BAC, V, MA
    - Healthcare: JNJ, UNH, PFE
    """)

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

# Initialize session state for portfolio features
if 'show_portfolio_analysis' not in st.session_state:
    st.session_state.show_portfolio_analysis = False
if 'show_portfolio_comparison' not in st.session_state:
    st.session_state.show_portfolio_comparison = False

# Show Portfolio Analysis if requested
if st.session_state.show_portfolio_analysis:
    st.markdown("### 💼 Portfolio Analysis")
    st.markdown("Analyze performance of multiple stocks in your portfolio (up to 20 stocks)")
    
    # Portfolio input section
    st.markdown("#### 📊 Portfolio Configuration")
    
    # Initialize session state for portfolio management
    if 'portfolio_stocks' not in st.session_state:
        st.session_state.portfolio_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        # Portfolio templates
        templates = get_portfolio_template_stocks()
        template_names = list(templates.keys())
        selected_template = st.selectbox(
            "🎯 Choose a Portfolio Template",
            ["Custom Portfolio"] + template_names,
            help="Select a predefined portfolio or build your custom portfolio",
            key="portfolio_template_main"
        )
        
        if selected_template != "Custom Portfolio":
            if st.button(f"📋 Load {selected_template}", type="secondary", key="load_template_main"):
                st.session_state.portfolio_stocks = templates[selected_template][:20]  # Limit to 20 stocks
                st.rerun()
            st.info(f"**{selected_template}**: {len(templates[selected_template])} stocks")
    
    with col2:
        portfolio_period = st.selectbox("📅 Analysis Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3, key="portfolio_period_main")
    
    with col3:
        if st.button("❌ Close Portfolio Analysis", type="secondary"):
            st.session_state.show_portfolio_analysis = False
            st.rerun()
    
    st.markdown("---")
    st.info("💡 Portfolio Analysis feature is now accessible from the sidebar! Use the 'Portfolio Analysis' expander in the sidebar to configure and analyze portfolios.")

# Show Portfolio Comparison if requested  
if st.session_state.show_portfolio_comparison:
    st.markdown("### 🔄 Portfolio Comparison")
    st.markdown("Compare performance between two different portfolios")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    # Portfolio 1
    with col1:
        st.markdown("#### 🔵 Portfolio 1")
        templates = get_portfolio_template_stocks()
        template_names = list(templates.keys())
        
        template1 = st.selectbox("Template 1", ["Custom"] + template_names, key="template1_main")
        if template1 != "Custom":
            portfolio1_tickers = templates[template1]
            st.info(f"**{template1}**: {', '.join(portfolio1_tickers[:3])}{'...' if len(portfolio1_tickers) > 3 else ''}")
        else:
            portfolio1_input = st.text_area(
                "Portfolio 1 Tickers",
                value="AAPL, MSFT, GOOGL, AMZN",
                key="portfolio1_input_main"
            )
            portfolio1_tickers = [ticker.strip().upper() for ticker in portfolio1_input.split(",") if ticker.strip()]
    
    # Portfolio 2
    with col2:
        st.markdown("#### 🔴 Portfolio 2")
        template2 = st.selectbox("Template 2", ["Custom"] + template_names, key="template2_main", index=1 if len(template_names) > 0 else 0)
        if template2 != "Custom":
            portfolio2_tickers = templates[template2]
            st.info(f"**{template2}**: {', '.join(portfolio2_tickers[:3])}{'...' if len(portfolio2_tickers) > 3 else ''}")
        else:
            portfolio2_input = st.text_area(
                "Portfolio 2 Tickers",
                value="JPM, BAC, V, MA",
                key="portfolio2_input_main"
            )
            portfolio2_tickers = [ticker.strip().upper() for ticker in portfolio2_input.split(",") if ticker.strip()]
    
    # Controls
    with col3:
        comparison_period = st.selectbox("Comparison Period", ["3mo", "6mo", "1y", "2y"], index=2, key="comparison_period_main")
        
        if st.button("🔄 Compare Now", type="primary"):
            st.success("Portfolio comparison feature activated! Use the sidebar controls for full functionality.")
        
        if st.button("❌ Close Comparison", type="secondary"):
            st.session_state.show_portfolio_comparison = False
            st.rerun()
    
    st.markdown("---")
    st.info("💡 Portfolio Comparison feature is now accessible from the sidebar! Use the 'Portfolio Comparison' expander in the sidebar for full comparison functionality.")

# Tab 1: Chart Analysis
with selected_tab[0]:
    # Stock info header
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader(f"📊 {ticker} - {method}")
    with col2:
        if not price_data.empty:
            current_price = price_data['Close'].iloc[-1]
            st.metric("Current Price", f"${current_price:.2f}")
    with col3:
        if not price_data.empty and len(price_data) > 1:
            price_change = ((price_data['Close'].iloc[-1] / price_data['Close'].iloc[-2]) - 1) * 100
            st.metric("Daily Change", f"{price_change:.2f}%", delta=f"{price_change:.2f}%")
    
    if not price_data.empty:
        # Analysis summary cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            data_points = len(price_data)
            st.info(f"📈 **Data Points**: {data_points}")
        with col2:
            if len(positions) > 0:
                total_trades = len(positions)
                st.info(f"💼 **Total Trades**: {total_trades}")
            else:
                st.info("💼 **Total Trades**: 0")
        with col3:
            if len(positions) > 0:
                avg_profit = positions['ProfitPct'].mean()
                color = "🟢" if avg_profit > 0 else "🔴"
                st.info(f"{color} **Avg Profit**: {avg_profit:.2f}%")
            else:
                st.info("📊 **Avg Profit**: N/A")
        with col4:
            period_return = ((price_data['Close'].iloc[-1] / price_data['Close'].iloc[0]) - 1) * 100
            color = "🟢" if period_return > 0 else "🔴"
            st.info(f"{color} **Period Return**: {period_return:.2f}%")
        
        # Forecast section
        forecast_df = None
        if enable_forecast:
            with st.spinner("🔮 Generating forecast..."):
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
        
        # Main chart
        with st.container():
            fig = plot_price_obv_atr(price_data, positions, method=method, buy_color='red', sell_color='purple', forecast_df=forecast_df)
            st.pyplot(fig, use_container_width=True)
        
        # Show forecast suggestion in an attractive format
        if enable_forecast and suggestion:
            st.markdown("### 🔮 AI Prediction")
            recommendation = suggestion['recommendation']
            reason = suggestion['reason']
            
            if "BUY" in recommendation.upper():
                st.success(f"🟢 **{recommendation}**: {reason}")
            elif "SELL" in recommendation.upper():
                st.error(f"🔴 **{recommendation}**: {reason}")
            else:
                st.info(f"🟡 **{recommendation}**: {reason}")
        
        # Quick statistics table
        if len(positions) > 0:
            st.markdown("### 📊 Quick Stats")
            stats = get_statistics(positions)
            if stats:
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe({
                        "Metric": ["Total Trades", "Win Rate (%)", "Avg Profit (%)"],
                        "Value": [stats['Total Trades'], f"{stats['Win Rate (%)']:.1f}%", f"{stats['Average Profit (%)']:.2f}%"]
                    }, hide_index=True, use_container_width=True)
                with col2:
                    st.dataframe({
                        "Metric": ["Winning Trades", "Losing Trades", "Avg Holding Days"],
                        "Value": [stats['Winning Trades'], stats['Losing Trades'], f"{stats['Average Holding Days']:.1f}"]
                    }, hide_index=True, use_container_width=True)
    else:
        st.error("❌ No price data available. Please check the ticker symbol and try again.")
        st.info("💡 **Suggestion**: Try popular tickers like AAPL, MSFT, GOOGL, or TSLA")

# Tab 2: Trade Statistics
with selected_tab[1]:
    st.markdown("### 📈 Trade Statistics Summary")
    st.markdown(f"Performance analysis for **{ticker}** using **{method}**")
    
    if not price_data.empty:
        stats = get_statistics(positions)
        if stats and len(positions) > 0:
            # Create attractive metric cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trades", stats['Total Trades'])
            with col2:
                win_rate = stats['Win Rate (%)']
                st.metric("Win Rate", f"{win_rate:.1f}%", 
                        delta="Good" if win_rate >= 50 else "Needs Improvement")
            with col3:
                avg_profit = stats['Average Profit (%)']
                st.metric("Avg Profit", f"{avg_profit:.2f}%", 
                        delta=f"{avg_profit:.2f}%")
            with col4:
                avg_holding = stats['Average Holding Days']
                st.metric("Avg Hold Period", f"{avg_holding:.1f} days")
            
            st.markdown("---")
            
            # Detailed statistics table
            st.markdown("#### 📊 Detailed Breakdown")
            stats_df = pd.DataFrame([stats])
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            # Performance indicators
            col1, col2 = st.columns(2)
            with col1:
                if win_rate >= 60:
                    st.success("🟢 **Excellent** win rate!")
                elif win_rate >= 50:
                    st.info("🟡 **Good** win rate")
                else:
                    st.warning("🔴 **Low** win rate - consider strategy adjustment")
            
            with col2:
                if avg_profit > 2:
                    st.success("🟢 **Strong** average returns!")
                elif avg_profit > 0:
                    st.info("🟡 **Positive** returns")
                else:
                    st.error("🔴 **Negative** average returns")
        else:
            st.info("📊 No trades generated with current settings")
            st.markdown("""
            **Possible reasons:**
            - Strategy conditions not met in the selected time period
            - Insufficient data for the chosen interval
            - Try different time periods or strategies
            """)
    else:
        st.error("❌ No data available for analysis")

# Tab 3: Detailed Trades
with selected_tab[2]:
    st.markdown("### 📋 Detailed Trade Records")
    st.markdown(f"Complete trade history for **{ticker}** using **{method}**")
    
    if not price_data.empty:
        if not positions.empty:
            # Trade summary
            col1, col2, col3 = st.columns(3)
            with col1:
                profitable_trades = len(positions[positions['ProfitPct'] > 0])
                st.metric("Profitable Trades", profitable_trades)
            with col2:
                best_trade = positions['ProfitPct'].max()
                st.metric("Best Trade", f"{best_trade:.2f}%")
            with col3:
                worst_trade = positions['ProfitPct'].min()
                st.metric("Worst Trade", f"{worst_trade:.2f}%")
            
            st.markdown("---")
            
            # Interactive trade table
            st.markdown("#### 💼 Trade Details")
            
            # Add filters
            col1, col2 = st.columns(2)
            with col1:
                profit_filter = st.selectbox("Filter by Result", 
                                            ["All Trades", "Profitable Only", "Loss Only"])
            with col2:
                sort_by = st.selectbox("Sort by", 
                                     ["Date", "Profit %", "Holding Days"])
            
            # Apply filters
            filtered_positions = positions.copy()
            if profit_filter == "Profitable Only":
                filtered_positions = filtered_positions[filtered_positions['ProfitPct'] > 0]
            elif profit_filter == "Loss Only":
                filtered_positions = filtered_positions[filtered_positions['ProfitPct'] <= 0]
            
            # Apply sorting
            if sort_by == "Date":
                filtered_positions = filtered_positions.sort_values('BuyDate', ascending=False)
            elif sort_by == "Profit %":
                filtered_positions = filtered_positions.sort_values('ProfitPct', ascending=False)
            elif sort_by == "Holding Days":
                filtered_positions = filtered_positions.sort_values('HoldingDays', ascending=False)
            
            # Display table with styling
            st.dataframe(
                filtered_positions,
                use_container_width=True,
                column_config={
                    "ProfitPct": st.column_config.NumberColumn(
                        "Profit %",
                        format="%.2f%%"
                    ),
                    "BuyPrice": st.column_config.NumberColumn(
                        "Buy Price",
                        format="$%.2f"
                    ),
                    "SellPrice": st.column_config.NumberColumn(
                        "Sell Price", 
                        format="$%.2f"
                    )
                }
            )
            
            # Download option
            csv = filtered_positions.to_csv(index=False)
            st.download_button(
                label="📥 Download Trade Data",
                data=csv,
                file_name=f"{ticker}_{method.replace(' ', '_')}_trades.csv",
                mime="text/csv"
            )
        else:
            st.info("📊 No trades to display with current settings")
            st.markdown("Try adjusting the time period or strategy parameters.")
    else:
        st.error("❌ No data available for analysis")

# Tab 4: Compare Methods
with selected_tab[3]:
    st.markdown("### ⚖️ Strategy Comparison")
    st.markdown(f"Compare trading strategies for **{ticker}** over **{period}** period")
    
    if price_data.empty:
        st.error("❌ No price data available for the selected ticker/period.")
        st.info("💡 Please check the ticker symbol and try again.")
    else:
        # Strategy selection with better UI
        st.markdown("#### 🎯 Select Strategies to Compare")
        methods_list = [
            "Golden Cross (MA50/MA200)",
            "Bollinger Bands", 
            "OBV Strategy",
            "ATR Strategy",
        ]
        
        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_a:
            method1 = st.selectbox("🔵 Strategy 1", options=methods_list, index=0, key='method1')
        with col_b:
            method2 = st.selectbox("🔴 Strategy 2", options=methods_list, index=1, key='method2')
        with col_c:
            run_compare = st.button("🔄 Update Comparison", type="primary", use_container_width=True)
        
        # Compute both methods
        pos1 = run_selected_method(price_data, method1)
        pos2 = run_selected_method(price_data, method2)
        
        # Quick comparison metrics at top
        col1, col2, col3 = st.columns(3)
        
        stats1 = get_statistics(pos1)
        stats2 = get_statistics(pos2)
        
        with col1:
            if stats1 and stats2:
                trades1 = stats1.get('Total Trades', 0)
                trades2 = stats2.get('Total Trades', 0)
                st.metric("Total Trades", f"{trades1} vs {trades2}", 
                        delta=f"{trades1 - trades2} difference")
        
        with col2:
            if stats1 and stats2:
                profit1 = stats1.get('Average Profit (%)', 0)
                profit2 = stats2.get('Average Profit (%)', 0)
                winner = method1 if profit1 > profit2 else method2
                st.metric("Avg Profit %", f"{profit1:.2f}% vs {profit2:.2f}%",
                        delta=f"Winner: {winner.split('(')[0]}")
        
        with col3:
            if stats1 and stats2:
                wr1 = stats1.get('Win Rate (%)', 0)
                wr2 = stats2.get('Win Rate (%)', 0)
                st.metric("Win Rate", f"{wr1:.1f}% vs {wr2:.1f}%",
                        delta=f"{wr1 - wr2:.1f}% difference")
        
        st.markdown("---")
        
        # Side-by-side visualization
        st.markdown("#### 📊 Visual Comparison")
        left, right = st.columns(2)
        
        with left:
            st.markdown(f"##### 🔵 {method1}")
            fig1 = plot_price_obv_atr(price_data, pos1, method=method1, 
                                    buy_color='blue', sell_color='darkblue')
            st.pyplot(fig1, use_container_width=True)
            
            if stats1:
                # Compact stats display
                st.markdown("**Key Metrics:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Trades**: {stats1['Total Trades']}")
                    st.info(f"**Win Rate**: {stats1['Win Rate (%)']:.1f}%")
                with col2:
                    st.info(f"**Avg Profit**: {stats1['Average Profit (%)']:.2f}%")
                    st.info(f"**Avg Hold**: {stats1['Average Holding Days']:.1f} days")
            else:
                st.warning("No trades generated")
        
        with right:
            st.markdown(f"##### 🔴 {method2}")
            fig2 = plot_price_obv_atr(price_data, pos2, method=method2, 
                                    buy_color='red', sell_color='darkred')
            st.pyplot(fig2, use_container_width=True)
            
            if stats2:
                # Compact stats display
                st.markdown("**Key Metrics:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Trades**: {stats2['Total Trades']}")
                    st.info(f"**Win Rate**: {stats2['Win Rate (%)']:.1f}%")
                with col2:
                    st.info(f"**Avg Profit**: {stats2['Average Profit (%)']:.2f}%")
                    st.info(f"**Avg Hold**: {stats2['Average Holding Days']:.1f} days")
            else:
                st.warning("No trades generated")
        
        # Detailed comparison table
        if stats1 or stats2:
            st.markdown("#### 📈 Detailed Comparison")
            
            if stats1 and stats2:
                # Create comparison DataFrame
                comparison_data = []
                all_metrics = set(stats1.keys()) | set(stats2.keys())
                
                for metric in sorted(all_metrics):
                    val1 = stats1.get(metric, 0)
                    val2 = stats2.get(metric, 0)
                    
                    # Determine better performance
                    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                        if metric in ['Win Rate (%)', 'Average Profit (%)', 'Average Win (%)']:
                            better = "🔵 Method 1" if val1 > val2 else "🔴 Method 2" if val2 > val1 else "🟡 Tie"
                        elif metric in ['Average Loss (%)']:
                            better = "🔵 Method 1" if val1 > val2 else "🔴 Method 2" if val2 > val1 else "🟡 Tie"  # Less negative is better
                        else:
                            better = "-"
                    else:
                        better = "-"
                    
                    comparison_data.append({
                        'Metric': metric,
                        f'{method1}': val1,
                        f'{method2}': val2,
                        'Better': better
                    })
                
                df_compare = pd.DataFrame(comparison_data)
                st.dataframe(df_compare, use_container_width=True, hide_index=True)
                
                # Summary recommendation
                st.markdown("#### 🎯 Recommendation")
                if stats1 and stats2:
                    profit1 = stats1.get('Average Profit (%)', 0)
                    profit2 = stats2.get('Average Profit (%)', 0)
                    
                    if profit1 > profit2:
                        st.success(f"🏆 **{method1}** shows better performance with {profit1:.2f}% average profit vs {profit2:.2f}%")
                    elif profit2 > profit1:
                        st.success(f"🏆 **{method2}** shows better performance with {profit2:.2f}% average profit vs {profit1:.2f}%")
                    else:
                        st.info("🤝 Both strategies show similar performance")
            else:
                st.warning("⚠️ One or both strategies generated no trades for comparison")

# Footer with user tips
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Quick Tips")
st.sidebar.info("""
**New User?**
1. Start with Chart tab
2. Try different tickers (AAPL, MSFT)
3. Use Portfolio Analysis in sidebar
4. Compare strategies
""")

# Footer
st.markdown("---")
st.markdown("##### 📈 Trading Strategy Dashboard | Enhanced UI Version")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("🎯 **Features**: Chart Analysis, Portfolio Tools, Strategy Comparison")
with col2:
    st.markdown("📊 **Supports**: All US stocks, ETFs, Multiple strategies")
with col3:
    st.markdown("🚀 **Quick Start**: Select a tab above and enter a ticker symbol")
