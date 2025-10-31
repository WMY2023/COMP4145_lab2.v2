import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
)

# Portfolio analysis functions
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

# Streamlit app configuration
st.set_page_config(
    page_title="💼 Portfolio Analysis Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/WMY2023/COMP4145_lab2.v2',
        'Report a bug': 'https://github.com/WMY2023/COMP4145_lab2.v2/issues',
        'About': "# Portfolio Analysis Dashboard\nAnalyze and optimize your investment portfolios."
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
    .portfolio-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">💼 Portfolio Analysis Dashboard</h1>', unsafe_allow_html=True)

# Navigation
col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
with col_nav1:
    if st.button("🏠 Back to Trading Dashboard", type="secondary", use_container_width=True):
        st.switch_page("app.py")
with col_nav2:
    st.info("📊 **Current Page**: Portfolio Analysis")
with col_nav3:
    if st.button("🔄 Go to Portfolio Comparison", type="primary", use_container_width=True):
        st.switch_page("pages/portfolio_comparison.py")

st.markdown("---")

# Initialize session state for portfolio management
if 'portfolio_stocks' not in st.session_state:
    st.session_state.portfolio_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Sidebar controls
st.sidebar.markdown("## 🎯 Portfolio Configuration")

# Template selection
with st.sidebar.expander("🎯 Portfolio Templates", expanded=True):
    templates = get_portfolio_template_stocks()
    template_names = list(templates.keys())
    selected_template = st.selectbox(
        "Choose a Portfolio Template",
        ["Custom Portfolio"] + template_names,
        help="Select a predefined portfolio or build your custom portfolio"
    )
    
    if selected_template != "Custom Portfolio":
        if st.button(f"📋 Load {selected_template}", type="primary"):
            st.session_state.portfolio_stocks = templates[selected_template][:20]  # Limit to 20 stocks
            st.success(f"✅ Loaded {selected_template}")
            st.rerun()
        st.info(f"**{selected_template}**: {len(templates[selected_template])} stocks")

# Analysis period
with st.sidebar.expander("📅 Analysis Period", expanded=True):
    portfolio_period = st.selectbox("Analysis Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)

# Main content area
st.markdown("### 📊 Portfolio Configuration")

# Custom portfolio input section
if selected_template == "Custom Portfolio":
    st.markdown("#### ✏️ Custom Portfolio Input")
    
    # Two methods: text input or search interface
    input_method = st.radio(
        "Input Method",
        ["📝 Manual Input", "🔍 Search & Add"],
        horizontal=True,
        help="Choose how you want to add stocks to your portfolio"
    )
    
    if input_method == "📝 Manual Input":
        portfolio_input = st.text_area(
            "Enter Stock Tickers (comma-separated)",
            value=", ".join(st.session_state.portfolio_stocks),
            height=120,
            help="Enter up to 20 stock tickers separated by commas. Example: AAPL, MSFT, GOOGL, AMZN"
        )
        
        if st.button("📋 Update Portfolio", type="primary"):
            new_tickers = [ticker.strip().upper() for ticker in portfolio_input.split(",") if ticker.strip()]
            if len(new_tickers) > 20:
                st.warning("⚠️ Portfolio limited to 20 stocks. Taking first 20.")
                new_tickers = new_tickers[:20]
            st.session_state.portfolio_stocks = new_tickers
            st.success(f"✅ Portfolio updated with {len(new_tickers)} stocks")
            st.rerun()
    
    else:  # Search & Add method
        col_search, col_add = st.columns([3, 1])
        with col_search:
            search_query = st.text_input(
                "🔍 Search for stocks",
                placeholder="Search by ticker or company name (e.g., Apple, AAPL)"
            )
        
        with col_add:
            if st.button("🔍 Search", type="primary"):
                if search_query:
                    search_results = search_stocks_by_name_or_ticker(search_query, max_results=10)
                    if search_results:
                        st.session_state.search_results = search_results
                    else:
                        st.warning("No stocks found matching your search.")
        
        # Display search results
        if hasattr(st.session_state, 'search_results') and st.session_state.search_results:
            st.markdown("**Search Results:**")
            for i, result in enumerate(st.session_state.search_results):
                col_info, col_action = st.columns([4, 1])
                with col_info:
                    st.write(f"**{result['ticker']}** - {result['name']}")
                with col_action:
                    if st.button(f"➕ Add", key=f"add_stock_{i}"):
                        if result['ticker'] not in st.session_state.portfolio_stocks:
                            if len(st.session_state.portfolio_stocks) < 20:
                                st.session_state.portfolio_stocks.append(result['ticker'])
                                st.success(f"✅ Added {result['ticker']} to portfolio")
                                st.rerun()
                            else:
                                st.warning("⚠️ Portfolio is full (20 stocks maximum)")
                        else:
                            st.info(f"ℹ️ {result['ticker']} already in portfolio")

# Current portfolio display
st.markdown("---")
st.markdown("### 📊 Current Portfolio")

if st.session_state.portfolio_stocks:
    # Portfolio overview
    portfolio_display = ", ".join(st.session_state.portfolio_stocks)
    st.markdown(f'<div class="portfolio-card"><h4>🎯 Portfolio Overview</h4><p><strong>Stocks ({len(st.session_state.portfolio_stocks)}):</strong> {portfolio_display}</p></div>', unsafe_allow_html=True)
    
    # Portfolio management
    col_manage1, col_manage2, col_manage3 = st.columns([2, 1, 1])
    
    with col_manage1:
        stock_to_remove = st.selectbox(
            "Remove a stock from portfolio",
            ["Select to remove..."] + st.session_state.portfolio_stocks
        )
    
    with col_manage2:
        if st.button("🗑️ Remove Stock", type="secondary"):
            if stock_to_remove != "Select to remove...":
                st.session_state.portfolio_stocks.remove(stock_to_remove)
                st.success(f"✅ Removed {stock_to_remove} from portfolio")
                st.rerun()
    
    with col_manage3:
        if st.button("🧹 Clear All", type="secondary"):
            st.session_state.portfolio_stocks = []
            st.success("✅ Portfolio cleared")
            st.rerun()
    
    st.markdown("---")
    
    # Analysis button
    if st.button("🚀 Run Portfolio Analysis", type="primary", use_container_width=True):
        if len(st.session_state.portfolio_stocks) > 0:
            with st.spinner("📊 Analyzing portfolio performance..."):
                # Get portfolio data
                portfolio_data = get_portfolio_data(st.session_state.portfolio_stocks, period=portfolio_period)
                
                if portfolio_data:
                    # Calculate metrics
                    individual_metrics, portfolio_returns = calculate_portfolio_metrics(portfolio_data)
                    correlation_matrix = calculate_portfolio_correlation(portfolio_returns)
                    portfolio_summary = calculate_equal_weight_portfolio(portfolio_returns)
                    
                    st.markdown("---")
                    st.markdown("## 📈 Portfolio Analysis Results")
                    
                    # Portfolio summary metrics
                    if portfolio_summary:
                        st.markdown("### 🎯 Equal-Weighted Portfolio Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            total_return = portfolio_summary['Total Return (%)']
                            st.metric("Total Return", f"{total_return:.2f}%", 
                                    delta="Positive" if total_return > 0 else "Negative")
                        with col2:
                            volatility = portfolio_summary['Volatility (%)']
                            st.metric("Volatility", f"{volatility:.2f}%")
                        with col3:
                            sharpe = portfolio_summary['Sharpe Ratio']
                            st.metric("Sharpe Ratio", f"{sharpe:.3f}",
                                    delta="Good" if sharpe > 1 else "Fair" if sharpe > 0.5 else "Poor")
                        with col4:
                            max_dd = portfolio_summary['Max Drawdown (%)']
                            st.metric("Max Drawdown", f"{max_dd:.2f}%")
                    
                    # Performance chart
                    st.markdown("### 📊 Portfolio Performance Chart")
                    perf_fig = plot_portfolio_performance(portfolio_data)
                    st.pyplot(perf_fig, use_container_width=True)
                    
                    # Individual stock metrics
                    st.markdown("### 📋 Individual Stock Performance")
                    if individual_metrics:
                        metrics_df = pd.DataFrame(individual_metrics).T
                        st.dataframe(metrics_df, use_container_width=True)
                        
                        # Top and bottom performers
                        col_top, col_bottom = st.columns(2)
                        
                        with col_top:
                            st.markdown("#### 🏆 Top Performers")
                            top_performers = metrics_df.nlargest(3, 'Total Return (%)')
                            for ticker in top_performers.index:
                                return_val = top_performers.loc[ticker, 'Total Return (%)']
                                st.success(f"**{ticker}**: {return_val:.2f}%")
                        
                        with col_bottom:
                            st.markdown("#### 📉 Bottom Performers")
                            bottom_performers = metrics_df.nsmallest(3, 'Total Return (%)')
                            for ticker in bottom_performers.index:
                                return_val = bottom_performers.loc[ticker, 'Total Return (%)']
                                st.error(f"**{ticker}**: {return_val:.2f}%")
                    
                    # Correlation analysis
                    if not correlation_matrix.empty and len(correlation_matrix) > 1:
                        st.markdown("### 🔗 Portfolio Correlation Analysis")
                        corr_fig = plot_correlation_heatmap(correlation_matrix)
                        st.pyplot(corr_fig, use_container_width=True)
                        
                        # Correlation insights
                        avg_correlation = correlation_matrix.values[correlation_matrix.values != 1.0].mean()
                        st.info(f"📊 **Average Portfolio Correlation**: {avg_correlation:.3f}")
                        
                        if avg_correlation > 0.7:
                            st.warning("⚠️ **High Correlation**: Your portfolio stocks are highly correlated. Consider diversifying across different sectors.")
                        elif avg_correlation > 0.4:
                            st.info("ℹ️ **Moderate Correlation**: Reasonable diversification, but room for improvement.")
                        else:
                            st.success("✅ **Low Correlation**: Well-diversified portfolio!")
                    
                    # Portfolio recommendations
                    st.markdown("### 💡 Portfolio Recommendations")
                    
                    col_rec1, col_rec2 = st.columns(2)
                    
                    with col_rec1:
                        st.markdown("#### 📊 Risk Assessment")
                        if portfolio_summary:
                            volatility = portfolio_summary['Volatility (%)']
                            sharpe = portfolio_summary['Sharpe Ratio']
                            
                            if volatility < 15:
                                st.success("🟢 **Low Risk**: Conservative portfolio with low volatility")
                            elif volatility < 25:
                                st.info("🟡 **Medium Risk**: Balanced risk-return profile")
                            else:
                                st.warning("🔴 **High Risk**: Aggressive portfolio with high volatility")
                    
                    with col_rec2:
                        st.markdown("#### 🎯 Performance Assessment")
                        if portfolio_summary:
                            total_return = portfolio_summary['Total Return (%)']
                            
                            if total_return > 15:
                                st.success("🏆 **Excellent Performance**: Strong returns achieved")
                            elif total_return > 5:
                                st.info("👍 **Good Performance**: Solid positive returns")
                            elif total_return > -5:
                                st.warning("😐 **Neutral Performance**: Mixed results")
                            else:
                                st.error("📉 **Poor Performance**: Consider portfolio rebalancing")
                    
                    # Download options
                    st.markdown("### 💾 Export Data")
                    col_export1, col_export2 = st.columns(2)
                    
                    with col_export1:
                        if individual_metrics:
                            metrics_csv = pd.DataFrame(individual_metrics).T.to_csv()
                            st.download_button(
                                "📥 Download Metrics",
                                data=metrics_csv,
                                file_name=f"portfolio_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                    
                    with col_export2:
                        if not correlation_matrix.empty:
                            corr_csv = correlation_matrix.to_csv()
                            st.download_button(
                                "📥 Download Correlations",
                                data=corr_csv,
                                file_name=f"portfolio_correlations_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                else:
                    st.error("❌ Unable to fetch data for the selected stocks. Please check the ticker symbols and try again.")
        else:
            st.warning("⚠️ Please add some stocks to your portfolio to run analysis.")
else:
    st.warning("⚠️ Please add some stocks to your portfolio to begin analysis.")
    st.markdown("""
    ### 🚀 Get Started
    
    1. **Choose a Template**: Select from our pre-built portfolios in the sidebar
    2. **Add Stocks Manually**: Enter ticker symbols separated by commas
    3. **Search & Add**: Use our search function to find specific stocks
    4. **Run Analysis**: Click the analysis button to see comprehensive results
    
    **Popular Portfolio Ideas:**
    - 🚀 Tech Giants: AAPL, MSFT, GOOGL, AMZN, META
    - 💰 Dividend Focus: JNJ, PG, KO, PEP, WMT
    - 🏦 Financial Sector: JPM, BAC, V, MA, GS
    """)

# Footer
st.markdown("---")
st.markdown("##### 💼 Portfolio Analysis Dashboard | Professional Investment Analysis Tool")
