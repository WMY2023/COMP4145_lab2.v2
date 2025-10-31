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

# Portfolio comparison functions
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
    page_title="🔄 Portfolio Comparison Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/WMY2023/COMP4145_lab2.v2',
        'Report a bug': 'https://github.com/WMY2023/COMP4145_lab2.v2/issues',
        'About': "# Portfolio Comparison Dashboard\nCompare and analyze multiple investment portfolios."
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
    .portfolio1-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .portfolio2-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .winner-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🔄 Portfolio Comparison Dashboard</h1>', unsafe_allow_html=True)

# Navigation
col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
with col_nav1:
    if st.button("🏠 Back to Trading Dashboard", type="secondary", use_container_width=True):
        st.switch_page("app.py")
with col_nav2:
    if st.button("💼 Go to Portfolio Analysis", type="primary", use_container_width=True):
        st.switch_page("portfolio_analysis.py")
with col_nav3:
    st.info("📊 **Current Page**: Portfolio Comparison")

st.markdown("---")

# Initialize comparison portfolios in session state
if 'comparison_portfolio1' not in st.session_state:
    st.session_state.comparison_portfolio1 = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
if 'comparison_portfolio2' not in st.session_state:
    st.session_state.comparison_portfolio2 = ['JPM', 'BAC', 'V', 'MA', 'WFC']

# Sidebar controls
st.sidebar.markdown("## ⚙️ Comparison Settings")

# Analysis period
with st.sidebar.expander("📅 Analysis Period", expanded=True):
    comparison_period = st.selectbox("Comparison Period", ["3mo", "6mo", "1y", "2y"], index=2)

# Templates
with st.sidebar.expander("🎯 Portfolio Templates", expanded=True):
    templates = get_portfolio_template_stocks()
    template_names = list(templates.keys())
    
    st.markdown("**Quick Load Templates:**")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        if st.button("🚀 Tech vs 🏦 Finance", type="secondary", use_container_width=True):
            st.session_state.comparison_portfolio1 = templates["🚀 Tech Giants"][:10]
            st.session_state.comparison_portfolio2 = templates["🏦 Financial Sector"][:10]
            st.success("✅ Loaded Tech vs Finance comparison")
            st.rerun()
    
    with col_t2:
        if st.button("📊 S&P vs 💰 Dividend", type="secondary", use_container_width=True):
            st.session_state.comparison_portfolio1 = templates["📊 S&P 500 Top 10"][:10]
            st.session_state.comparison_portfolio2 = templates["💰 Dividend Aristocrats"][:10]
            st.success("✅ Loaded S&P vs Dividend comparison")
            st.rerun()

# Main content
st.markdown("## 📊 Portfolio Configuration")

col1, col2 = st.columns(2)

# Portfolio 1 Configuration
with col1:
    st.markdown("### 🔵 Portfolio 1")
    
    template1 = st.selectbox("Template for Portfolio 1", ["Custom"] + template_names, key="template1")
    if template1 != "Custom":
        portfolio1_tickers = templates[template1][:15]  # Limit for comparison
        st.info(f"**{template1}**: {len(portfolio1_tickers)} stocks")
        if st.button("📋 Load Portfolio 1", key="load_p1", type="primary"):
            st.session_state.comparison_portfolio1 = portfolio1_tickers
            st.success("✅ Portfolio 1 loaded")
            st.rerun()
    else:
        portfolio1_input = st.text_area(
            "Portfolio 1 Tickers (comma-separated)",
            value=", ".join(st.session_state.comparison_portfolio1),
            height=120,
            key="portfolio1_input"
        )
        if st.button("📋 Update Portfolio 1", key="update_p1", type="primary"):
            new_tickers = [ticker.strip().upper() for ticker in portfolio1_input.split(",") if ticker.strip()]
            st.session_state.comparison_portfolio1 = new_tickers[:15]
            st.success(f"✅ Portfolio 1 updated ({len(st.session_state.comparison_portfolio1)} stocks)")
            st.rerun()
    
    # Display current portfolio 1
    st.markdown('<div class="portfolio1-card"><h4>🔵 Current Portfolio 1</h4><p>' + ", ".join(st.session_state.comparison_portfolio1) + '</p></div>', unsafe_allow_html=True)

# Portfolio 2 Configuration
with col2:
    st.markdown("### 🔴 Portfolio 2")
    
    template2 = st.selectbox("Template for Portfolio 2", ["Custom"] + template_names, key="template2", index=1 if len(template_names) > 0 else 0)
    if template2 != "Custom":
        portfolio2_tickers = templates[template2][:15]  # Limit for comparison
        st.info(f"**{template2}**: {len(portfolio2_tickers)} stocks")
        if st.button("📋 Load Portfolio 2", key="load_p2", type="primary"):
            st.session_state.comparison_portfolio2 = portfolio2_tickers
            st.success("✅ Portfolio 2 loaded")
            st.rerun()
    else:
        portfolio2_input = st.text_area(
            "Portfolio 2 Tickers (comma-separated)",
            value=", ".join(st.session_state.comparison_portfolio2),
            height=120,
            key="portfolio2_input"
        )
        if st.button("📋 Update Portfolio 2", key="update_p2", type="primary"):
            new_tickers = [ticker.strip().upper() for ticker in portfolio2_input.split(",") if ticker.strip()]
            st.session_state.comparison_portfolio2 = new_tickers[:15]
            st.success(f"✅ Portfolio 2 updated ({len(st.session_state.comparison_portfolio2)} stocks)")
            st.rerun()
    
    # Display current portfolio 2
    st.markdown('<div class="portfolio2-card"><h4>🔴 Current Portfolio 2</h4><p>' + ", ".join(st.session_state.comparison_portfolio2) + '</p></div>', unsafe_allow_html=True)

st.markdown("---")

# Run comparison button
if st.button("🔄 Run Portfolio Comparison", type="primary", use_container_width=True):
    if len(st.session_state.comparison_portfolio1) > 0 and len(st.session_state.comparison_portfolio2) > 0:
        with st.spinner("📊 Running portfolio comparison analysis..."):
            # Get data for both portfolios
            portfolio1_data = get_portfolio_data(st.session_state.comparison_portfolio1, period=comparison_period)
            portfolio2_data = get_portfolio_data(st.session_state.comparison_portfolio2, period=comparison_period)
            
            if portfolio1_data and portfolio2_data:
                # Calculate metrics for both portfolios
                metrics1, returns1 = calculate_portfolio_metrics(portfolio1_data)
                metrics2, returns2 = calculate_portfolio_metrics(portfolio2_data)
                
                portfolio1_summary = calculate_equal_weight_portfolio(returns1)
                portfolio2_summary = calculate_equal_weight_portfolio(returns2)
                
                st.markdown("---")
                st.markdown("## 📊 Portfolio Comparison Results")
                
                # Quick comparison overview
                if portfolio1_summary and portfolio2_summary:
                    col_over1, col_over2, col_over3 = st.columns(3)
                    
                    with col_over1:
                        p1_return = portfolio1_summary['Total Return (%)']
                        p2_return = portfolio2_summary['Total Return (%)']
                        winner = "Portfolio 1" if p1_return > p2_return else "Portfolio 2"
                        st.metric("Total Return Winner", winner, 
                                f"🔵 {p1_return:.2f}% vs 🔴 {p2_return:.2f}%")
                    
                    with col_over2:
                        p1_sharpe = portfolio1_summary['Sharpe Ratio']
                        p2_sharpe = portfolio2_summary['Sharpe Ratio']
                        risk_winner = "Portfolio 1" if p1_sharpe > p2_sharpe else "Portfolio 2"
                        st.metric("Risk-Adjusted Winner", risk_winner,
                                f"🔵 {p1_sharpe:.3f} vs 🔴 {p2_sharpe:.3f}")
                    
                    with col_over3:
                        p1_vol = portfolio1_summary['Volatility (%)']
                        p2_vol = portfolio2_summary['Volatility (%)']
                        stable = "Portfolio 1" if p1_vol < p2_vol else "Portfolio 2"
                        st.metric("More Stable", stable,
                                f"🔵 {p1_vol:.2f}% vs 🔴 {p2_vol:.2f}%")
                
                # Summary comparison table
                if portfolio1_summary and portfolio2_summary:
                    st.markdown("### 🎯 Performance Summary")
                    comparison_table = create_comparison_metrics_table(
                        portfolio1_summary, portfolio2_summary, 
                        "Portfolio 1 (🔵)", "Portfolio 2 (🔴)"
                    )
                    if not comparison_table.empty:
                        st.dataframe(comparison_table, use_container_width=True, hide_index=True)
                
                # Side-by-side performance charts
                st.markdown("### 📈 Performance Comparison Charts")
                comparison_fig = plot_portfolio_comparison(
                    portfolio1_data, portfolio2_data,
                    "Portfolio 1 (🔵)", "Portfolio 2 (🔴)"
                )
                st.pyplot(comparison_fig, use_container_width=True)
                
                # Overlay comparison
                st.markdown("### 🔄 Portfolio Overlay Comparison")
                overlay_fig = plot_portfolio_overlay_comparison(
                    returns1, returns2,
                    "Portfolio 1 (🔵)", "Portfolio 2 (🔴)"
                )
                st.pyplot(overlay_fig, use_container_width=True)
                
                # Detailed metrics comparison
                st.markdown("### 📋 Detailed Metrics Comparison")
                
                col_metrics1, col_metrics2 = st.columns(2)
                
                with col_metrics1:
                    st.markdown("#### 🔵 Portfolio 1 Metrics")
                    if portfolio1_summary:
                        for key, value in portfolio1_summary.items():
                            st.metric(key, f"{value}")
                    
                    if metrics1:
                        st.markdown("**Individual Stock Performance:**")
                        p1_df = pd.DataFrame(metrics1).T[['Total Return (%)', 'Volatility (%)', 'Sharpe Ratio']]
                        st.dataframe(p1_df, use_container_width=True)
                
                with col_metrics2:
                    st.markdown("#### 🔴 Portfolio 2 Metrics")
                    if portfolio2_summary:
                        for key, value in portfolio2_summary.items():
                            st.metric(key, f"{value}")
                    
                    if metrics2:
                        st.markdown("**Individual Stock Performance:**")
                        p2_df = pd.DataFrame(metrics2).T[['Total Return (%)', 'Volatility (%)', 'Sharpe Ratio']]
                        st.dataframe(p2_df, use_container_width=True)
                
                # Winner analysis
                st.markdown("### 🏆 Final Analysis")
                if portfolio1_summary and portfolio2_summary:
                    p1_return = portfolio1_summary['Total Return (%)']
                    p2_return = portfolio2_summary['Total Return (%)']
                    p1_sharpe = portfolio1_summary['Sharpe Ratio']
                    p2_sharpe = portfolio2_summary['Sharpe Ratio']
                    
                    # Determine overall winner
                    p1_score = 0
                    p2_score = 0
                    
                    if p1_return > p2_return:
                        p1_score += 1
                    else:
                        p2_score += 1
                    
                    if p1_sharpe > p2_sharpe:
                        p1_score += 1
                    else:
                        p2_score += 1
                    
                    if p1_score > p2_score:
                        winner_text = "🏆 Portfolio 1 (🔵) is the overall winner!"
                        winner_color = "portfolio1-card"
                    elif p2_score > p1_score:
                        winner_text = "🏆 Portfolio 2 (🔴) is the overall winner!"
                        winner_color = "portfolio2-card"
                    else:
                        winner_text = "🤝 It's a tie! Both portfolios show similar performance."
                        winner_color = "winner-card"
                    
                    st.markdown(f'<div class="{winner_color}"><h3>{winner_text}</h3></div>', unsafe_allow_html=True)
                    
                    col_winner1, col_winner2 = st.columns(2)
                    with col_winner1:
                        if p1_return > p2_return:
                            st.success(f"🏆 **Portfolio 1** wins on Total Return: {p1_return:.2f}% vs {p2_return:.2f}%")
                        elif p2_return > p1_return:
                            st.success(f"🏆 **Portfolio 2** wins on Total Return: {p2_return:.2f}% vs {p1_return:.2f}%")
                        else:
                            st.info("🤝 **Tie** on Total Return")
                    
                    with col_winner2:
                        if p1_sharpe > p2_sharpe:
                            st.success(f"🏆 **Portfolio 1** wins on Risk-Adjusted Return: {p1_sharpe:.3f} vs {p2_sharpe:.3f}")
                        elif p2_sharpe > p1_sharpe:
                            st.success(f"🏆 **Portfolio 2** wins on Risk-Adjusted Return: {p2_sharpe:.3f} vs {p1_sharpe:.3f}")
                        else:
                            st.info("🤝 **Tie** on Risk-Adjusted Return")
                
                # Investment recommendations
                st.markdown("### 💡 Investment Recommendations")
                
                col_rec1, col_rec2 = st.columns(2)
                
                with col_rec1:
                    st.markdown("#### 🎯 For Conservative Investors")
                    if portfolio1_summary and portfolio2_summary:
                        p1_vol = portfolio1_summary['Volatility (%)']
                        p2_vol = portfolio2_summary['Volatility (%)']
                        
                        if p1_vol < p2_vol:
                            st.info("🔵 **Portfolio 1** is more suitable - lower volatility")
                        else:
                            st.info("🔴 **Portfolio 2** is more suitable - lower volatility")
                
                with col_rec2:
                    st.markdown("#### 🚀 For Growth Investors")
                    if portfolio1_summary and portfolio2_summary:
                        if p1_return > p2_return:
                            st.info("🔵 **Portfolio 1** is more suitable - higher returns")
                        else:
                            st.info("🔴 **Portfolio 2** is more suitable - higher returns")
                
                # Export comparison results
                st.markdown("### 💾 Export Comparison Data")
                col_export1, col_export2 = st.columns(2)
                
                with col_export1:
                    if not comparison_table.empty:
                        comparison_csv = comparison_table.to_csv(index=False)
                        st.download_button(
                            "📥 Download Comparison Summary",
                            data=comparison_csv,
                            file_name=f"portfolio_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                
                with col_export2:
                    if metrics1 and metrics2:
                        combined_metrics = {
                            'Portfolio_1': pd.DataFrame(metrics1).T,
                            'Portfolio_2': pd.DataFrame(metrics2).T
                        }
                        # Create a combined CSV
                        combined_csv = ""
                        for name, df in combined_metrics.items():
                            combined_csv += f"\n{name}\n"
                            combined_csv += df.to_csv()
                        
                        st.download_button(
                            "📥 Download Detailed Metrics",
                            data=combined_csv,
                            file_name=f"portfolio_detailed_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
            else:
                st.error("❌ Unable to fetch data for one or both portfolios. Please check the ticker symbols.")
    else:
        st.warning("⚠️ Please ensure both portfolios have stocks before running comparison.")

else:
    st.markdown("""
    ### 🚀 Get Started with Portfolio Comparison
    
    **How to use this tool:**
    
    1. **Configure Portfolio 1** (Blue): Select from templates or enter custom tickers
    2. **Configure Portfolio 2** (Red): Choose a different portfolio to compare against
    3. **Set Analysis Period**: Choose how far back to analyze performance
    4. **Run Comparison**: Click the comparison button to see detailed results
    
    **Popular Comparison Ideas:**
    
    - 🚀 **Tech vs Finance**: Compare technology stocks against financial sector
    - 📊 **S&P 500 vs Dividend**: Large-cap growth vs dividend-focused investing
    - 🌱 **ESG vs Traditional**: Sustainable investing vs conventional approach
    - ⚡ **Energy vs Healthcare**: Sector rotation strategies
    
    **What You'll Get:**
    
    - 📈 Side-by-side performance charts
    - 🎯 Risk-return analysis and Sharpe ratios
    - 🏆 Clear winner identification
    - 💡 Investment recommendations based on your risk profile
    - 💾 Downloadable comparison reports
    """)

# Footer
st.markdown("---")
st.markdown("##### 🔄 Portfolio Comparison Dashboard | Professional Investment Comparison Tool")
