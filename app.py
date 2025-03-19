import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from io import BytesIO
import requests
import os
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Nexus Portfolio Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    .css-18e3th9 {
        padding-top: 2rem;
    }
    .css-1d391kg {
        padding-top: 3.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #262730;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4D34FF;
    }
    .stMetric {
        background-color: #1E2130;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #4D34FF;
    }
    .css-1v0mbdj {
        border-radius: 10px;
        padding: 10px;
        background-color: #1E2130;
    }
    .css-1r6slb0 {
        background-color: #262730;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .css-50ug3q {
        font-size: 16px;
        font-weight: 600;
    }
    .stAlert {
        background-color: #1E2130;
        color: white;
        border-radius: 10px;
    }
    .stButton > button {
        background-color: #4D34FF;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
    }
    .stButton > button:hover {
        background-color: #5D44FF;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.image("https://www.svgrepo.com/show/401150/chart.svg", width=80)
    st.title("Nexus Portfolio")
    st.markdown("---")
    page = st.radio("Navigation", ["Dashboard", "Portfolio Analysis", "Stock Screener", "Prediction Models", "Settings"])
    
    st.markdown("---")
    st.markdown("### Portfolio Information")
    
    # Sample portfolio data - in a real app, this would be stored and loaded from a database
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'NVDA'],
            'Shares': [10, 5, 3, 2, 8, 4],
            'Purchase Price': [150.75, 280.50, 3200.00, 2500.00, 220.30, 180.75],
            'Purchase Date': ['2021-01-15', '2021-02-20', '2021-03-10', '2021-04-05', '2021-06-15', '2021-05-10']
        })
    
    uploaded_file = st.file_uploader("Upload Portfolio CSV", type="csv")
    if uploaded_file is not None:
        portfolio_data = pd.read_csv(uploaded_file)
        st.session_state.portfolio = portfolio_data
        st.success("Portfolio uploaded successfully!")
    
    st.markdown("---")
    
    # Time period selector
    st.markdown("### Time Period")
    time_period = st.selectbox("Select Time Period", ["1 Month", "3 Months", "6 Months", "1 Year", "3 Years", "5 Years", "Max"])
    
    # Market indices to compare
    st.markdown("### Benchmark")
    benchmark = st.selectbox("Select Benchmark", ["S&P 500 (^GSPC)", "NASDAQ (^IXIC)", "Dow Jones (^DJI)"])
    
    st.markdown("---")
    st.markdown("### Theme")
    theme = st.selectbox("Select Theme", ["Dark", "Light"])

# Helper Functions
@st.cache_data(ttl=3600)
def get_stock_data(ticker, period="1y"):
    """Fetch stock data using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_portfolio_data(portfolio_df, period="1y"):
    """Get historical data for all stocks in portfolio"""
    all_data = {}
    for symbol in portfolio_df['Symbol']:
        all_data[symbol] = get_stock_data(symbol, period)
    return all_data

def calculate_portfolio_value(portfolio_df, all_stock_data, date=None):
    """Calculate the total portfolio value for a given date"""
    total_value = 0
    portfolio_values = {}
    
    for _, row in portfolio_df.iterrows():
        symbol = row['Symbol']
        shares = row['Shares']
        
        if date is None:
            # Use the latest price
            if symbol in all_stock_data and not all_stock_data[symbol].empty:
                price = all_stock_data[symbol]['Close'][-1]
                total_value += price * shares
                portfolio_values[symbol] = price * shares
        else:
            # Use the price on the given date
            if symbol in all_stock_data and not all_stock_data[symbol].empty:
                if date in all_stock_data[symbol].index:
                    price = all_stock_data[symbol].loc[date, 'Close']
                    total_value += price * shares
                    portfolio_values[symbol] = price * shares
    
    return total_value, portfolio_values

def calculate_portfolio_history(portfolio_df, all_stock_data):
    """Calculate daily portfolio value over time"""
    # Find common date range
    if not all([not data.empty for data in all_stock_data.values()]):
        return pd.DataFrame()  # Return empty DataFrame if any stock data is missing
        
    start_date = max([data.index[0] for data in all_stock_data.values() if not data.empty])
    end_date = min([data.index[-1] for data in all_stock_data.values() if not data.empty])
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    portfolio_history = []
    
    for date in dates:
        value, _ = calculate_portfolio_value(portfolio_df, all_stock_data, date)
        portfolio_history.append({'Date': date, 'Value': value})
    
    return pd.DataFrame(portfolio_history)

def get_period_days(period):
    """Convert period string to number of days"""
    if period == "1 Month":
        return "1mo"
    elif period == "3 Months":
        return "3mo"
    elif period == "6 Months":
        return "6mo"
    elif period == "1 Year":
        return "1y"
    elif period == "3 Years":
        return "3y"
    elif period == "5 Years":
        return "5y"
    else:  # Max
        return "max"

def calculate_roi(initial_value, final_value):
    """Calculate ROI percentage"""
    if initial_value == 0:
        return 0
    return ((final_value - initial_value) / initial_value) * 100

def predict_stock_price(stock_data, days=30):
    """Predict stock price using ARIMA model with error handling"""
    try:
        # Check if we have enough data points
        if len(stock_data) < 60:  # Require at least 60 data points for a decent prediction
            raise ValueError("Not enough historical data for prediction")
            
        # Handle non-stationary data with differencing
        close_values = stock_data['Close'].values
        
        # Check for NaN values and replace if necessary
        if np.isnan(close_values).any():
            close_values = pd.Series(close_values).interpolate().values
            
        # Use a simpler model with more robust parameters
        model = ARIMA(close_values, order=(2,1,0))  # Simpler model (p=2, d=1, q=0)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=days)
        
        # Ensure we don't have negative price predictions
        forecast = np.maximum(forecast, close_values[-1] * 0.5)  # Limit the downside to 50% of last price
        
        return forecast
    except Exception as e:
        # If the ARIMA model fails, fall back to a simple linear extrapolation
        try:
            # Use last 30 days to predict future trend
            days_back = min(30, len(stock_data)-1)
            recent_data = stock_data['Close'].values[-days_back:]
            
            # Create a simple trend line
            x = np.arange(days_back)
            y = recent_data
            
            # Linear regression
            slope, intercept = np.polyfit(x, y, 1)
            
            # Project forward
            future_x = np.arange(days_back, days_back + days)
            forecast = slope * future_x + intercept
            
            # Ensure we don't have negative price predictions
            forecast = np.maximum(forecast, stock_data['Close'].iloc[-1] * 0.5)
            
            return forecast
        except Exception as fallback_error:
            # If all else fails, raise the original error
            raise ValueError(f"Could not generate prediction: {str(e)}") from e

def calculate_beta(stock_returns, market_returns):
    """Calculate beta (volatility compared to market)"""
    try:
        covariance = np.cov(stock_returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        return covariance / market_variance if market_variance != 0 else 1
    except Exception as e:
        return 1  # Default to neutral beta on error

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio (risk-adjusted return)"""
    try:
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        if np.std(excess_returns) == 0:
            return 0
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    except Exception as e:
        return 0  # Default to zero on error

# Convert time period selection to yfinance format
yf_period = get_period_days(time_period)

# Get stock data for all portfolio stocks
all_stock_data = get_portfolio_data(st.session_state.portfolio, period=yf_period)

# Get benchmark data
benchmark_data = get_stock_data(benchmark, period=yf_period)

# Main Dashboard
if page == "Dashboard":
    # Header
    st.title("📈 Nexus Portfolio Dashboard")
    st.markdown("Interactive stock portfolio analysis and visualization platform")
    
    # Current portfolio value
    current_value, stock_values = calculate_portfolio_value(st.session_state.portfolio, all_stock_data)
    
    # Purchase value
    purchase_value = sum(st.session_state.portfolio['Shares'] * st.session_state.portfolio['Purchase Price'])
    
    # ROI
    total_roi = calculate_roi(purchase_value, current_value)
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Portfolio Value", f"${current_value:,.2f}", f"{total_roi:.2f}%")
    
    with col2:
        # Daily change
        if not benchmark_data.empty:
            daily_change = benchmark_data['Close'].pct_change().iloc[-1] * 100
            st.metric("Market Today", f"{benchmark.split(' ')[0]}", f"{daily_change:.2f}%")
    
    with col3:
        # Portfolio diversity (number of stocks)
        st.metric("Portfolio Diversity", f"{len(st.session_state.portfolio)} Assets")
    
    with col4:
        # Risk level (calculated as portfolio beta)
        portfolio_returns = pd.Series()
        for symbol, data in all_stock_data.items():
            if not data.empty:
                returns = data['Close'].pct_change().dropna()
                shares = st.session_state.portfolio[st.session_state.portfolio['Symbol'] == symbol]['Shares'].values[0]
                weighted_returns = returns * shares * data['Close'].iloc[0]
                portfolio_returns = portfolio_returns.add(weighted_returns, fill_value=0)
        
        market_returns = benchmark_data['Close'].pct_change().dropna()
        
        # Match the index of portfolio_returns and market_returns
        common_index = portfolio_returns.index.intersection(market_returns.index)
        if len(common_index) > 0:
            portfolio_returns = portfolio_returns.loc[common_index]
            market_returns = market_returns.loc[common_index]
            
            beta = calculate_beta(portfolio_returns.values, market_returns.values)
            risk_level = "High" if beta > 1.2 else "Medium" if beta > 0.8 else "Low"
            st.metric("Risk Level", risk_level, f"Beta: {beta:.2f}")
        else:
            st.metric("Risk Level", "N/A", "Insufficient data")
    
    # Portfolio performance chart
    st.markdown("### Portfolio Performance")
    
    portfolio_history = calculate_portfolio_history(st.session_state.portfolio, all_stock_data)
    
    if not portfolio_history.empty:
        # Normalize benchmark data to match portfolio initial value
        if not benchmark_data.empty and not portfolio_history.empty:
            # Convert benchmark_data to DataFrame with Date as index
            benchmark_df = benchmark_data.reset_index()[['Date', 'Close']]
            benchmark_df.columns = ['Date', 'Value']
            
            # Scale benchmark to match portfolio starting value
            scale_factor = portfolio_history['Value'].iloc[0] / benchmark_df['Value'].iloc[0]
            benchmark_df['Value'] = benchmark_df['Value'] * scale_factor
            
            fig = go.Figure()
            
            # Add portfolio line
            fig.add_trace(go.Scatter(
                x=portfolio_history['Date'],
                y=portfolio_history['Value'],
                mode='lines',
                name='Portfolio',
                line=dict(color='#4D34FF', width=3)
            ))
            
            # Add benchmark line
            fig.add_trace(go.Scatter(
                x=benchmark_df['Date'],
                y=benchmark_df['Value'],
                mode='lines',
                name=benchmark.split(' ')[0],
                line=dict(color='#FF5733', width=2, dash='dash')
            ))
            
            fig.update_layout(
                height=500,
                template='plotly_dark',
                title='Portfolio vs Benchmark Performance',
                xaxis_title='Date',
                yaxis_title='Value ($)',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    tickprefix='$'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Insufficient data to create portfolio performance chart. Try selecting a longer time period.")
    
    # Asset Allocation
    st.markdown("### Asset Allocation")
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Asset allocation pie chart
        labels = list(stock_values.keys())
        values = list(stock_values.values())
        
        if len(labels) > 0 and len(values) > 0:
            fig = px.pie(
                names=labels,
                values=values,
                title="Portfolio Allocation",
                hole=0.5,
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            
            fig.update_layout(
                template='plotly_dark',
                legend=dict(orientation="h", y=-0.1),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No portfolio data available for allocation chart.")
    
    with col2:
        # Portfolio breakdown table
        st.markdown("#### Portfolio Composition")
        
        portfolio_breakdown = []
        
        for symbol, shares in zip(st.session_state.portfolio['Symbol'], st.session_state.portfolio['Shares']):
            if symbol in all_stock_data and not all_stock_data[symbol].empty:
                current_price = all_stock_data[symbol]['Close'].iloc[-1]
                purchase_price = st.session_state.portfolio[st.session_state.portfolio['Symbol'] == symbol]['Purchase Price'].values[0]
                value = current_price * shares
                gain_loss = ((current_price - purchase_price) / purchase_price) * 100
                
                portfolio_breakdown.append({
                    'Symbol': symbol,
                    'Shares': shares,
                    'Current Price': f"${current_price:.2f}",
                    'Value': f"${value:.2f}",
                    'Gain/Loss': f"{gain_loss:.2f}%"
                })
        
        if portfolio_breakdown:
            breakdown_df = pd.DataFrame(portfolio_breakdown)
            st.dataframe(breakdown_df, hide_index=True, use_container_width=True)
        else:
            st.warning("No portfolio data available for breakdown table.")
    
    # Stock Performance Comparison
    st.markdown("### Individual Stock Performance")
    
    # Normalize all stock prices to 100 at the beginning
    normalized_data = pd.DataFrame()
    
    for symbol, data in all_stock_data.items():
        if not data.empty:
            normalized = data['Close'] / data['Close'].iloc[0] * 100
            normalized_data[symbol] = normalized
    
    if not normalized_data.empty:
        normalized_data.index.name = 'Date'
        normalized_data = normalized_data.reset_index()
        
        # Melt the dataframe for plotly
        melted_data = pd.melt(normalized_data, id_vars=['Date'], value_vars=list(normalized_data.columns[1:]))
        melted_data.columns = ['Date', 'Symbol', 'Normalized Price']
        
        # Create line chart
        fig = px.line(
            melted_data, 
            x='Date', 
            y='Normalized Price', 
            color='Symbol',
            title='Normalized Stock Performance (Base = 100)',
            labels={'Normalized Price': 'Normalized Price (Base=100)', 'Date': 'Date'}
        )
        
        fig.update_layout(
            height=500,
            template='plotly_dark',
            xaxis_title='Date',
            yaxis_title='Normalized Price (Base=100)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)'
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No stock performance data available for comparison chart.")
    
    # Upcoming Events
    st.markdown("### Upcoming Events & Announcements")
    
    # Sample earnings dates (in a real app, you would fetch these from an API)
    upcoming_events = [
        {"Date": "2023-05-03", "Symbol": "AAPL", "Event": "Earnings Release", "Expected Impact": "High"},
        {"Date": "2023-05-10", "Symbol": "MSFT", "Event": "Dividend Payout", "Expected Impact": "Medium"},
        {"Date": "2023-05-15", "Symbol": "AMZN", "Event": "Product Launch", "Expected Impact": "Medium"},
        {"Date": "2023-05-22", "Symbol": "GOOGL", "Event": "Investor Day", "Expected Impact": "Low"},
    ]
    
    events_df = pd.DataFrame(upcoming_events)
    
    st.dataframe(events_df, hide_index=True, use_container_width=True)

# Portfolio Analysis Page
elif page == "Portfolio Analysis":
    st.title("📊 Advanced Portfolio Analysis")
    
    # Risk-Return Profile
    st.markdown("### Risk-Return Profile")
    
    # Calculate returns and volatility for each stock
    risk_return_data = []
    
    for symbol, data in all_stock_data.items():
        if not data.empty:
            returns = data['Close'].pct_change().dropna()
            annualized_return = returns.mean() * 252 * 100  # Annualized return percentage
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility percentage
            
            risk_return_data.append({
                'Symbol': symbol,
                'Return': annualized_return,
                'Risk': volatility
            })
    
    risk_return_df = pd.DataFrame(risk_return_data)
    
    # Add benchmark
    if not benchmark_data.empty and not risk_return_df.empty:
        benchmark_returns = benchmark_data['Close'].pct_change().dropna()
        benchmark_annual_return = benchmark_returns.mean() * 252 * 100
        benchmark_volatility = benchmark_returns.std() * np.sqrt(252) * 100
        
        # Create a DataFrame with benchmark data and concat it to existing DataFrame
        benchmark_df = pd.DataFrame({
            'Symbol': [benchmark.split(' ')[0]],
            'Return': [benchmark_annual_return],
            'Risk': [benchmark_volatility]
        })
        risk_return_df = pd.concat([risk_return_df, benchmark_df], ignore_index=True)
    
    if not risk_return_df.empty:
        # Risk-Return Scatter Plot
        fig = px.scatter(
            risk_return_df,
            x='Risk',
            y='Return',
            text='Symbol',
            size=[10] * len(risk_return_df),
            color='Symbol',
            title="Risk-Return Analysis",
            labels={'Risk': 'Volatility (Annual %)', 'Return': 'Return (Annual %)'}
        )
        
        fig.update_traces(
            textposition='top center',
            marker=dict(size=15, opacity=0.8)
        )
        
        fig.update_layout(
            height=600,
            template='plotly_dark',
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)'
            )
        )
        
        # Add a diagonal line representing risk-reward balance
        max_risk = risk_return_df['Risk'].max() * 1.1
        max_return = risk_return_df['Return'].max() * 1.1
        fig.add_shape(
            type="line",
            x0=0, y0=0,
            x1=max_risk, y1=max_risk,
            line=dict(color="white", width=1, dash="dash")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Insufficient data for risk-return analysis. Try selecting a longer time period.")
    
    # Correlation Heatmap
    st.markdown("### Stock Correlation Analysis")
    
    # Create a dataframe with all stock returns
    all_returns = pd.DataFrame()
    
    for symbol, data in all_stock_data.items():
        if not data.empty:
            all_returns[symbol] = data['Close'].pct_change().fillna(0)
    
    if not all_returns.empty and len(all_returns.columns) > 1:  # Need at least 2 stocks for correlation
        # Calculate correlation matrix
        corr_matrix = all_returns.corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='Viridis',
            title="Stock Correlation Matrix"
        )
        
        fig.update_layout(
            height=500,
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Insufficient data for correlation analysis. Need at least two stocks with price data.")
    
    # Portfolio Metrics
    st.markdown("### Portfolio Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    # Calculate portfolio returns
    portfolio_returns = pd.Series(dtype=float)
    for symbol, data in all_stock_data.items():
        if not data.empty:
            returns = data['Close'].pct_change().dropna()
            shares = st.session_state.portfolio[st.session_state.portfolio['Symbol'] == symbol]['Shares'].values[0]
            weighted_returns = returns * shares * data['Close'].iloc[0]
            portfolio_returns = portfolio_returns.add(weighted_returns, fill_value=0)
    
    if len(portfolio_returns) > 0:
        # Sharpe Ratio
        sharpe_ratio = calculate_sharpe_ratio(portfolio_returns)
        with col1:
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            st.markdown("*Higher is better. >1 is good.*")
        
        # Maximum Drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = drawdown.min() * 100
        
        with col2:
            st.metric("Maximum Drawdown", f"{max_drawdown:.2f}%")
            st.markdown("*Lower is better. Measures worst peak-to-trough drop.*")
        
        # Consistency (% of positive days)
        positive_days = (portfolio_returns > 0).sum() / len(portfolio_returns) * 100
        
        with col3:
            st.metric("Win Rate", f"{positive_days:.2f}%")
            st.markdown("*Higher is better. Percentage of positive return days.*")
    else:
        st.warning("Insufficient data for portfolio performance metrics.")
    
    # Monthly Returns Heatmap
    st.markdown("### Monthly Returns Heatmap")
    
    if len(portfolio_returns) > 0:
        # Create a datetime index for portfolio returns
        if not isinstance(portfolio_returns.index, pd.DatetimeIndex):
            portfolio_returns.index = pd.to_datetime(portfolio_returns.index)
        
        # Resample to monthly returns
        monthly_returns = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        
        # Create a DataFrame with year and month columns
        monthly_df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values
        })
        
        # Check if we have enough data for the heatmap
        if len(monthly_df) > 0:
            # Create a pivot table for the heatmap
            try:
                monthly_pivot = pd.pivot_table(
                    data=monthly_df,
                    values='return',
                    index='year',
                    columns='month',
                    aggfunc='mean'
                )
                
                # Get available months (columns) in the pivot table
                available_months = monthly_pivot.columns.tolist()
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                # Only use month names for available months
                month_labels = [month_names[i-1] for i in available_months]
                
                # Update column names safely - ensuring lengths match
                monthly_pivot.columns = month_labels
                
                # Create the heatmap
                fig = px.imshow(
                    monthly_pivot,
                    text_auto=".2f",
                    color_continuous_scale='RdBu_r',
                    color_continuous_midpoint=0,
                    title="Monthly Portfolio Returns (%)"
                )
                
                fig.update_layout(
                    height=400,
                    template='plotly_dark',
                    xaxis_title='Month',
                    yaxis_title='Year',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except ValueError as e:
                st.warning(f"Not enough diverse data to create a monthly heatmap. Try selecting a longer time period or adding more stocks.")
        else:
            st.warning("Not enough data to create a monthly returns heatmap. Try selecting a longer time period.")
    else:
        st.warning("Insufficient data for monthly returns analysis.")

# Stock Screener Page
elif page == "Stock Screener":
    st.title("🔍 Stock Screener")
    
    # Filters
    st.markdown("### Screening Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_price = st.number_input("Minimum Price ($)", value=0.0, step=10.0)
        max_pe = st.number_input("Maximum P/E Ratio", value=100.0, step=5.0)
    
    with col2:
        max_price = st.number_input("Maximum Price ($)", value=1000.0, step=10.0)
        min_dividend = st.number_input("Minimum Dividend Yield (%)", value=0.0, step=0.5)
    
    with col3:
        sector = st.selectbox("Sector", ["All", "Technology", "Healthcare", "Financial", "Consumer", "Energy"])
        growth_rate = st.number_input("Minimum Growth Rate (%)", value=0.0, step=5.0)
    
    st.markdown("---")
    
    # List of stocks (this would come from an API in a real app)
    stocks_data = [
        {"Symbol": "AAPL", "Name": "Apple Inc.", "Price": 175.30, "P/E": 29.8, "Div Yield": 0.52, "Sector": "Technology", "Growth": 8.2},
        {"Symbol": "MSFT", "Name": "Microsoft Corp.", "Price": 330.50, "P/E": 35.2, "Div Yield": 0.75, "Sector": "Technology", "Growth": 15.8},
        {"Symbol": "AMZN", "Name": "Amazon.com Inc.", "Price": 3400.00, "P/E": 65.4, "Div Yield": 0.0, "Sector": "Consumer", "Growth": 22.5},
        {"Symbol": "GOOGL", "Name": "Alphabet Inc.", "Price": 2750.00, "P/E": 27.1, "Div Yield": 0.0, "Sector": "Technology", "Growth": 24.8},
        {"Symbol": "JNJ", "Name": "Johnson & Johnson", "Price": 170.50, "P/E": 24.5, "Div Yield": 2.48, "Sector": "Healthcare", "Growth": 4.5},
        {"Symbol": "PG", "Name": "Procter & Gamble", "Price": 145.80, "P/E": 25.3, "Div Yield": 2.38, "Sector": "Consumer", "Growth": 6.1},
        {"Symbol": "JPM", "Name": "JPMorgan Chase", "Price": 155.70, "P/E": 10.5, "Div Yield": 2.65, "Sector": "Financial", "Growth": 7.8},
        {"Symbol": "XOM", "Name": "Exxon Mobil Corp.", "Price": 65.25, "P/E": 15.2, "Div Yield": 5.23, "Sector": "Energy", "Growth": 3.2},
    ]
    
    # Apply filters
    filtered_stocks = [
        stock for stock in stocks_data
        if (stock['Price'] >= min_price and
            stock['Price'] <= max_price and
            stock['P/E'] <= max_pe and
            stock['Div Yield'] >= min_dividend and
            stock['Growth'] >= growth_rate and
            (sector == "All" or stock['Sector'] == sector))
    ]
    
    # Display results
    if filtered_stocks:
        st.markdown(f"### Results ({len(filtered_stocks)} stocks)")
        
        # Convert to DataFrame for display
        filtered_df = pd.DataFrame(filtered_stocks)
        
        # Format columns
        filtered_df['Price'] = filtered_df['Price'].apply(lambda x: f"${x:.2f}")
        filtered_df['P/E'] = filtered_df['P/E'].apply(lambda x: f"{x:.1f}")
        filtered_df['Div Yield'] = filtered_df['Div Yield'].apply(lambda x: f"{x:.2f}%")
        filtered_df['Growth'] = filtered_df['Growth'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(filtered_df, hide_index=True, use_container_width=True)
        
        # Add stocks button
        if st.button("Add Selected Stocks to Portfolio"):
            st.success("Selected stocks would be added to your portfolio in a real app!")
    else:
        st.warning("No stocks match your criteria. Try adjusting the filters.")
    
    # Stock comparison
    st.markdown("### Stock Comparison")
    
    # Allow user to select stocks to compare
    selected_symbols = st.multiselect("Select stocks to compare", [stock['Symbol'] for stock in stocks_data])
    
    if selected_symbols:
        # Get data for selected stocks
        comparison_data = {}
        for symbol in selected_symbols:
            stock_info = next((stock for stock in stocks_data if stock['Symbol'] == symbol), None)
            if stock_info:
                comparison_data[symbol] = stock_info
        
        # Create radar chart for comparison
        categories = ['P/E', 'Div Yield', 'Growth']
        
        fig = go.Figure()
        
        for symbol, data in comparison_data.items():
            fig.add_trace(go.Scatterpolar(
                r=[data['P/E']/max(s['P/E'] for s in stocks_data) * 100,
                   data['Div Yield']/max(s['Div Yield'] for s in stocks_data) * 100,
                   data['Growth']/max(s['Growth'] for s in stocks_data) * 100],
                theta=categories,
                fill='toself',
                name=symbol
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            template='plotly_dark',
            title="Stock Comparison (% of maximum values)",
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Industry average comparison
    st.markdown("### Industry Averages")
    
    # Industry averages (would come from an API in a real app)
    industry_data = {
        "Technology": {"P/E": 30.5, "Div Yield": 0.7, "Growth": 18.5},
        "Healthcare": {"P/E": 25.2, "Div Yield": 2.1, "Growth": 8.3},
        "Financial": {"P/E": 12.4, "Div Yield": 3.2, "Growth": 6.7},
        "Consumer": {"P/E": 26.3, "Div Yield": 2.5, "Growth": 7.5},
        "Energy": {"P/E": 16.8, "Div Yield": 4.8, "Growth": 3.5}
    }
    
    # Create bar chart for industry comparison
    industry_df = pd.DataFrame.from_dict(industry_data, orient='index')
    
    # P/E ratio comparison
    fig = px.bar(
        industry_df,
        y=industry_df.index,
        x='P/E',
        title="Industry Average P/E Ratios",
        orientation='h',
        height=400
    )
    
    fig.update_layout(
        template='plotly_dark',
        xaxis_title="P/E Ratio",
        yaxis_title="Industry",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Prediction Models Page
elif page == "Prediction Models":
    st.title("🔮 Stock Price Prediction")
    
    # Stock selection for prediction
    stock_to_predict = st.selectbox("Select Stock for Prediction", st.session_state.portfolio['Symbol'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        prediction_days = st.slider("Prediction Horizon (Days)", 7, 90, 30)
    
    with col2:
        confidence_level = st.slider("Confidence Level (%)", 50, 95, 80)
    
    # Get historical data for the selected stock
    if stock_to_predict in all_stock_data and not all_stock_data[stock_to_predict].empty:
        stock_data = all_stock_data[stock_to_predict]
        
        # Display stock info
        st.markdown(f"### Historical Data: {stock_to_predict}")
        
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name='Price Action'
        ))
        
        # Add volume bar chart
        fig.add_trace(go.Bar(
            x=stock_data.index,
            y=stock_data['Volume'],
            name='Volume',
            yaxis='y2',
            marker=dict(color='rgba(200, 200, 200, 0.5)')
        ))
        
        # Layout
        fig.update_layout(
            title=f"{stock_to_predict} Historical Price and Volume",
            template='plotly_dark',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            xaxis_rangeslider_visible=False,
            yaxis=dict(
                domain=[0.3, 1]
            ),
            yaxis2=dict(
                domain=[0, 0.2],
                title='Volume',
                showticklabels=False
            ),
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction
        st.markdown("### Price Prediction")
        
        # Generate Prediction Button
        if st.button("Generate Prediction"):
            # Show a spinner during calculation
            with st.spinner("Generating prediction..."):
                try:
                    # Check if we have sufficient data
                    if len(stock_data) < 20:
                        st.error("Not enough historical data available. Need at least 20 data points.")
                    else:
                        # Calculate prediction
                        forecast = predict_stock_price(stock_data, days=prediction_days)
                        
                        # Create dates for the prediction
                        last_date = stock_data.index[-1]
                        prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days)
                        
                        # Calculate confidence intervals
                        z_score = {
                            50: 0.67,
                            60: 0.84,
                            70: 1.04,
                            80: 1.28,
                            90: 1.65,
                            95: 1.96
                        }[min(confidence_level, 95)]
                        
                        # Estimate volatility from historical data (with safeguards)
                        historical_returns = stock_data['Close'].pct_change().dropna()
                        if len(historical_returns) > 0:
                            historical_volatility = max(0.01, historical_returns.std())  # Minimum volatility floor
                        else:
                            historical_volatility = 0.02  # Default if we can't calculate
                        
                        # Calculate confidence bands
                        conf_interval = z_score * historical_volatility * stock_data['Close'].iloc[-1] * np.sqrt(np.arange(1, prediction_days + 1))
                        upper_bound = forecast + conf_interval
                        lower_bound = forecast - conf_interval
                        
                        # Create prediction plot
                        fig = go.Figure()
                        
                        # Add historical data
                        fig.add_trace(go.Scatter(
                            x=stock_data.index,
                            y=stock_data['Close'],
                            mode='lines',
                            name='Historical',
                            line=dict(color='#4D34FF')
                        ))
                        
                        # Add prediction
                        fig.add_trace(go.Scatter(
                            x=prediction_dates,
                            y=forecast,
                            mode='lines',
                            name='Prediction',
                            line=dict(color='#FF5733')
                        ))
                        
                        # Add confidence intervals
                        fig.add_trace(go.Scatter(
                            x=pd.concat([pd.Series(prediction_dates), pd.Series(prediction_dates[::-1])]),
                            y=pd.concat([pd.Series(upper_bound), pd.Series(lower_bound[::-1])]),
                            fill='toself',
                            fillcolor='rgba(255, 87, 51, 0.2)',
                            line=dict(color='rgba(255, 255, 255, 0)'),
                            name=f'{confidence_level}% Confidence Interval'
                        ))
                        
                        # Layout
                        fig.update_layout(
                            title=f"{stock_to_predict} Price Prediction for Next {prediction_days} Days",
                            xaxis_title='Date',
                            yaxis_title='Price ($)',
                            template='plotly_dark',
                            height=500,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Prediction summary
                        st.markdown("### Prediction Summary")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Current Price", 
                                f"${stock_data['Close'].iloc[-1]:.2f}"
                            )
                        
                        with col2:
                            predicted_price = forecast[-1]
                            change = ((predicted_price / stock_data['Close'].iloc[-1]) - 1) * 100
                            st.metric(
                                f"Predicted Price ({prediction_days} days)", 
                                f"${predicted_price:.2f}",
                                f"{change:.2f}%"
                            )
                        
                        with col3:
                            st.metric(
                                "Prediction Range", 
                                f"${lower_bound[-1]:.2f} - ${upper_bound[-1]:.2f}"
                            )
                            
                        # Add a note about prediction accuracy
                        st.info("Note: Price predictions are estimates based on historical patterns and may not reflect future market conditions. Always do your own research before making investment decisions.")
                        
                        # Technical indicators
                        st.markdown("### Technical Indicators")
                        
                        # Calculate some technical indicators
                        # Moving Averages
                        stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
                        stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
                        
                        # RSI (Relative Strength Index)
                        delta = stock_data['Close'].diff()
                        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                        rs = gain / loss
                        stock_data['RSI'] = 100 - (100 / (1 + rs))
                        
                        # MACD (Moving Average Convergence Divergence)
                        stock_data['EMA_12'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
                        stock_data['EMA_26'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
                        stock_data['MACD'] = stock_data['EMA_12'] - stock_data['EMA_26']
                        stock_data['Signal_Line'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()
                        
                        # Display technical indicators table
                        last_row = stock_data.iloc[-1]
                        
                        indicators = {
                            "Indicator": ["Price", "SMA (20)", "SMA (50)", "RSI (14)", "MACD", "Signal Line"],
                            "Value": [
                                f"${last_row['Close']:.2f}",
                                f"${last_row['SMA_20']:.2f}" if not np.isnan(last_row['SMA_20']) else "N/A",
                                f"${last_row['SMA_50']:.2f}" if not np.isnan(last_row['SMA_50']) else "N/A",
                                f"{last_row['RSI']:.2f}" if not np.isnan(last_row['RSI']) else "N/A",
                                f"{last_row['MACD']:.4f}" if not np.isnan(last_row['MACD']) else "N/A",
                                f"{last_row['Signal_Line']:.4f}" if not np.isnan(last_row['Signal_Line']) else "N/A"
                            ],
                            "Signal": [
                                "N/A",
                                "Buy" if not np.isnan(last_row['SMA_20']) and last_row['Close'] > last_row['SMA_20'] else "Sell" if not np.isnan(last_row['SMA_20']) else "N/A",
                                "Buy" if not np.isnan(last_row['SMA_50']) and last_row['Close'] > last_row['SMA_50'] else "Sell" if not np.isnan(last_row['SMA_50']) else "N/A",
                                "Oversold" if not np.isnan(last_row['RSI']) and last_row['RSI'] < 30 else "Overbought" if not np.isnan(last_row['RSI']) and last_row['RSI'] > 70 else "Neutral" if not np.isnan(last_row['RSI']) else "N/A",
                                "Bullish" if not np.isnan(last_row['MACD']) and not np.isnan(last_row['Signal_Line']) and last_row['MACD'] > last_row['Signal_Line'] else "Bearish" if not np.isnan(last_row['MACD']) and not np.isnan(last_row['Signal_Line']) else "N/A",
                                "N/A"
                            ]
                        }
                        
                        indicators_df = pd.DataFrame(indicators)
                        st.dataframe(indicators_df, hide_index=True, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error generating prediction: {str(e)}")
                    st.info("Try selecting a different stock or time period with more historical data.")
    else:
        st.warning(f"No data available for {stock_to_predict}")

# Settings Page
elif page == "Settings":
    st.title("⚙️ Dashboard Settings")
    
    # Portfolio Management
    st.markdown("### Portfolio Management")
    
    edit_mode = st.toggle("Edit Portfolio")
    
    if edit_mode:
        edited_df = st.data_editor(
            st.session_state.portfolio,
            num_rows="dynamic",
            use_container_width=True
        )
        
        if st.button("Save Changes"):
            st.session_state.portfolio = edited_df
            st.success("Portfolio updated successfully!")
    
    # Data refresh settings
    st.markdown("### Data Refresh Settings")
    
    refresh_interval = st.select_slider(
        "Data Refresh Interval",
        options=["1 minute", "5 minutes", "15 minutes", "30 minutes", "1 hour", "Manual only"]
    )
    
    # Notification settings
    st.markdown("### Notification Settings")
    
    st.checkbox("Price Alert Notifications", value=True)
    st.checkbox("Earnings Announcement Reminders", value=True)
    st.checkbox("Portfolio Summary (Daily)", value=False)
    
    price_change_threshold = st.slider("Price Change Alert Threshold (%)", 1.0, 10.0, 5.0)
    
    # Display settings
    st.markdown("### Display Settings")
    
    default_chart = st.radio("Default Chart Type", ["Candlestick", "Line", "OHLC"])
    show_volume = st.checkbox("Show Volume on Charts", value=True)
    
    # Export data
    st.markdown("### Export Data")
    
    export_format = st.selectbox("Export Format", ["CSV", "Excel", "PDF", "JSON"])
    
    if st.button("Export Portfolio Data"):
        st.success(f"Portfolio data would be exported as {export_format} in a real app!")
    
    # Clear data
    st.markdown("### Data Management")
    
    if st.button("Reset to Default Portfolio"):
        st.session_state.portfolio = pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'NVDA'],
            'Shares': [10, 5, 3, 2, 8, 4],
            'Purchase Price': [150.75, 280.50, 3200.00, 2500.00, 220.30, 180.75],
            'Purchase Date': ['2021-01-15', '2021-02-20', '2021-03-10', '2021-04-05', '2021-06-15', '2021-05-10']
        })
        st.success("Portfolio reset to default values!")

# Add a footer 
st.markdown("---")
st.markdown("### Nexus Portfolio Analyzer | Created with Streamlit")
st.markdown("🌟 For educational purposes only. Not financial advice.")

# Run the app with the following command:
# streamlit run app.py_ratio(