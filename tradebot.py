import yfinance as yf
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np

# Fetch stock data using Yahoo Finance
def fetch_stock_data_yahoo(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    data = stock.history(period='1d', interval='5m')  # Fetch intraday 5-minute data for 1 day
    return data

# NewsAPI to fetch news and analyze sentiment
def fetch_newsapi_news(stock_symbol):
    api_key = 'insert_api_key'  # Replace with your NewsAPI key
    url = f'https://newsapi.org/v2/everything?q={stock_symbol}&apiKey={api_key}'
    response = requests.get(url)
    articles = response.json().get('articles', [])
    
    headlines = [article['title'] for article in articles]
    return headlines

def analyze_sentiment(headlines):
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(headline)['compound'] for headline in headlines]
    return sum(scores) / len(scores) if scores else 0

# Analyze news trends and rank stocks
def analyze_trends(stock_symbols):
    stock_trends = {}

    for stock in stock_symbols:
        headlines = fetch_newsapi_news(stock)
        if not headlines:
            print(f"No headlines found for {stock}.")
            continue
        sentiment_score = analyze_sentiment(headlines)
        stock_trends[stock] = {
            'sentiment': sentiment_score,
            'mentions': len(headlines)
        }
    
    return stock_trends

def rank_trending_stocks(stock_trends):
    ranked_stocks = sorted(stock_trends.items(), key=lambda x: (x[1]['mentions'], x[1]['sentiment']), reverse=True)
    return ranked_stocks[:5]  # Return top 5 stocks

# Technical analysis using Yahoo Finance data
def calculate_sma(data, window):
    return data.rolling(window=window).mean()

def calculate_rsi(data, window=14):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, slow=26, fast=12, signal=9):
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd = ema_fast - ema_slow
    signal_line = calculate_ema(macd, signal)
    return macd, signal_line

def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    sma = calculate_sma(data, window)
    std_dev = data.rolling(window=window).std()
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    return upper_band, lower_band

def get_buy_sell_signals(data):
    buy_signals = [np.nan] * len(data)
    sell_signals = [np.nan] * len(data)

    for i in range(1, len(data)):
        if data['SMA_50'].iloc[i] > data['SMA_200'].iloc[i] and data['SMA_50'].iloc[i-1] <= data['SMA_200'].iloc[i-1]:
            buy_signals[i] = data['Close'].iloc[i]
        elif data['SMA_50'].iloc[i] < data['SMA_200'].iloc[i] and data['SMA_50'].iloc[i-1] >= data['SMA_200'].iloc[i-1]:
            sell_signals[i] = data['Close'].iloc[i]
        elif data['RSI'].iloc[i] < 30:
            buy_signals[i] = data['Close'].iloc[i]
        elif data['RSI'].iloc[i] > 70:
            sell_signals[i] = data['Close'].iloc[i]
        elif data['MACD'].iloc[i] > data['Signal_Line'].iloc[i] and data['MACD'].iloc[i-1] <= data['Signal_Line'].iloc[i-1]:
            buy_signals[i] = data['Close'].iloc[i]
        elif data['MACD'].iloc[i] < data['Signal_Line'].iloc[i] and data['MACD'].iloc[i-1] >= data['Signal_Line'].iloc[i-1]:
            sell_signals[i] = data['Close'].iloc[i]
        elif data['Close'].iloc[i] < data['Lower_Band'].iloc[i]:
            buy_signals[i] = data['Close'].iloc[i]
        elif data['Close'].iloc[i] > data['Upper_Band'].iloc[i]:
            sell_signals[i] = data['Close'].iloc[i]

    return buy_signals, sell_signals

# Run strategy using Yahoo Finance stock data
def run_strategy(stock_symbol):
    data = fetch_stock_data_yahoo(stock_symbol)
    data['SMA_50'] = calculate_sma(data['Close'], 50)
    data['SMA_200'] = calculate_sma(data['Close'], 200)
    data['RSI'] = calculate_rsi(data['Close'])
    data['MACD'], data['Signal_Line'] = calculate_macd(data['Close'])
    data['Upper_Band'], data['Lower_Band'] = calculate_bollinger_bands(data['Close'])

    buy_signals, sell_signals = get_buy_sell_signals(data)
    data['Buy_Signal'] = buy_signals
    data['Sell_Signal'] = sell_signals
    return data[['Close', 'Buy_Signal', 'Sell_Signal']]

# Simulate investments based on buy/sell signals for one run
def simulate_investment_run(top_stocks, budget, run_name):
    # Allocate the budget across buy signals
    investment_plan = {}
    buy_signals_stocks = [stock for stock in top_stocks if stock['buy_signal']]
    
    if buy_signals_stocks:
        portion_per_stock = budget / len(buy_signals_stocks)
        for stock in buy_signals_stocks:
            investment_plan[stock['symbol']] = portion_per_stock
    
    # Print investment recommendations
    print(f"\nInvestment Recommendations for {run_name}:")
    if investment_plan:
        for stock, amount in investment_plan.items():
            print(f"Buy ${amount:.2f} worth of {stock}")
    else:
        print("No Buy Signals today.")
    
    # Print sell recommendations
    for stock in top_stocks:
        if stock['sell_signal']:
            print(f"Sell {stock['symbol']} based on Sell Signal.")

# Example usage:

expansive_watchlist = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Step 1: Analyze trends based on news sentiment
stock_trends = analyze_trends(expansive_watchlist)

# Step 2: Rank the top 5 trending stocks
top_stocks_info = rank_trending_stocks(stock_trends)

# Step 3: Get technical analysis for top trending stocks
top_stocks_with_signals = []
for stock, _ in top_stocks_info:
    result = run_strategy(stock)
    result_with_signals = {
        'symbol': stock,
        'buy_signal': not result['Buy_Signal'].isna().all(),
        'sell_signal': not result['Sell_Signal'].isna().all()
    }
    top_stocks_with_signals.append(result_with_signals)

# Step 4: Simulate a single run (e.g., "Run 1, Week 1")
simulate_investment_run(top_stocks_with_signals, budget=125, run_name="Run 1, Week 1")
