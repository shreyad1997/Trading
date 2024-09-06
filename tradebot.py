
pip install vaderSentiment
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import defaultdict
import yfinance as yf
import pandas as pd
import numpy as np

# Define an expansive watchlist
expansive_watchlist = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
     'NFLX', 'NVDA', 'BABA', 'V', 'SPY', "BRK.B","VOO","DT", "PLTR","UBER","AMC","IBM","META"
    # Add more stocks as needed
]

def fetch_alpha_vantage_news(stock_symbol):
    api_key = 'UDV6EF6GVUCM3E5O'
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={stock_symbol}&apikey={api_key}'
    response = requests.get(url)
    articles = response.json().get('feed', [])

    headlines = [article['title'] for article in articles]
    return headlines

def analyze_sentiment(headlines):
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(headline)['compound'] for headline in headlines]
    return sum(scores) / len(scores) if scores else 0

def analyze_trends(stock_symbols):
    stock_trends = defaultdict(dict)

    for stock in stock_symbols:
        headlines = fetch_alpha_vantage_news(stock)
        if not headlines:
            print(f"No headlines found for {stock}.")
            continue
        sentiment_score = analyze_sentiment(headlines)

        # Count keyword occurrences in headlines
        reasons = defaultdict(int)
        for headline in headlines:
            if "earnings" in headline.lower():
                reasons["Earnings"] += 1
            if "acquisition" in headline.lower():
                reasons["Acquisition"] += 1
            if "launch" in headline.lower():
                reasons["Product Launch"] += 1
            # Add more keywords as needed

        # Store the results
        stock_trends[stock]['mentions'] = len(headlines)
        stock_trends[stock]['sentiment'] = sentiment_score
        stock_trends[stock]['reasons'] = reasons

        print(stock_trends)

    return stock_trends

def rank_trending_stocks(stock_trends):
    # Rank based on mentions and sentiment
    ranked_stocks = sorted(stock_trends.items(), key=lambda x: (x[1]['mentions'], x[1]['sentiment']), reverse=True)
    return ranked_stocks[:5]  # Return top 5 stocks

# Function to run the strategy for each stock
def calculate_sma(data, window):
    return data.rolling(window=window).mean()

def calculate_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()

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

def run_strategy(stock):
    data = yf.download(stock, interval='5m', period='1d')
    data['SMA_50'] = calculate_sma(data['Close'], 50)
    data['SMA_200'] = calculate_sma(data['Close'], 200)
    data['RSI'] = calculate_rsi(data['Close'])
    data['MACD'], data['Signal_Line'] = calculate_macd(data['Close'])
    data['Upper_Band'], data['Lower_Band'] = calculate_bollinger_bands(data['Close'])

    buy_signals, sell_signals = get_buy_sell_signals(data)
    data['Buy_Signal'] = buy_signals
    data['Sell_Signal'] = sell_signals
    return data[['Close', 'Buy_Signal', 'Sell_Signal']]

# Analyze trends in the expansive watchlist
stock_trends = analyze_trends(expansive_watchlist)

# Get the top 5 trending stocks
top_stocks = rank_trending_stocks(stock_trends)

# Run the strategy for the top trending stocks
for stock, _ in top_stocks:
    result = run_strategy(stock)
    print(f"Results for {stock}:\n", result.dropna(how='all').to_string())





