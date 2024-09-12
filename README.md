Stock Investment Strategy Simulation

This Python script simulates an investment strategy using news sentiment analysis and technical analysis to generate buy and sell signals for stocks. The script fetches stock price data from Yahoo Finance using the yfinance package and performs sentiment analysis on news headlines using NewsAPI. Based on these inputs, it generates buy and sell recommendations.

Key Features:

•	Yahoo Finance (yfinance): Used to fetch real-time and historical stock data.
•	NewsAPI: Fetches news headlines for stocks, and sentiment analysis is performed using the VADER Sentiment Intensity Analyzer.
•	Technical Analysis: The script calculates technical indicators such as:
•	Simple Moving Average (SMA)
•	Relative Strength Index (RSI)
•	Moving Average Convergence Divergence (MACD)
•	Bollinger Bands
•	Investment Simulation: Based on the buy/sell signals from the technical analysis, the script suggests investment actions for a specific weekly run (e.g., “Run 1, Week 1”).

How It Works:

1.	Fetch News Data:
•	The script uses NewsAPI to fetch news articles for a list of stocks. It then performs sentiment analysis using VADER to score the headlines, determining the sentiment for each stock.
2.	Rank Stocks:
•	The stocks are ranked based on the number of mentions in the news and their sentiment scores.
3.	Fetch Stock Price Data:
•	Using Yahoo Finance, the script fetches intraday stock price data for the top 5 ranked stocks.
4.	Technical Analysis:
•	The script calculates various technical indicators such as SMA, RSI, MACD, and Bollinger Bands.
•	Based on these indicators, the script generates buy and sell signals.
5.	Investment Simulation:
•	The script simulates a weekly investment run (e.g., “Run 1, Week 1”) based on the buy/sell signals.
•	It allocates a predefined budget (e.g., $125) to stocks with buy signals and prints investment recommendations.
