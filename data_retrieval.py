import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def query_yf(stock: tuple, period: int) -> pd.DataFrame:
    """
    Query the Yahoo Finance API for a given stock and store the data.

    :param stock: tuple
        A tuple containing three elements:
        - company (str): The name of the company.
        - ticker (str): The stock ticker symbol of the company.
        - exchange (str): The stock exchange where the company is listed.
    :param period: int
        The number of days of data to retrieve.

    :return: data: a pandas dataframe of the data
    """
    try:
        # Extract company, ticker, and exchange from the stock tuple
        company, ticker, exchange = stock

        yesterday = datetime.now() - timedelta(days=period)
        today = datetime.now()

        yesterday_str = yesterday.strftime('%Y-%m-%d')
        today_str = today.strftime('%Y-%m-%d')

        # Query the API using yfinance
        stock_data = yf.Ticker(ticker)
        data = stock_data.history(period='1d', start=yesterday_str, end=today_str)

        # Add the company name to the dataframe
        data['Company'] = company
        data['Ticker'] = ticker
        data['Exchange'] = exchange

        # Reorder the columns
        data = data[['Company', 'Ticker', 'Exchange', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # Convert DataFrame index to a 'DatetimeIndex' without time zone information
        data.index = data.index.tz_localize(None)  # this is necessary for some algorithms

        return data
    except Exception as e:
        print(f"Error retrieving data: {e}")
        return None

def query_yf_list(stocks: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    Query the Yahoo Finance API for all stocks in a list based on their exchange.

    :param stocks: list
        A list of tuples containing three elements:
        - company (str): The name of the company.
        - ticker (str): The stock ticker symbol of the company.
        - exchange (str): The stock exchange where the company is listed.
    :param period: int
        The number of days of data to retrieve.

    :return: data: a list of pandas dataframes of the data
    """
    # Create an empty list
    data = []

    # Loop through the list of stocks
    for _, row in stocks.iterrows():
        stock = (row['Company'], row['Ticker'], row['Stock Exchange'])

        # Query the API using yfinance
        stock_data = query_yf(stock, period)

        # Append the data to the dataframe
        data.append(stock_data)

    return data

