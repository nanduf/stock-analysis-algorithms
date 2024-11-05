import numpy as np
import pandas as pd
from scipy.fft import fft


def close_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the change in price from the previous day.
    :param df: dataframe of stock data
    :return: dataframe of stock data with added column for change in 'close' from previous day
    """
    df['Close Price Change'] = df['Close'].diff()
    df['Close Price Change'].iloc[0] = 0
    return df


def day_average(df: pd.DataFrame, column: str = 'Close', days: int = 30, company_name: str = None) -> pd.DataFrame:
    """
    Compute the n-day average of the stock price.
    :param column: column to average
    :param df: dataframe of stock data
    :param days: number of days to average
    :return: dataframe of stock data with added column for an n day average
    """
    if company_name is not None:
        column_title = f"{company_name}-{column} Day Average"
    else:
        column_title = f"{column} Day Average"
    df[column_title] = df[column].rolling(days).mean().shift(1)  # shift to not include today
    day_mean = df[column].iloc[0:days + 1].mean()
    df[column_title] = df[column_title].fillna(day_mean)

    return df


def day_average_std(df: pd.DataFrame, columns: [str] = ['Close', 'Open', 'High', 'Low', 'Volume'], days: int = 30,
                    company_name: str = None) -> pd.DataFrame:
    """
    Compute the n-day average and standard deviation of the stock price.
    :param column: column to average
    :param df: dataframe of stock data
    :param days: number of days to average
    :return: dataframe of stock data with added column for an n day average
    """
    for column in columns:
        if company_name is not None:
            column_title = f"{company_name}-{column} Day Avg"
        else:
            column_title = f"{column} Day Avg"

        df[column_title] = df[column].rolling(days).mean().shift(1).astype('float32')  # shift to not include today
        day_mean = df[column].iloc[0:days + 1].mean().astype('float32')
        df[column_title] = df[column_title].fillna(day_mean)

        # now add the standard deviation
        if company_name is not None:
            column_title = f"{company_name}-{column} Day Std"
        else:
            column_title = f"{column} Day Std"
        df[column_title] = df[column].rolling(days).std().shift(1).astype('float32')  # shift to not include today
        day_std = df[column].iloc[0:days + 1].std().astype('float32')
        df[column_title] = df[column_title].fillna(day_std)

    return df


def previous_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the previous day's data to the dataframe as columns.
    :param df: dataframe of stock data
    :return: dataframe of stock data with added column for previous day's closing price
    """
    df['Previous Close'] = df['Close'].shift(1)
    df['Previous Close'].iloc[0] = df['Close'].iloc[0]

    df['Previous Open'] = df['Open'].shift(1)
    df['Previous Open'].iloc[0] = df['Open'].iloc[0]

    df['Previous High'] = df['High'].shift(1)
    df['Previous High'].iloc[0] = df['High'].iloc[0]

    df['Previous Low'] = df['Low'].shift(1)
    df['Previous Low'].iloc[0] = df['Low'].iloc[0]

    df['Previous Volume'] = df['Volume'].shift(1)
    df['Previous Volume'].iloc[0] = df['Volume'].iloc[0]

    return df


def detect_seasonality(df: pd.DataFrame, max_period: int = 365) -> (int, int, int):
    """
    Detect the seasonalities in a time series using the Fourier transform.
    :param df: Dataframe of stock data
    :param max_period: the max period to consider for seasonalities
    :return: the top three seasonalities for the 'Close' column
    """

    num_seasonalities = 3
    tolerance = 0.05

    # Perform the Fourier transform on the 'Close' column
    close_array = df['Close'].to_numpy()
    close_fft = fft(close_array)
    frequencies = np.fft.fftfreq(len(close_fft), d=1)  # d is the spacing between samples, which is 1 trading day
    power_spectrum = np.abs(close_fft) ** 2

    # Exclude the zero frequency term for analysis
    positive_frequencies = frequencies[frequencies > 0]
    positive_power_spectrum = power_spectrum[frequencies > 0]

    # Find all peaks in the frequency spectrum sorted by power
    peaks_indices = np.argsort(positive_power_spectrum)[::-1]  # Sort in descending order of power
    peak_frequencies = positive_frequencies[peaks_indices]
    peaks_periods = 1 / peak_frequencies

    # Filter to find peaks corresponding to periods under the specified max_period
    filtered_peaks_periods = peaks_periods[peaks_periods < max_period]

    # Select the top three seasonalities, excluding harmonics
    likely_seasonalities = []
    for period in filtered_peaks_periods:
        if not any(abs(period - p) < tolerance for p in likely_seasonalities):
            likely_seasonalities.append(period)
        if len(likely_seasonalities) == num_seasonalities:
            break

    # Return the likley seasonalities rounded to integers
    return np.round(likely_seasonalities).astype(int)


def overall_averages(data_list: [pd.DataFrame]) -> pd.DataFrame:
    """
    Compute the average values for each column for all stocks in the list.
    :param data_list: list of dataframes of stock data
    :return: dataframe of stock data with added columns for overall averages
    """
    # Concatenate all dataframes
    concatenated_df = pd.concat(data_list)
    concatenated_df.drop(columns=['Company', 'Ticker', 'Exchange'], inplace=True)
    concatenated_df.set_index('Date', inplace=True)
    daily_averages = concatenated_df.groupby(concatenated_df.index).mean()
    daily_averages.columns = ['All Avg ' + col for col in daily_averages.columns]
    daily_averages.drop(columns=['All Avg index'], inplace=True)

    return daily_averages


def exchange_averages(data_list: [pd.DataFrame]) -> [pd.DataFrame]:
    """
    Compute the average values for each column for all stocks in the exchange.
    :param data_list: list of dataframes of stock data
    :return: list of dataframes of stock data with added columns for exchange averages
    """
    # Concatenate all dataframes
    concatenated_df = pd.concat(data_list)

    # Group by 'Date' and 'Exchange'
    grouped = concatenated_df.groupby(['Date', 'Exchange'])

    # Calculate the average of the required columns
    averages = grouped[['Open', 'High', 'Low', 'Close', 'Volume']].mean()
    averages.columns = ['Exchange Avg ' + col for col in averages.columns]

    # Create a dictionary of dataframes for each exchange
    exchange_dfs = {exchange: df.reset_index() for exchange, df in averages.groupby(level='Exchange')}

    return exchange_dfs


def add_exchange_averages_to_stocks(data_list: [pd.DataFrame], exchange_dfs: dict) -> [pd.DataFrame]:
    """
    Add exchange averages to each stock in the data list.

    :param data_list: list of dataframes of individual stock data.
    :param exchange_dfs: dictionary of dataframes containing exchange averages.
    :return: list of dataframes with added exchange average columns.
    """
    updated_data_list = []

    for stock_df in data_list:
        exchange_name = stock_df['Exchange'].iloc[0]

        # Get the corresponding exchange average dataframe
        exchange_avg_df = exchange_dfs.get(exchange_name, None)

        updated_df = pd.merge(stock_df, exchange_avg_df, how='left', on=['Date', 'Exchange'])
        updated_data_list.append(updated_df)

    return updated_data_list


def add_overall_averages_to_stocks(data_list: [pd.DataFrame], average_dfs: [pd.DataFrame]):
    """
    Add overall averages to each stock in the data list.

    :param data_list: list of dataframes of individual stock data.
    :param average_dfs: list of dataframes containing overall averages.
    :return: list of dataframes with added overall average columns.
    """
    updated_data_list = []

    for stock_df in data_list:
        updated_df = pd.merge(stock_df, average_dfs, how='left', on='Date')
        updated_data_list.append(updated_df)

    return updated_data_list
