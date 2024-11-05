import pandas as pd
from prophet import Prophet

from feature_engineering import day_average
from evaluation import calculate_standard_metrics, direction_evaluation, calculate_accuracy, simulate_trading


def run_moving_average(df: pd.DataFrame, truth_column: str = 'Close', window_size: int = 20) -> dict:
    """
    Run a moving average 'model' on the data and evaluate its performance.
    :param df: stock data
    :param truth_column: column of the true values
    :param window_size: window size for the moving average
    :return: dictionary of metrics
    """

    # preprocessing
    df = df.copy()
    split = round(len(df) * 0.8)
    test_df = df.iloc[split:]

    # model?
    test_df['Predicted Close'] = test_df[truth_column].rolling(window=window_size).mean()
    n_day_mean = df[truth_column].iloc[0:window_size].mean()
    test_df['Predicted Close'] = test_df['Predicted Close'].fillna(n_day_mean)

    # evaluation
    true_series = test_df['Close'].iloc[:-1]
    pred_series = test_df['Predicted Close'].shift(1).iloc[1:]

    metrics_dict = {}
    rmse, mae, mape, rsq, ev = calculate_standard_metrics(true_series, pred_series)
    metrics_dict['RMSE'] = rmse
    metrics_dict['MAE'] = mae
    metrics_dict['MAPE'] = mape
    metrics_dict['RSQ'] = rsq

    direction_df = direction_evaluation(test_df, 'Close', 'Predicted Close')
    accuracy, precision = calculate_accuracy(direction_df, 'Close', 'Predicted Close')
    metrics_dict['Accuracy'] = accuracy
    metrics_dict['Precision'] = precision

    net_gain, avg_profit, net_gain_percent = simulate_trading(direction_df, 10000, 'Close', 'Predicted Close')
    metrics_dict['Net Gain'] = net_gain
    metrics_dict['Avg Profit'] = avg_profit
    metrics_dict['Net Gain Percent'] = net_gain_percent

    return metrics_dict