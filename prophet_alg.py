import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from prophet import Prophet

from data_handling.feature_engineering import day_average
from evaluation.evaluation import calculate_standard_metrics, direction_evaluation, calculate_accuracy, simulate_trading


def run_prophet(df: pd.DataFrame, custom_season: int = None) -> [dict]:
    """
    Runs the Prophet model on a given DataFrame to predict future stock prices.

    This function preprocesses the input DataFrame, splits it into training and testing datasets,
    builds and fits a Prophet model with optional custom seasonality, and evaluates the model's predictions.
    It returns a dictionary containing various performance metrics.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing stock market data. It must include the columns:
      'Date', 'Company', 'Ticker', 'Exchange', 'Open', 'High', 'Low', 'Close', and 'Volume'.
    - custom_season (int, optional): The number of days for the custom seasonality period.
      If None, no custom seasonality is added. Default is None.

    Returns:
    - dict: A dictionary containing the following keys and their corresponding values:
      - 'RMSE' (float): Root Mean Squared Error of the model's predictions.
      - 'MAE' (float): Mean Absolute Error of the model's predictions.
      - 'MAPE' (float): Mean Absolute Percentage Error of the model's predictions.
      - 'RSQ' (float): R-squared value indicating the goodness of fit.
      - 'Accuracy' (float): Accuracy of the model's directional predictions.
      - 'Precision' (float): Precision of the model's directional predictions.
      - 'Net Gain' (float): Net gain from simulated trading based on model predictions.
      - 'Avg Profit' (float): Average profit per trade from simulated trading.
      - 'Net Gain Percent' (float): Net gain percentage from simulated trading.

    The function internally uses the Prophet model for time series forecasting and several custom functions
    for preprocessing and evaluation:
    - `day_average` for adding day average feature,
    - `calculate_standard_metrics` for basic evaluation metrics,
    - `direction_evaluation` for evaluating the direction of change,
    - `calculate_accuracy` for accuracy and precision calculation,
    - `simulate_trading` for simulating trading based on model predictions.

    Example:
    ```python
    import pandas as pd
    df = pd.read_csv('stock_data.csv')
    results = run_prophet(df)
    print(results)
    ```

    Note:
    The function assumes that 'Close' column in the input DataFrame is the target variable for prediction,
    and the next day's closing price is predicted. The function also modifies the input DataFrame by adding
    new columns and dropping unnecessary ones. Ensure that the DataFrame is in the correct format before calling this function.
    """

    # preprocessing
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['y'] = df['Close'].shift(-1)
    df['ds'] = df['Date']
    df = day_average(df)
    df = df.drop(columns=['Date', 'Company', 'Ticker', 'Exchange', 'Close'], inplace=False)

    split = round(len(df) * 0.8)

    train_df = df.iloc[:split-100]
    test_df = df.iloc[split-100:]

    # model
    model = Prophet(daily_seasonality=True, weekly_seasonality=False, yearly_seasonality=False)
    model.add_regressor('Open')
    model.add_regressor('High')
    model.add_regressor('Low')
    model.add_regressor('Volume')
    model.add_regressor('Day Average')

    if custom_season is not None:
        model.add_seasonality(name='custom_season', period=custom_season, fourier_order=5)

    model.fit(train_df)

    event_horizons = [5, 10, 20, 50, 100]
    metrics_dicts = []

    for horizon in event_horizons:
        if horizon > len(test_df['y'].values):
            print("Warning: Event horizon is larger than the test set size. Default test set size is 20% of the " +
                  f"dataset. Current horizon is: {len(test_df['y'].values)}.")
        else:
            test_horizon_df = test_df.iloc[:horizon]

        future = test_horizon_df.drop('y', axis=1)

        forecast = model.predict(future)
        test_horizon_df['Predicted Close'] = forecast['yhat'].values

        # evaluation
        true_series = test_horizon_df['y'].iloc[:-1]
        pred_series = test_horizon_df['Predicted Close'].shift(1).iloc[1:]

        metrics_dict = {}
        rmse, mae, mape, rsq, ev = calculate_standard_metrics(true_series, pred_series)
        metrics_dict['RMSE'] = rmse
        metrics_dict['MAE'] = mae
        metrics_dict['MAPE'] = mape
        metrics_dict['RSQ'] = rsq

        direction_df = direction_evaluation(test_horizon_df, 'y', 'Predicted Close')
        accuracy, precision = calculate_accuracy(direction_df, 'y', 'Predicted Close')
        metrics_dict['Accuracy'] = accuracy
        metrics_dict['Precision'] = precision

        net_gain, avg_profit, net_gain_percent = simulate_trading(direction_df, 10000, 'y', 'Predicted Close')
        metrics_dict['Net Gain'] = net_gain
        metrics_dict['Avg Profit'] = avg_profit
        metrics_dict['Net Gain Percent'] = net_gain_percent

        metrics_dicts.append(metrics_dict)

    return metrics_dicts
