from data_handling.feature_engineering import day_average, previous_day
from evaluation.evaluation import calculate_standard_metrics, direction_evaluation, calculate_accuracy, simulate_trading

import numpy as np
import pandas as pd
import os
import logging
pd.options.mode.chained_assignment = None  # default='warn'

import lightning.pytorch as pl
from pytorch_forecasting import DeepAR, TimeSeriesDataSet
from pytorch_forecasting.metrics import MultivariateDistributionLoss, NormalDistributionLoss

pl.seed_everything(42)

logging.basicConfig(level=logging.INFO, filename='deepar.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')


def run_deepar(data: pd.DataFrame, max_prediction_length: int, max_encoder_length: int) -> [dict]:
    """
    Runs the DeepAR algorithm on the input DataFrame and returns a list of dictionaries containing the metrics for each event horizon.

    :param df: The input DataFrame containing stock market data. It must include the columns:
    'Date', 'Company', 'Ticker', 'Exchange', 'Open', 'High', 'Low', 'Close', and 'Volume'.

    :return: dict: A list of dictionaries containing the metrics for each event horizon.
      - 'RMSE' (float): Root Mean Squared Error of the model's predictions.
      - 'MAE' (float): Mean Absolute Error of the model's predictions.
      - 'MAPE' (float): Mean Absolute Percentage Error of the model's predictions.
      - 'RSQ' (float): R-squared value indicating the goodness of fit.
      - 'Accuracy' (float): Accuracy of the model's directional predictions.
      - 'Precision' (float): Precision of the model's directional predictions.
      - 'Net Gain' (float): Net gain from simulated trading based on model predictions.
      - 'Avg Profit' (float): Average profit per trade from simulated trading.
      - 'Net Gain Percent' (float): Net gain percentage from simulated trading.
    """

    logging.info("Preprocessing data for DeepAR")
    # minor preprocessing
    data = day_average(data)
    data = previous_day(data)
    data['Date'] = pd.to_datetime(data['Date'])
    original_df = data.copy()
    data.drop(['Ticker', 'Exchange'], axis=1, inplace=True)
    data.reset_index(inplace=True)

    logging.info("Splitting data into training and validation sets")
    # train / test split
    batch_size = 32
    length = len(data)
    n_batches = length // batch_size
    train_n_batches = round(n_batches * 0.7)
    test_n_batches = n_batches - train_n_batches
    test_size = test_n_batches * batch_size
    if test_n_batches % batch_size != 0:
        test_size -= test_n_batches % batch_size
    if test_size < max_prediction_length:
        print(f"Test size is too small for largest event horizon for: {data['Company'].iloc[0]}")
    split = train_n_batches * batch_size

    # encoder errors occur when there isnt enough data
    # I dont think this is the correct check but I'm still working on figuring out how to correctly check
    if test_size < max_prediction_length + max_encoder_length:
        max_encoder_length = 30  # this is the minimum encoder length, 6 weeks instead of 12 (5 day trade weeks)

    logging.info("Number of batches: " + str(n_batches))
    logging.info("Train batches: " + str(train_n_batches))
    logging.info("Test batches: " + str(test_n_batches))
    logging.info("Max encoder length: " + str(max_encoder_length))

    logging.info("Creating TimeSeriesDataSet")
    # create training and validation sets and dataloaders
    training = TimeSeriesDataSet(
        data[lambda x: x.index < split],
        time_idx="index",
        target="Close",
        group_ids=["Company"],  # column name that identifies a time series
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=['Company'],
        time_varying_known_reals=["Day Average", "Previous Close", "Previous Open", "Previous High", "Previous Low",
                                  "Previous Volume"],
        time_varying_unknown_reals=["Close"]  # this is the target
    )
    logging.info(f"Dataset created for {data['Company'].iloc[0]}")
    logging.info("Creating dataloaders")
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=3,
                                              persistent_workers=True)
    val_dataloader = TimeSeriesDataSet.from_dataset(training, data.iloc[split:split + test_size],
                                                    min_prediction_idx=split,
                                                    stop_randomization=True).to_dataloader(train=False,
                                                                                           batch_size=batch_size,
                                                                                           num_workers=3,
                                                                                           persistent_workers=True)
    logging.info("Creating model")
    # create model and train
    loss = MultivariateDistributionLoss()
    deepar = DeepAR.from_dataset(dataset=training,
                                 learning_rate=5e-4,
                                 hidden_size=12,
                                 rnn_layers=3,
                                 optimizer="AdamW",
                                 loss=loss
                                 )
    trainer = pl.Trainer(
        accelerator='auto',
        max_epochs=100,
        gradient_clip_val=0.1,
        enable_checkpointing=True
    )

    logging.info("Fitting model")
    trainer.fit(
        deepar,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_deepar = DeepAR.load_from_checkpoint(best_model_path)

    logging.info("Generating predictions")
    # generating predictions for test set
    predict_df = data.iloc[split:]
    predictions = best_deepar.predict(predict_df).cpu().numpy().reshape(-1)

    # evaluation for each event horizon
    event_horizons = [1, 5, 10, 20, 50, 100]
    metrics = []  # this will be a list of dictionaries containing the metrics for each event horizon
    try:
        for horizon in event_horizons:
            logging.info("Calculating metrics for event horizon: " + str(horizon))
            val_data = data.iloc[split:split + horizon]

            val_data['Predicted Close'] = predictions[:horizon]
            metrics_dict = {}
            rmse, mae, mape, rsq, ev = calculate_standard_metrics(val_data['Close'], val_data['Predicted Close'])
            metrics_dict['RMSE'] = rmse
            metrics_dict['MAE'] = mae
            metrics_dict['MAPE'] = mape
            metrics_dict['RSQ'] = rsq

            direction_df = direction_evaluation(val_data, 'Close', 'Predicted Close')
            accuracy, precision = calculate_accuracy(direction_df, 'Close', 'Predicted Close')
            metrics_dict['Accuracy'] = accuracy
            metrics_dict['Precision'] = precision

            net_gain, avg_profit, net_gain_percent = simulate_trading(direction_df, 10000)
            metrics_dict['Net Gain'] = net_gain
            metrics_dict['Avg Profit'] = avg_profit
            metrics_dict['Net Gain Percent'] = net_gain_percent
            metrics.append(metrics_dict)

        horizons_metrics_list = []
        for i in range(len(event_horizons)):
            row = {
                'Company': original_df['Company'][0],
                'Ticker': original_df['Ticker'][0],
                'Exchange': original_df['Exchange'][0],
                'Event Horizon': event_horizons[i],
                'RMSE': metrics[i]['RMSE'],
                'MAE': metrics[i]['MAE'],
                'MAPE': metrics[i]['MAPE'],
                'RSQ': metrics[i]['RSQ'],
                'Accuracy': metrics[i]['Accuracy'],
                'Precision': metrics[i]['Precision'],
                'Net Gain': metrics[i]['Net Gain'],
                'Avg Profit': metrics[i]['Avg Profit'],
                'Net Gain Percent': metrics[i]['Net Gain Percent']
            }
            horizons_metrics_list.append(row)

    except Exception as e:
        print(f"Error: {e} when calculating metrics for: {data['Company'].iloc[0]}")

    return horizons_metrics_list


def run_deepar_all_stocks(data_list: [pd.DataFrame]) -> [pd.DataFrame]:
    """
    Runs DeepAR on all stocks in the input DataFrame and returns a list event horizon metrics dataframes for each stock.
    :param df: dataframe containing stock market data for all stocks
    :return: list of dataframes containing event horizon metrics for each stock.

    Note: The returned list of dataframes is a list of all of the metrics dataframes that run_deepar normally generates
    one of.
    """

    logging.info("Preprocessing data for DeepAR")
    # minor preprocessing
    company_names = []
    lengths = []
    for data in data_list:
        data = preprocess_stock(data)
        lengths.append(len(data))
        company_names.append(data['Company'].iloc[0])

    smallest_length = min(lengths)
    max_encoder_length = round(0.1 * smallest_length)
    if max_encoder_length > 60:
        max_encoder_length = 60
    elif max_encoder_length < 60:
        max_encoder_length = 30
    # note: the encoder lengths have a huge impact on the performance of the model
    # a big issue I've had is some stocks do not have enough data
    # for a 12 or 6 week encoder length, those stocks are listed in the stocks_without_enough_data folder
    max_prediction_length = 100

    logging.info("Smallest dataset length: " + str(smallest_length))
    logging.info("Max encoder length: " + str(max_encoder_length))


    # train / test split
    batch_size = 32
    length = smallest_length
    n_batches = length // batch_size
    train_n_batches = round(n_batches * 0.7)
    test_n_batches = n_batches - train_n_batches
    test_size = test_n_batches * batch_size
    if test_n_batches % batch_size != 0:
        test_size -= test_n_batches % batch_size
    if test_size < max_prediction_length:
        print(f"Test size is too small for largest event horizon for: {data['Company'].iloc[0]}")
    split = train_n_batches * batch_size

    logging.info("Number of batches: " + str(n_batches))
    logging.info("Train batches: " + str(train_n_batches))
    logging.info("Test batches: " + str(test_n_batches))
    logging.info("Max encoder length: " + str(max_encoder_length))

    # close_columns, stock_previous_day_columns, data_list = company_name_columns(data_list)
    dataset = pd.concat(data_list, axis=0)
    dataset.drop(['Open', 'High', 'Low', 'Volume'], axis=1, inplace=True)

    # create training and validation sets and dataloaders
    training = TimeSeriesDataSet(
        dataset.iloc[:split],
        time_idx="index",
        target='Close',
        group_ids=["Company"],  # column name that identifies a time series
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=["Close Day Average", "Previous Close", "Previous Open", "Previous High", "Previous Low",
                                  "Previous Volume"],
        time_varying_unknown_reals=['Close']  # this includes the target
    )
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=3,
                                              persistent_workers=True)
    val_dataloader = TimeSeriesDataSet.from_dataset(training, dataset.iloc[split:split + test_size],
                                                    min_prediction_idx=split,
                                                    stop_randomization=True, predict=True).to_dataloader(train=False,
                                                                                           batch_size=batch_size,
                                                                                           num_workers=3,
                                                                                           persistent_workers=True)
    # create model and train
    loss = NormalDistributionLoss()
    deepar = DeepAR.from_dataset(dataset=training,
                                 learning_rate=5e-4,
                                 hidden_size=12,
                                 rnn_layers=3,
                                 optimizer="AdamW",
                                 loss=loss
                                 )
    trainer = pl.Trainer(
        accelerator='auto',
        max_epochs=100,
        gradient_clip_val=0.1,
        enable_checkpointing=True
    )

    trainer.fit(
        deepar,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_deepar = DeepAR.load_from_checkpoint(best_model_path)

    # now I need to figure out what form the predictions are in so I can actually generate metrics for each stock
    # so I probably need to remove the numpy().reshape(-1)
    predictions = best_deepar.predict(val_dataloader, mode='prediction').cpu()
    print("Predictions: ", predictions)
    # print("Predictions items: ", predictions.items())
    logging.info("Predictions: ", predictions)

    # # evaluation for each event horizon and each stock
    # event_horizons = [1, 5, 10, 20, 50, 100]
    # metrics = []  # this will be a list of dictionaries containing the metrics for each event horizon
    # try:
    #     for horizon in event_horizons:
    #         val_data = data.iloc[split:split + horizon]
    #
    #         val_data['Predicted Close'] = predictions[:horizon]
    #         metrics_dict = {}
    #         rmse, mae, mape, rsq, ev = calculate_standard_metrics(val_data['Close'], val_data['Predicted Close'])
    #         metrics_dict['RMSE'] = rmse
    #         metrics_dict['MAE'] = mae
    #         metrics_dict['MAPE'] = mape
    #         metrics_dict['RSQ'] = rsq
    #
    #         direction_df = direction_evaluation(val_data, 'Close', 'Predicted Close')
    #         accuracy, precision = calculate_accuracy(direction_df, 'Close', 'Predicted Close')
    #         metrics_dict['Accuracy'] = accuracy
    #         metrics_dict['Precision'] = precision
    #
    #         net_gain, avg_profit, net_gain_percent = simulate_trading(direction_df, 10000)
    #         metrics_dict['Net Gain'] = net_gain
    #         metrics_dict['Avg Profit'] = avg_profit
    #         metrics_dict['Net Gain Percent'] = net_gain_percent
    #         metrics.append(metrics_dict)
    #
    #     horizons_metrics_list = []
    #     for i in range(len(event_horizons)):
    #         row = {
    #             'Company': original_df['Company'][0],
    #             'Ticker': original_df['Ticker'][0],
    #             'Exchange': original_df['Exchange'][0],
    #             'Event Horizon': event_horizons[i],
    #             'RMSE': metrics[i]['RMSE'],
    #             'MAE': metrics[i]['MAE'],
    #             'MAPE': metrics[i]['MAPE'],
    #             'RSQ': metrics[i]['RSQ'],
    #             'Accuracy': metrics[i]['Accuracy'],
    #             'Precision': metrics[i]['Precision'],
    #             'Net Gain': metrics[i]['Net Gain'],
    #             'Avg Profit': metrics[i]['Avg Profit'],
    #             'Net Gain Percent': metrics[i]['Net Gain Percent']
    #         }
    #         horizons_metrics_list.append(row)
    #
    # except Exception as e:
    #     print(f"Error: {e} when calculating metrics for: {data['Company'].iloc[0]}")
    random_return = pd.DataFrame()
    return random_return



def preprocess_stock(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the input DataFrame for use with DeepAR by adding the 'Day Average' and 'Previous Day' columns.
    :param df: stock dataframe 
    :return: stock dataframe with added columns for use with DeepAR
    """

    data = day_average(df)
    data = previous_day(data)
    data['Date'] = pd.to_datetime(data['Date'])
    data.drop(['Ticker', 'Exchange'], axis=1, inplace=True)
    data['index'] = data.index  # add index column for use with TimeSeriesDataSet

    return data

def company_name_columns(data_list: [pd.DataFrame]) -> ([str], [[str]], [pd.DataFrame]):
    """
    Rewrites the columns for each stock to be unique using the company name.

    :param data_list: List of DataFrames, each containing stock market data for a different stock.
    :return: Tuple containing two lists of column names and a list of dataframes:
        - close_columns: List of column names for 'Close' prices for each stock.
        - previous_day_columns: List of column names for 'Previous Day' metrics for each stock.
        - data_list_new: List of DataFrames, each containing stock market data for a different stock with unique column names.
    """
    data_list_new = []
    close_columns = []
    previous_day_columns = []

    for data in data_list:
        company_name = data['Company'].iloc[0]  # Replace spaces with underscores for column names
        new_column_names = {col: f'{company_name}-{col}' for col in data.columns if col != 'Company'}
        data.rename(columns=new_column_names, inplace=True)

        close_column = f'{company_name}-Close'
        previous_columns = [f'{company_name}-{metric}' for metric in
                            ['Previous Open', 'Previous Close', 'Previous High', 'Previous Low', 'Previous Volume']]
        close_columns.append(close_column)
        previous_day_columns.append(previous_columns)
        data.drop([f'{company_name}-{metric}' for metric in ['Open', 'High', 'Low', 'Volume']], axis=1, inplace=True)
        data_list_new.append(data)

    return close_columns, previous_day_columns, data_list_new
