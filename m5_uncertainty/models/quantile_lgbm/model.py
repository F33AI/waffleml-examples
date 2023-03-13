# Copyright (C) Fourteen33 Inc. - All Rights Reserved

""" Template file for model.py """

from typing import Any, Dict

import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from lightgbm import LGBMRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error


def read_data(path: str):
    x = pd.read_csv(path)
    y = x["sales_next_4"]
    #x.drop(["row_idx", "sales", "date"], axis=1, inplace=True)
    return x, y


def read_transform_data(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    en1 = ['nan', 'SuperBowl', 'ValentinesDay', 'PresidentsDay', 'LentStart',
           'LentWeek2', 'StPatricksDay', 'Ramadan starts', 'OrthodoxEaster',
           'Pesach End', 'Cinco De Mayo', "Mother's day", 'MemorialDay',
           'NBAFinalsStart', 'NBAFinalsEnd', "Father's day", 'IndependenceDay',
           'Purim End', 'Eid al-Fitr', 'LaborDay', 'ColumbusDay', 'NewYear',
           'EidAlAdha', 'VeteransDay', 'Thanksgiving', 'Chanukah End', 'Easter',
           'OrthodoxChristmas', 'MartinLutherKingDay', 'Christmas', 'Halloween']

    month = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    year = [2011, 2012, 2013, 2014, 2015, 2016]
    state = ['WI', 'CA', 'TX']
    cat = ['FOODS', 'HOBBIES', 'HOUSEHOLD']
    dept = ['HOBBIES_1', 'HOBBIES_2', 'HOUSEHOLD_1', 'HOUSEHOLD_2', 'FOODS_1',
            'FOODS_2', 'FOODS_3']
    store = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1',
             'WI_2', 'WI_3']
    en2 = ['nan', 'Easter', 'Cinco De Mayo', 'OrthodoxEaster', "Father's day"]
    et1 = ['nan', 'Sporting', 'Cultural', 'National', 'Religious']
    et2 = ['nan', 'Cultural', 'Religious']
    ohe = OneHotEncoder(categories=[month, year, en1, et1, en2, et2, state,
                                    store, dept, cat], sparse_output=False)

    df.fillna('nan', inplace=True)
    df2 = pd.DataFrame(ohe.fit_transform(
        df[['month', 'year', 'event_name_1', 'event_type_1', 'event_name_2',
            'event_type_2', 'state_id', 'store_id', 'dept_id', 'cat_id']]))
    cont_cols = ["last_30_sales_avg","last_30_sales_var","last_30_price_avg",
                 "last_7_sales_avg","last_7_sales_var","last_7_price_avg",
                 "last_7_sales_ratio","week_ago_sales","ystd_sales","ystd_price",
                 "next_4_price_avg","snap"]
    x = np.hstack((df[cont_cols], df2))
    y = df["sales_next_4"]
    return x, y


def train(training_data_path: str, valid_data_path: str,
          hyperparameters: Dict[str, Any]) -> str:
    """ A function that creates a model and performs training.

    Args:
        training_data_path: path to a file on the local filesystem
                            containing training examples
        valid_data_path: path to a file on the local filesystem
                            containing validation examples
        hyperparameters: dictionary (group -> key -> value) containing
                            passed hyper-parameters
    Returns:
        model_path:        path to a file on the local filesystem containing
                           saved weights
    """

    x_train, y_train = read_transform_data(training_data_path)
    x_val, y_val = read_transform_data(valid_data_path)

    lgbm = LGBMRegressor(objective="quantile", alpha=0.05, random_state=42)
    lgbm.fit(x_train, y_train, eval_set=[(x_val, y_val)])

    output_file = "model.joblib"
    joblib.dump(lgbm, output_file)

    return Path.cwd() / output_file


def predict(test_data_path: str, model_path: str) -> str:
    """ A function that creates predictions on test data.

    Args:
        test_data_path: path to a file on the local filesystem
                        containing test examples
        model_path:     path to model file on the local filesystem
                        that should be used to make predictions
    Returns:
        artifact_path:  path to a file on the local filesystem containing
                        predictions made by the model
    """
    x_test, _ = read_transform_data(test_data_path)

    model = joblib.load(model_path)
    y_pred = model.predict(x_test)
    x_test["prediction"] = y_pred

    results = x_test[["prediction"]]
    output_file = "output.csv"
    results.to_csv(output_file)

    return Path.cwd() / output_file


def evaluate(ground_truth_path: str, predictions_path: str) -> str:
    """ ...

    Args:
        ground_truth_path: path to a file on the local filesystem
                           containing test examples
        predictions_path:  path to model file on the local filesystem
                           that should be used to make predictions
    Returns:
        artifact_path:     path to a file on the local filesystem containing
                           output metrics
    """
    _, y_gt = read_data(ground_truth_path)
    df_pred = pd.read_csv(predictions_path)
    y_pred = df_pred["prediction"]

    mse = mean_squared_error(y_gt, y_pred)
    mae = mean_absolute_error(y_gt, y_pred)

    metrics = {"mse": mse,
               "mae": mae}
    output_file = "metrics.yaml"

    with open(output_file, "w") as outfile:
        yaml.dump(metrics, outfile, default_flow_style=True)

    return Path.cwd() / output_file


