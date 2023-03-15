# Copyright (C) Fourteen33 Inc. - All Rights Reserved

""" Template file for model.py """

from typing import Any, Dict

import yaml
import joblib
import pandas as pd
from pathlib import Path

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from hamilton import driver
from featurize import cat_feats


def read_data(path: str):
    data = pd.read_csv(path)

    dr = driver.Driver(data, cat_feats)
    all_feats = dr.list_available_variables()
    x = dr.execute(all_feats)
    y = x["sales"]
    x.drop(["sales"], inplace=True)

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

    x_train, y_train = read_data(training_data_path)
    x_val, y_val = read_data(valid_data_path)

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
    x_test, _ = read_data(test_data_path)

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


