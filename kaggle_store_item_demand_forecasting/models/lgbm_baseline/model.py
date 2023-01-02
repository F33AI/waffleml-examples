""" LightGBM baseline model """

import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
import numpy as np

from common.featurization.transform_data import prepare_train, prepare_test
from common.evaluation.evaluate_preds import evaluate as eval_pred
from common import utils

def get_model_name():
    import os.path

    model_name = os.path.basename(os.path.dirname(__file__))
    return model_name

logger = logging.getLogger(get_model_name())
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

TARGET="sales"
FEATURES = ["store", "item", "sales", "year", "month", "week_of_year", "day_of_week", "year_progress_pct", "month_progress_pct"]

def train(training_data_path: str, valid_data_path: str, hyperparameters: Dict[str, Any]) -> str:
    """ ...

    Args:
        ground_truth_path: path to a file on the local filesystem
                           contatining test examples
        predictions_path:  path to model file on the local filesystem
                           that should be used to make predictions
    Returns:
        artifact_path:     path to a file on the local filesystem containng
                           output metrics
    """

    logger.info("Load & transform data...")
    X_trn, y_trn, trn_stub = prepare_train(training_data_path, features=FEATURES)
    logger.info("done. Training model...")

    model = Pipeline([
        ("lgbm",
            LGBMRegressor(**(hyperparameters["ml"]),
                objective="regression",
                metric="mae",
                verbosity=1,)
        )
    ])
    model.fit(
        X_trn, y_trn,
        lgbm__eval_metric=["mae", "rmse"],
        lgbm__eval_set=[(X_trn, y_trn)],)

    logger.info("done. Exporting model...")
    output_file = utils.export_sklearn((model, hyperparameters, trn_stub))
    logger.info("done.")
    return Path.cwd() / output_file

def predict(test_data_path: str, model_path: str) -> str:
    """ A function that creates predictions on test data.

    Args:
        test_data_path: path to a file on the local filesystem
                        contatining test examples
        model_path:     path to model file on the local filesystem
                        that should be used to make predictions
    Returns:
        artifact_path:  path to a file on the local filesystem containng
                        predictions made by the model
    """
    output_file = utils.predict_sklearn(test_data_path, model_path, TARGET, FEATURES, logger)

    return Path.cwd() / output_file

def evaluate(ground_truth_path: str, predictions_path: str) -> str:
    """ ...

    Args:
        ground_truth_path: path to a file on the local filesystem
                           contatining test examples
        predictions_path:  path to model file on the local filesystem
                           that should be used to make predictions
    Returns:
        artifact_path:     path to a file on the local filesystem containng
                           output metrics
    """

    return eval_pred(ground_truth_path, predictions_path)
