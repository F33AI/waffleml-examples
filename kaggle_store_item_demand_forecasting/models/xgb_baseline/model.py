""" LightGBM baseline model """

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from common.evaluation.evaluate_preds import evaluate as eval_pred
from common.featurization.transform_data import prepare_train
from common import utils
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


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


def train(training_data_path: str, hyperparameters: Dict[str, Any]) -> str:
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
    X_trn, y_trn, trn_stub = prepare_train(training_data_path, features = FEATURES)
    logger.info("done. Training model...")

    model = Pipeline([
        ("xgb",
            XGBRegressor(**(hyperparameters["ml"]),
                objective="reg:squarederror",
                tree_method="hist",
                verbosity=1,
                eval_metric=["mae", "rmse"],)
        )
    ])
    model.fit(
        X_trn, y_trn,
        xgb__eval_set=[(X_trn, y_trn)],)

    logger.info("done. Exporting model...")

    output_file = utils.export_sklearn((
        model,
        hyperparameters,
        trn_stub
    ))

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

