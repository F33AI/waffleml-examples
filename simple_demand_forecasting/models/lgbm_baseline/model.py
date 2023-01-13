""" LightGBM baseline model """

import logging
from pathlib import Path
from typing import Any, Dict

from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline

from utils import export_sklearn, predict_sklearn, read_train_test
from evaluation import evaluate_preds
from featurization import featurize


def get_model_name():
    import os.path

    model_name = os.path.basename(os.path.dirname(__file__))
    return model_name


logger = logging.getLogger(get_model_name())
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

TARGET = "sales"
FEATURES = ["store", "item", "sales", "year", "month", "week_of_year",
            "day_of_week", "year_progress_pct", "month_progress_pct"]


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

    logger.info("Load & transform data...")
    X_trn = read_train_test(training_data_path)
    X_trn = featurize(X_trn)
    y_trn = X_trn[TARGET]
    X_trn.drop([TARGET], axis=1, inplace=True)
    logger.info("done. Training model...")

    model = Pipeline([
        ("lgbm", LGBMRegressor(**(hyperparameters["ml"]),
                               objective="regression", metric="mae", verbosity=1)
         )
    ])
    model.fit(
        X_trn, y_trn,
        lgbm__eval_metric=["mae", "rmse"],
        lgbm__eval_set=[(X_trn, y_trn)],)

    logger.info("done. Exporting model...")
    output_file = export_sklearn((model, hyperparameters))
    logger.info("done.")
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
    output_file = predict_sklearn(test_data_path, model_path,
                                  TARGET, FEATURES, logger)

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

    return evaluate_preds(ground_truth_path, predictions_path)
