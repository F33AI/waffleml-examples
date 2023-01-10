import joblib
import pandas as pd
from pathlib import Path
from typing import Any, Dict

from evaluation import evaluate_preds, read_train_test


def train(training_data_path: str, valid_data_path: str,
          hyperparameters: Dict[str, Any]) -> str:
    """ ...

    Args:
        training_data_path: path to a file on the local filesystem
                           containing test examples
        valid_data_path: path to a file on the local filesystem
                           containing validation examples
        hyperparameters:  path to model file on the local filesystem
                           that should be used to make predictions
    Returns:
        artifact_path:     path to a file on the local filesystem containing
                           output metrics
    """

    df = read_train_test(training_data_path)

    all_items = pd.unique(df["item"]).tolist()
    all_stores = pd.unique(df["store"]).tolist()

    last_train_day = df["date"].max()
    all_pairs = {
        (s, i): df[(df["date"] >= last_train_day) & (df["store"] == s) &
                   (df["item"] == i)]["sales"].tolist()[0]
        for s in all_stores for i in all_items
    }

    output_file = "model.joblib"

    joblib.dump(all_pairs, output_file)
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

    model = joblib.load(model_path)
    df = read_train_test(test_data_path)

    fcsts = []
    mean_sales = sum(model.values())/len(model)
    for _tst_idx, tst_row in df.iterrows():
        fcst = model.get((tst_row["store"], tst_row["item"]), mean_sales)
        fcsts.append(fcst)

    df["prediction"] = fcsts

    results = df.drop(df.columns.difference(["prediction"]), axis=1)

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

    return evaluate_preds(ground_truth_path, predictions_path)
