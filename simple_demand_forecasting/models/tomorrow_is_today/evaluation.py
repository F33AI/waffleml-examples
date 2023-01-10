import yaml
from pathlib import Path
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error


def read_train_test(path:str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",")
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(by=["store", "item", "date"], ascending=True, inplace=True)
    return df


def read_preds(path:str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",")
    return df


def ground_truth(ground_truth_path: str, target: str) -> list:
    return read_train_test(ground_truth_path)[target].to_list()


def predictions(predictions_path: str) -> list:
    return read_preds(predictions_path)["prediction"].to_list()


def mae(ground_truth: list, predictions: list) -> float:
    return float(mean_absolute_error(ground_truth, predictions))


def mse(ground_truth: list, predictions: list) -> float:
    return float(mean_squared_error(ground_truth, predictions))


def evaluate_preds(ground_truth_path, predictions_path):
    gt = ground_truth(ground_truth_path, target="sales")
    pred = predictions(predictions_path)
    metrics = {"mse": mse(gt, pred),
               "mae": mae(gt, pred)}
    output_file = "metrics.yaml"
    with open(output_file, "w") as outfile:
        yaml.dump(metrics, outfile, default_flow_style=True)

    return Path.cwd() / output_file
