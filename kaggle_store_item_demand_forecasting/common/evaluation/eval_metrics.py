from sklearn.metrics import mean_absolute_error, mean_squared_error
from hamilton.function_modifiers import (tag)

from .read_data import read_train_test, read_preds


def ground_truth(ground_truth_path:str, target:str) -> list:
    return read_train_test(ground_truth_path)[target].to_list()

def predictions(predictions_path:str) -> list:
    return read_preds(predictions_path)["prediction"].to_list()

@tag(stage="product")
def mae(ground_truth: list, predictions: list) -> float:
    return float(mean_absolute_error(ground_truth, predictions))

@tag(stage="product")
def mse(ground_truth: list, predictions: list) -> float:
    return float(mean_squared_error(ground_truth, predictions))

