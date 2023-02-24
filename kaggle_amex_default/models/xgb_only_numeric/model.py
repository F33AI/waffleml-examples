import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
import yaml
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score as auc
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import features.featurize as featurize


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



def predict(test_data_path: str, model_path: str) -> str:
    logger.info("Reading model instance...")
    model_dump = joblib.load(model_path)
    model, _hyperparameters = model_dump

    logger.info("done. Loading test dataset...")
    data = featurize.prepare_test_dataset(test_data_path)

    logger.info("done. Generating predictions ...")
    pred = model.predict_proba(data)
    pred0 = [p[1] for p in pred]
    data.loc[:, "prediction"] = pred0
    results = data.drop(data.columns.difference(["prediction"]), axis=1)

    logger.info("done. Exporting predictions...")
    output_file = "output.csv"
    results.to_csv(output_file)
    logger.info("done")
    return Path.cwd() / output_file


def evaluate(ground_truth_path: str, predictions_path: str) -> str:
    output_file = "metrics.yaml"

    logger.info("Loading ground truth a.k.a target...")
    gt = featurize.prepare_ground_truth_dataset(ground_truth_path)
    logger.info("done. Loading predictions...")
    pred = pd.read_csv(predictions_path)["prediction"].to_list()
    logger.info("done. Evaluating metrics...")

    model_metrics = {
        "log_loss" : float(log_loss(gt, pred)),
        "auc": float(auc(gt, pred))
    }

    logger.info("done. Exporting metrics...")

    with open(output_file, "w") as outfile:
        yaml.dump(model_metrics, outfile, default_flow_style=True)

    logger.info("done.")
    return Path.cwd() / output_file


def train(traininig_data_path: str, valid_data_path: str, hyperparameters: Dict[str, Any]) -> str:
    logger.info("Loading train dataset...")
    trn = featurize.prepare_train_dataset(traininig_data_path)
    val = featurize.prepare_train_dataset(valid_data_path)
    logger.info("done. Training model...")

    model = Pipeline([
        ("xgb",
            XGBClassifier(**(hyperparameters["ml"]),
                objective="binary:logistic",
                tree_method="hist",
                verbosity=1,
                eval_metric="auc",))
    ])

    model.fit(
        X=trn["X"],
        y=trn["target"],
        xgb__eval_set = [(trn["X"], trn["target"]), (val["X"], val["target"])])

    logger.info("done. Saving model instance...")

    output_file = "model.joblib"

    model_dump = (
        model,
        hyperparameters
    )

    joblib.dump(model_dump, output_file)
    logger.info("done")
    return Path.cwd() / output_file
