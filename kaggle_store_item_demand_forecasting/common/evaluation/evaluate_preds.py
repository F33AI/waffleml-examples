import importlib
from pathlib import Path

import yaml
from hamilton import base, driver
from hamilton.function_modifiers import tag

from . import eval_metrics


def evaluate(ground_truth_path:str, predictions_path:str) -> str:
    config = {
        "ground_truth_path": ground_truth_path,
        "predictions_path": predictions_path,
        "target": "sales"
    }
    adapter = base.SimplePythonGraphAdapter(base.DictResult())
    dr = driver.Driver(config, eval_metrics, adapter=adapter)
    desired_outputs = [
        "mae",
        "mse"
    ]
    metrics = dr.execute(desired_outputs)

    output_file = "metrics.yaml" 
    with open(output_file, "w") as outfile:
        yaml.dump(metrics, outfile, default_flow_style=True)

    return Path.cwd() / output_file