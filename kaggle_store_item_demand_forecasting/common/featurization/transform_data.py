from typing import List, Tuple

import pandas as pd
from hamilton import driver

from . import (ftrs_autoregression, ftrs_calendar, ftrs_common_prep, ftrs_test,
               ftrs_train)

def select_features(dr, features) -> List[str]:
    all_possible_outputs = dr.list_available_variables()
    desired_outputs = [
        o.name for o in all_possible_outputs
        if o.tags.get("stage") == "product"]

    assert all([(f in desired_outputs) for f in features]), f"Features {[(f not in desired_outputs) for f in features]} are not existing."

    if len(features) > 0:
        return features

    return desired_outputs


def prepare_train(data_path:str, features:List[str]=[]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    config = {
        "data_path": data_path,
        "train_stub_history_days": 31,
        "target": "sales",
        "phase": "train",
    }

    dr = driver.Driver(config, ftrs_train, ftrs_autoregression, ftrs_calendar, ftrs_common_prep)
    features = select_features(dr, features)
    train_df = dr.execute(features)
    train_stub = dr.execute(["train_stub"])
    return (train_df.drop(["sales"], axis=1), train_df["sales"], train_stub)

def prepare_test(data_path:str, train_stub:pd.DataFrame, features:List[str]=[]) -> pd.DataFrame:
    config = {
        "data_path": data_path,
        "train_stub": train_stub,
        "phase": "test"
    }

    dr = driver.Driver(config, ftrs_test, ftrs_autoregression, ftrs_calendar, ftrs_common_prep)
    features = select_features(dr, features)

    df = dr.execute(features)
    return(df)

if __name__=="__main__":
    train, trn_stub = prepare_train("../../experiments_kaggle-store-item-demand-forecasting_folds_perf_1_hyper_2_train.csv")
    print(prepare_test("../../experiments_kaggle-store-item-demand-forecasting_folds_perf_1_hyper_2_test.csv", trn_stub))
