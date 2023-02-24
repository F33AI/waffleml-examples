from typing import Any, Dict

import pandas as pd
from pandas import DataFrame


def read_dataset(path:str, features_schema:Dict[str, Any]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for dt_col in features_schema["ftr_datetime"]:
        df[dt_col] = pd.to_datetime(df[dt_col])

    return df

def read_target(path:str, ftr_target:str) -> pd.Series:
    df = pd.read_parquet(path, columns=[ftr_target])

    return df[ftr_target]
