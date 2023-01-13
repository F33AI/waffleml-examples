import random
import pandas as pd
from typing import List
import logging

import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

FORECAST_HORIZON = 1
GAP_DAYS = 3

def read_data(path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["date"])
    df["date"] = pd.to_datetime(df["date"])
    df["row_idx"] = list(range(df.shape[0]))
    return df

def split_train_valid_test(df, last_train_day, last_valid_day, last_test_day):
    logger.info(f">> split: {last_train_day} :  {last_valid_day} : {last_test_day}")
    return((
                df[df["date"] <= last_train_day].row_idx.to_list(),
                df[(df["date"] > last_train_day) & (df["date"] <= last_valid_day)].row_idx.to_list(),
                df[(df["date"] > last_valid_day) & (df["date"] <= last_test_day)].row_idx.to_list()
        ))

def update_days(last_train_day, last_valid_day, last_test_day):
    one_day_delta = pd.Timedelta(f"{FORECAST_HORIZON} day")
    return (
            last_train_day - one_day_delta,
            last_valid_day - one_day_delta,
            last_test_day - one_day_delta
        )

def walk_days(flds_cnt:int, df:pd.DataFrame):
    one_day_delta = pd.Timedelta(f"{FORECAST_HORIZON} day")
    last_test_day = df["date"].max()
    logger.info(f">>> Last test day = {last_test_day}")
    last_valid_day = last_test_day - one_day_delta*GAP_DAYS
    last_train_day = last_valid_day - one_day_delta*GAP_DAYS

    for fld_idx in range(flds_cnt):
        yield split_train_valid_test(df, last_train_day, last_valid_day, last_test_day)
        last_train_day, last_valid_day, last_test_day = update_days(last_train_day, last_valid_day, last_test_day)

def create_cv_folds(n: int, csv_path: str, train_ratio: float, val_ratio: float,
                    test_ratio: float) -> List[str]:
    """ Splits given CSV file into a `n` groups (folds). Each group should
        have defined three subgroups (train/val/test).

    Args:
        n (int): number of folds
        csv_path (str): path to a CSV file
        train_ratio (float): [0.0 - 1.0]
        val_ratio (float): [0.0 - 1.0]
        test_ratio (float): [0.0 - 1.0]

    Notes:
        * `train_ratio`, `val_ratio` and `test_ratio` are positive and
            sum up to 1.0

    Returns:
        List[Tuple[List[int], List[int], List[int]]: A list of `n` groups where
            each group is a tuple of three lists: train/val/text. Each `list`
            contains indicies of rows that belong to that group.
    """

    groups = []

    logger.info(f"Reading data from {csv_path}...")
    df = read_data(csv_path)
    logger.info(f"done. Generating {n} folds...")
    for (trn_idx, val_idx, tst_idx) in walk_days(n, df):
        groups.append([trn_idx, val_idx, tst_idx])

    logger.info("done")
    return groups