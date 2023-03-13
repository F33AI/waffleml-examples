import random
import pandas as pd
from typing import List
import logging

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

FORECAST_HORIZON = 12 * 7
GAP_DAYS = 1


def transform_data(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    en1 = ['nan', 'SuperBowl', 'ValentinesDay', 'PresidentsDay', 'LentStart',
           'LentWeek2', 'StPatricksDay', 'Ramadan starts', 'OrthodoxEaster',
           'Pesach End', 'Cinco De Mayo', "Mother's day", 'MemorialDay',
           'NBAFinalsStart', 'NBAFinalsEnd', "Father's day", 'IndependenceDay',
           'Purim End', 'Eid al-Fitr', 'LaborDay', 'ColumbusDay', 'NewYear',
           'EidAlAdha', 'VeteransDay', 'Thanksgiving', 'Chanukah End', 'Easter',
           'OrthodoxChristmas', 'MartinLutherKingDay', 'Christmas', 'Halloween']

    month = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    year = [2011, 2012, 2013, 2014, 2015, 2016]
    state = ['WI', 'CA', 'TX']
    cat = ['FOODS', 'HOBBIES', 'HOUSEHOLD']
    dept = ['HOBBIES_1', 'HOBBIES_2', 'HOUSEHOLD_1', 'HOUSEHOLD_2', 'FOODS_1',
            'FOODS_2', 'FOODS_3']
    store = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1',
             'WI_2', 'WI_3']
    en2 = ['nan', 'Easter', 'Cinco De Mayo', 'OrthodoxEaster', "Father's day"]
    et1 = ['nan', 'Sporting', 'Cultural', 'National', 'Religious']
    et2 = ['nan', 'Cultural', 'Religious']
    ohe = OneHotEncoder(categories=[month, year, en1, et1, en2, et2, state,
                                    store, dept, cat], sparse_output=False)
    # ct = ColumnTransformer([("month", ohe, ["month"]),
    #                        #("year", year_ohe, ["year"])
    #                        ], remainder='passthrough')
    df.fillna('nan', inplace=True)
    df2 = pd.DataFrame(ohe.fit_transform(
        df[['month', 'year', 'event_name_1', 'event_type_1', 'event_name_2',
            'event_type_2', 'state_id', 'store_id', 'dept_id', 'cat_id']]))
    print(df2)


def read_data(path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["idx", "date"])
    df["date"] = pd.to_datetime(df["date"])
    df["rowid"] = list(range(df.shape[0]))
    return df


def split_train_valid_test(df, last_train_day, last_valid_day, last_test_day):
    logger.info(f">> split: {last_train_day} :  {last_valid_day} : {last_test_day}")
    return((
                df[df["date"] <= last_train_day].rowid.to_list(),
                df[(df["date"] > last_train_day) & (df["date"] <= last_valid_day)].rowid.to_list(),
                df[(df["date"] > last_valid_day) & (df["date"] <= last_test_day)].rowid.to_list()
        ))


def update_days(last_train_day, last_valid_day, last_test_day):
    delta = pd.Timedelta(f"{FORECAST_HORIZON} day")
    return (
            last_train_day - delta,
            last_valid_day - delta,
            last_test_day - delta
        )


def walk_days(flds_cnt: int, df: pd.DataFrame):
    days_delta = pd.Timedelta(f"{FORECAST_HORIZON} day")
    last_test_day = df["date"].max()
    logger.info(f">>> Last test day = {last_test_day}")
    last_valid_day = last_test_day - days_delta*GAP_DAYS
    last_train_day = last_valid_day - days_delta*GAP_DAYS

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
            each group is a tuple of three lists: train/val/test. Each `list`
            contains indices of rows that belong to that group.
    """

    groups = []

    logger.info(f"Reading data from {csv_path}...")
    df = read_data(csv_path)
    logger.info(f"done. Generating {n} folds...")
    for (trn_idx, val_idx, tst_idx) in walk_days(n, df):
        groups.append([trn_idx, val_idx, tst_idx])

    logger.info("done")

    #for g in groups:
    #    print(len(g[0]), len(g[1]), len(g[2]) )
    #print(groups[0][1])

    return groups


#if __name__ == "__main__":
#    create_cv_folds(5, "train_mondays.csv", 0.7, 0.2, 0.1)
