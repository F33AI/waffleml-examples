import pandas as pd
from pandas import DataFrame

from .read_data import read_train_test


def sales_data_set_test(data_path:str) -> DataFrame:
    df = read_train_test(data_path)
    return df

def first_test_day(sales_data_set_test:pd.DataFrame) -> pd.Timestamp:
    res = sales_data_set_test["date"].min()
    return res

def strip_train_stub(train_stub:pd.DataFrame, first_test_day:pd.Timestamp) -> pd.DataFrame:
    return train_stub[train_stub["date"] >= first_test_day - pd.Timedelta("7 days")]

def sales_data_set(sales_data_set_test:pd.DataFrame, strip_train_stub:DataFrame) -> pd.DataFrame:
    df = pd.concat([sales_data_set_test, strip_train_stub], axis=0)

    df.sort_values(by=["store", "item", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def output_idx(sales_data_set:pd.DataFrame, first_test_day:pd.Timestamp) -> pd.Index:
    return sales_data_set[sales_data_set["date"] >= first_test_day].index
