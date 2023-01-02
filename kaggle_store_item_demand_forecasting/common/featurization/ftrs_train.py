import pandas as pd
from pandas import DataFrame

from .read_data import read_train_test


def sales_data_set(data_path:str) -> DataFrame:
    df = read_train_test(data_path)
    return df

def train_stub(sales_data_set:pd.DataFrame, train_stub_history_days:int) -> DataFrame:
    last_train_day = sales_data_set["date"].max()
    train_stub = sales_data_set[sales_data_set["date"] >= last_train_day - pd.Timedelta(f"{train_stub_history_days} days")]
    return train_stub

def output_idx(lag_sales_31:pd.Series) -> pd.Index:
    return lag_sales_31.index
