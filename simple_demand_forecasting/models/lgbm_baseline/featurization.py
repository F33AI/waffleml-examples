import numpy as np
import pandas as pd


# Calendar features
def create_calendar_feats(df: pd.DataFrame):
    df["year"] = df.date.dt.year
    df["month"] = df.date.dt.month
    df["day"] = df.date.dt.day
    df["day_of_week"] = df.date.dt.dayofweek
    df["week_of_year"] = df.date.dt.isocalendar().week
    df["week_of_year"] = df["week_of_year"].astype("int64")
    df.drop(["date"], axis=1, inplace=True)
    return df


# Sales lag features
def create_sales_lag_feats(df, gpby_cols, target_col, lags):
    gpby = df.groupby(gpby_cols)
    for i in lags:
        df['_'.join([target_col, 'lag', str(i)])] = gpby[target_col].shift(i).values
    return df


def featurize(df: pd.DataFrame):
    df = create_sales_lag_feats(df, gpby_cols=['store', 'item'], target_col='sales',
                                lags=[1, 2, 3])
    df.drop(["date"], axis=1, inplace=True)
    return df




