import pandas as pd
from hamilton.function_modifiers import (extract_columns,
                                         tag)
from pandas import DataFrame
from pandas.core.groupby import DataFrameGroupBy


def grp_date_store_item(sales_data_set:DataFrame) -> DataFrameGroupBy:
    return sales_data_set.groupby(by=["store", "item"])

@tag(stage="product")
def lag_sales_31(grp_date_store_item:DataFrameGroupBy) -> pd.Series:
    df = grp_date_store_item.shift(31)
    res = df["sales"].dropna()
    return res


@tag(stage="product")
@extract_columns("store", "item", "sales")
def sales_data_columns(sales_data_set_output_idx:pd.DataFrame) -> pd.DataFrame:
    return sales_data_set_output_idx

