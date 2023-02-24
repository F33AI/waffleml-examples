import hamilton.function_modifiers as fun_mod
import pandas as pd


@fun_mod.config.when(stage="train")
def target__train(input_dataset:pd.DataFrame, ftr_target:str) -> pd.Series:
    return input_dataset[ftr_target]

@fun_mod.config.when(stage="train")
def X__train(input_dataset:pd.DataFrame, ftr_numerical:list) -> pd.DataFrame:
    return input_dataset[ftr_numerical]

@fun_mod.config.when(stage="test")
def X__test(input_dataset:pd.DataFrame, ftr_numerical:list) -> pd.DataFrame:
    return input_dataset[ftr_numerical]

def day_of_week(S_2:pd.Series) -> pd.Series:
    return S_2.dt.day_of_week