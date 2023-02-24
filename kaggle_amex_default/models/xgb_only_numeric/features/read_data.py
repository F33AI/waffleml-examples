from typing import Any, Dict, List
from xml.sax.handler import feature_external_ges

import hamilton.function_modifiers as fun_mod
import numpy as np
import pandas as pd
from hamilton.function_modifiers import extract_columns, extract_fields

from . import data_io as io
from . import schema




@extract_fields(dict([(k, list) for k in schema.FEATURES.keys()] + [("ftr_target", str)]))
def features_schema() -> Dict[str, Any]:
    res = {"ftr_target": schema.TARGET}
    res.update(schema.FEATURES)
    return res 

@fun_mod.config.when(stage="train")
def input_dataset__train(data_path:str, ftr_all:List[str], ftr_target:str, features_schema:Dict[str, Any]) -> pd.DataFrame:
    df = io.read_dataset(data_path, features_schema) 
    
    return df

@extract_columns(
   *schema.FEATURES["ftr_all"] 
)
def all_features(df_trn:pd.DataFrame) -> pd.DataFrame:
    return df_trn

@fun_mod.config.when(stage="test")
def input_dataset__test(data_path: str, ftr_all: List[str], features_schema:Dict[str, Any]) -> pd.DataFrame:
    df = io.read_dataset(data_path, features_schema)
    
    return df[ftr_all]

@fun_mod.config.when(stage="evaluate")
def input_dataset__evaluate(data_path:str, ftr_target:str) -> np.array:
    df = io.read_target(data_path, ftr_target)
 
    target = df.to_numpy()
    return target
