import pandas as pd


def read_train_test(path:str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",")
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(by=["store", "item", "date"], ascending=True, inplace=True)
    return df

def read_preds(path:str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",")    
    return df