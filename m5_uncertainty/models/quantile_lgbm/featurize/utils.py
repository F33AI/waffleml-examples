import numpy as np
import pandas as pd
from typing import List

from sklearn.preprocessing import OneHotEncoder


def oh_transform(data: pd.Series, names: List) -> pd.DataFrame:

    cols = [str(i) for i in range(len(names))]
    data = data.fillna("nan")
    ohe = OneHotEncoder(categories=[names], sparse_output=False,
                        handle_unknown='ignore')
    data = ohe.fit_transform(np.array(data).reshape(-1, 1))

    return pd.DataFrame(data, columns=cols)
