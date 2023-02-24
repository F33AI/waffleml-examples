# Copyright (C) Fourteen33 Inc. - All Rights Reserved
import logging
from typing import Any, Dict, List, Tuple
import sys

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

PERF_FOLDS = 5
HYPER_FOLDS = 5

def read_data(path:str) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["customer_ID"])
    df["row_idx"] = list(range(df.shape[0]))

    return df

def create_cv_folds(n: int, csv_path: str, train_ratio: float, val_ratio: float,
                    test_ratio: float) -> List[List[int]]:
    logger.info(f"Folds cnt = {n}")
    logger.info(f"train ratio: {train_ratio}")
    logger.info(f"val ratio: {val_ratio}")
    logger.info(f"test ratio: {test_ratio}")

    df = read_data(csv_path)
    idx = np.array(df["row_idx"])
    all_customers_ids = list(df.customer_ID)
    shuffled_unique_customers_idx = np.array(df.customer_ID.unique())
    np.random.shuffle(shuffled_unique_customers_idx)

    logger.info(f"Number of all rows = {df.shape[0]} / {len(all_customers_ids)}")
    logger.info(f"Number of unique customers IDs = {shuffled_unique_customers_idx.shape[0]}")

    groups = []

    fold_size = shuffled_unique_customers_idx.shape[0] // n

    for fold_idx in range(n):
        logger.info(f"Processing fold_idx={fold_idx} / {n}...")
        fld_customers = set(shuffled_unique_customers_idx[fold_idx * fold_size: (fold_idx + 1) * fold_size])

        logger.info(f">>>> len(fld_customers)={len(fld_customers)}")

        indicies = idx[[(v in fld_customers) for v in all_customers_ids]]
        train_size = int(train_ratio * len(indicies))
        val_size = train_size + int(val_ratio * len(indicies))
        logger.info(f">>>> Selected {indicies.shape} / {len(all_customers_ids)}")
        logger.info(f">>>>>>> Train size={train_size}")
        logger.info(f">>>>>>> Valid size={val_size - train_size}")
        logger.info(f">>>>>>> Test size={len(indicies) - val_size}")
        groups.append([indicies[0: train_size].tolist(), indicies[train_size: val_size].tolist(), indicies[val_size:].tolist()])
        logger.info("done")
    return groups
