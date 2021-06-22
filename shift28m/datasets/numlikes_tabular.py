import os
import glob
import pandas as pd
import numpy as np

from shift28m import constants as C
from shift28m import msgs as M
from shift28m.datasets import df_manipulations


def load_numlikes_tabular(root: str = C.ROOT):
    # load *.jsonl files in the root directory
    jsonls: list = glob.glob(os.path.join(root, f"*.{C.JSONL}"))
    if len(jsonls) == 0:
        raise RuntimeError(M.DATASET_NOT_FOUND)

    df: pd.DataFrame = pd.read_json(jsonls[0], orient=C.RECORDS, lines=True)
    for jsonl in jsonls[1:]:
        df_: pd.DataFrame = pd.read_json(jsonl, orient=C.RECORDS, lines=True)
        df = pd.concat([df, df_])

    df: pd.DataFrame = df.sort_values(C.Keys.PUBLISH_DATE)

    # generate input features
    user_ids: np.ndarray = df[C.Keys.USER].map(df_manipulations.extract_user_id).values
    price_sum: np.ndarray = df[C.Keys.ITEMS].map(df_manipulations.price_sum).values
    price_mean: np.ndarray = df[C.Keys.ITEMS].map(df_manipulations.price_mean).values
    price_max: np.ndarray = df[C.Keys.ITEMS].map(df_manipulations.price_max).values
    price_min: np.ndarray = df[C.Keys.ITEMS].map(df_manipulations.price_min).values
    category_ids_1: np.ndarray = (
        df[C.Keys.ITEMS].map(df_manipulations.categories_count_embedding_id1).to_numpy()
    )

    y: np.ndarray = np.array(df.like_num)
    x: np.ndarray = np.array(
        (
            user_ids,
            price_sum,
            price_mean,
            price_max,
            price_min,
        )
    )

    x: np.ndarray = x.T
    x: np.ndarray = np.hstack([x, np.stack(category_ids_1)])

    return (x, y)
