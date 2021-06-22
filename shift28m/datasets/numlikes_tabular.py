import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from shift28m import constants as C
from shift28m import msgs as M
from shift28m.datasets import df_manipulations


class NumLikesRegression(object):
    def __init__(self, root: str = C.ROOT):
        # load *.jsonl files in the root directory
        self.jsonls: list = glob.glob(os.path.join(root, f"*.{C.JSONL}"))
        if len(self.jsonls) == 0:
            raise RuntimeError(M.DATASET_NOT_FOUND)

        self.df: pd.DataFrame = pd.read_json(
            self.jsonls[0], orient=C.RECORDS, lines=True
        )
        for jsonl in self.jsonls[1:]:
            df_: pd.DataFrame = pd.read_json(jsonl, orient=C.RECORDS, lines=True)
            self.df = pd.concat([self.df, df_])

        self.df: pd.DataFrame = self.df.sort_values(C.Keys.PUBLISH_DATE)

        self.__prepare_features()

    def __prepare_features(self):
        # generate input features
        self.user_ids: np.ndarray = (
            self.df[C.Keys.USER].map(df_manipulations.extract_user_id).values
        )
        self.price_sum: np.ndarray = (
            self.df[C.Keys.ITEMS].map(df_manipulations.price_sum).values
        )
        self.price_mean: np.ndarray = (
            self.df[C.Keys.ITEMS].map(df_manipulations.price_mean).values
        )
        self.price_max: np.ndarray = (
            self.df[C.Keys.ITEMS].map(df_manipulations.price_max).values
        )
        self.price_min: np.ndarray = (
            self.df[C.Keys.ITEMS].map(df_manipulations.price_min).values
        )
        self.category_ids_1: np.ndarray = (
            self.df[C.Keys.ITEMS]
            .map(df_manipulations.categories_count_embedding_id1)
            .to_numpy()
        )

    def load_dataset(
        self,
        train_size: int = 10000,
        test_size: int = 10000,
        target_shift: float = -1,
        random_seed: int = 128,
        max_iter: int = 100,
    ):

        self.y: np.ndarray = np.array(self.df.like_num)
        self.x: np.ndarray = np.array(
            (
                self.user_ids,
                self.price_sum,
                self.price_mean,
                self.price_max,
                self.price_min,
            )
        )

        self.x: np.ndarray = self.x.T
        self.x: np.ndarray = np.hstack([self.x, np.stack(self.category_ids_1)])

        N = len(self.x)
        x_test: np.ndarray = self.x[N - test_size :]
        y_test: np.ndarray = self.y[N - test_size :]

        x_pool: np.ndarray = self.x[: N - test_size]
        y_pool: np.ndarray = self.y[: N - test_size]

        if target_shift < 0:
            x_train, x_pool, y_train, y_pool = train_test_split(
                self.x, self.y, train_size=train_size
            )
        else:
            best_seed: int = random_seed
            best_dist: float = 1e9
            best_x_train: np.ndarray = None
            best_y_train: np.ndarray = None
            for i in range(max_iter):
                x_train_, _, y_train_, _ = train_test_split(
                    x_pool, y_pool, train_size=train_size, random_state=random_seed + i
                )

                dist = np.abs(y_train_.mean() - y_test.mean())
                if np.abs(dist - target_shift) < best_dist:
                    best_dist = np.abs(dist - target_shift)
                    best_seed = random_seed + i
                    best_x_train = x_train_
                    best_y_train = y_train_

                if dist < 1e-9:
                    break

            x_train = best_x_train
            y_train = best_y_train

        return (x_train, y_train), (x_test, y_test)
