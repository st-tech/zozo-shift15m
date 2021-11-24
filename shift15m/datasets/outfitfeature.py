import datetime
import gzip
import json
import os
import pathlib
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shift15m.constants as C
import torch
from set_matching.datasets.transforms import FeatureListTransform
from shift15m.constants import Keys as K
from sklearn.model_selection import train_test_split


def get_loader(
    dataset: Any,
    batch_size: int,
    num_workers: Optional[int] = None,
    is_train: bool = True,
):
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=is_train,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers if num_workers else os.cpu_count(),
        drop_last=is_train,
    )


class MultisetSplitDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sets: List,
        root: pathlib.Path,
        n_sets: int,
        n_drops: Optional[int] = None,
        max_elementnum_per_set: Optional[int] = 8,
    ):
        self.sets = sets
        self.feat_dir = root
        self.n_sets = n_sets
        self.n_drops = n_drops
        if n_drops is None:
            n_drops = max_elementnum_per_set // 2
        setX_size = (max_elementnum_per_set - n_drops) * n_sets
        setY_size = n_drops * n_sets
        self.transform_x = FeatureListTransform(
            max_set_size=setX_size, apply_shuffle=True, apply_padding=True
        )
        self.transform_y = FeatureListTransform(
            max_set_size=setY_size, apply_shuffle=True, apply_padding=True
        )

    def __len__(self):
        return len(self.sets)

    def __getitem__(self, i):
        if self.n_sets > 1:  # you can conduct "superset matching" by using n_sets > 1
            indices = np.delete(np.arange(len(self.sets)), i)
            indices = np.random.choice(indices, self.n_sets - 1, replace=False)
            indices = [i] + list(indices)
        else:
            indices = [i]

        setX_features, setX_ids = [], []
        setY_features, setY_ids = [], []
        for j in indices:
            set_ = self.sets[j]
            items = set_["items"]
            features = []
            for item in items:
                feat_name = str(item["item_id"]) + ".json.gz"
                path = str(self.feat_dir / feat_name)
                features.append(self._load_feature(path))
            features = np.array(features)

            n_features = len(features)
            if self.n_drops is None:
                n_drops = n_features // 2
            else:
                n_drops = self.n_drops

            xy_mask = [True] * (n_features - n_drops) + [False] * n_drops
            xy_mask = np.random.permutation(xy_mask)
            setX_features.extend(list(features[xy_mask, :]))
            setY_features.extend(list(features[~xy_mask, :]))
            setX_ids.extend([j] * (n_features - n_drops))
            setY_ids.extend([j] * n_drops)

        setX_features, _, setX_mask = self.transform_x(setX_features, setX_ids)
        setY_features, _, setY_mask = self.transform_y(setY_features, setY_ids)
        return setX_features, setX_mask, setY_features, setY_mask

    def _load_feature(self, path: str):
        with gzip.open(path, mode="rt", encoding="utf-8") as f:
            feature = json.loads(f.read())
        return np.array(feature, dtype=np.float32)


class FINBsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sets: List,
        root: pathlib.Path,
        n_cand_sets: int,
        max_set_size_query: int,
        max_set_size_answer: int,
    ):
        self.sets = sets
        self.root = root
        self.n_cand_sets = n_cand_sets
        self.transform_q = FeatureListTransform(
            max_set_size=max_set_size_query, apply_shuffle=False, apply_padding=True
        )
        self.transform_a = FeatureListTransform(
            max_set_size=max_set_size_answer, apply_shuffle=False, apply_padding=True
        )

    def __len__(self):
        return len(self.sets)

    def __getitem__(self, idx):
        _set = self.sets[idx]
        query_features, query_categories = [], []
        for item in _set["query"]:
            with gzip.open(self.root / item, "r") as f:
                feature = json.load(f)
            query_features.append(feature)
            query_categories.append(1)

        question, _, q_mask = self.transform_q(query_features, query_categories)

        n_answers = min(len(_set["answers"]), self.n_cand_sets)
        answers, a_masks = [], []
        for cand in _set["answers"][:n_answers]:
            features, categories = [], []
            for item in cand:
                with gzip.open(self.root / item, "r") as f:
                    feature = json.load(f)
                features.append(feature)
                categories.append(1)
            ans, _, a_mask = self.transform_a(features, categories)
            answers.append(ans)
            a_masks.append(a_mask)

        return question, q_mask, np.array(answers), np.array(a_masks)


class FeatureLabelDataset(torch.utils.data.Dataset):
    def __init__(self, data, root) -> None:
        self.data = data
        self.feature_dir = root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        feat_name = str(item["item_id"]) + ".json.gz"
        path = os.path.join(self.feature_dir, feat_name)
        feature = self._read_feature(path)
        label = item["label"]

        return feature, label

    def _read_feature(self, path):
        with gzip.open(path, mode="rt", encoding="utf-8") as f:
            feature = json.loads(f.read())
        return np.array(feature, dtype=np.float32)


class IQONOutfits:
    def __init__(
        self,
        root: str = C.ROOT,
    ) -> None:
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if not (self.root / "iqon_outfits.json").exists():
            # download outfit json
            self._download_outfit_label()

        self._label_dir = self.root / "set_matching/labels"
        if not self._label_dir.exists():
            self._label_dir.mkdir(parents=True, exist_ok=True)
            self._make_trainval_dataset()

        self._feature_dir = self.root / "features"
        if not self._feature_dir.exists():
            # download feature jsons
            self._download_features()

    @property
    def feature_dir(self) -> pathlib.Path:
        return self._feature_dir

    def _download_features(self):
        msg = "It requires about 100GB storage to store 2.3M feature files. Do you continue to download? (y/[n]):"
        res = input(msg)
        if res.lower() not in ("y", "yes"):
            print("Skip download.")
            return

        from shift15m.datasets.download_tarfiles import main as dltars
        from shift15m.datasets.feature_tar_extractor import _extract_tarfiles as ext

        root = pathlib.Path(C.ROOT)
        dltars(str(root), os.cpu_count())
        ext(str(root))
        for fname in open(root / "tar_files.txt"):
            os.remove(root / fname.strip())

    def _download_outfit_label(self):
        print(f"Download {C.OUTFIT_JSON_URL} to {str(self.root)}")
        cmd = f"wget {C.OUTFIT_JSON_URL} -P {str(self.root)}"
        res = subprocess.run(cmd, shell=True, text=True)
        res.check_returncode()

    def _make_trainval_dataset(
        self, min_num_categories: int = 4, min_like_num: int = 50, seed: int = 0
    ):
        print("Make train/val dataset.")

        np.random.seed(seed)
        num_train, num_val, num_test = 30816, 3851, 3851  # max size

        df = pd.read_json(str(self.root / "iqon_outfits.json"), orient="records")
        df["num_categories"] = df["items"].apply(
            lambda x: len(set([item[K.CATEGORY_ID] for item in x]))
        )
        df = df[
            (df["num_categories"] >= min_num_categories)
            & (df["like_num"] >= min_like_num)
        ]
        df["publish_year"] = df[K.PUBLISH_DATE].apply(
            lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").year
        )

        ys = 2013
        for ye in range(2013, 2018):
            str_year = f"{ys}-{ye}"
            df_ys = df[df["publish_year"] == ys]

            df_ys = df_ys.sample(frac=1, random_state=seed)
            df_train = df_ys.head(num_train)
            if ys != ye:
                df_ye = df[df["publish_year"] == ye]
                df_ye = df_ye.sample(frac=1, random_state=seed).head(num_val + num_test)
                df_val, df_test = train_test_split(
                    df_ye, test_size=0.5, random_state=seed
                )
            else:
                df_ye = df_ys.tail(-num_train)
                df_val, df_test = train_test_split(
                    df_ye, test_size=0.5, random_state=seed
                )

            out_dir = self._label_dir / str_year
            out_dir.mkdir(exist_ok=True)
            df_train.to_json(str(out_dir / "train.json"), orient="records", indent=2)
            df_val.to_json(str(out_dir / "valid.json"), orient="records", indent=2)
            df_test.to_json(str(out_dir / "test.json"), orient="records", indent=2)

    def get_trainval_data(self, label_dir_name: str) -> Tuple[List, List]:
        path = self._label_dir / label_dir_name
        train = json.load(open(path / "train.json"))
        valid = json.load(open(path / "valid.json"))
        return train, valid

    def get_test_data(self, label_dir_name: str) -> List[Dict]:
        path = self._label_dir / label_dir_name
        test = json.load(open(path / "test.json"))
        return test

    def get_fitb_data(
        self, label_dir_name: str, n_comb: int = 1, n_cands: int = 8, seed: int = 0
    ) -> List:
        dir_name = self._label_dir / label_dir_name
        path = dir_name / f"test_examples_ncomb_{n_comb}_ncands_{n_cands}.json"
        if not path.exists():
            self._make_test_examples(dir_name, n_comb, n_cands, seed)
        test_examples = json.load(open(path))
        return test_examples

    def _make_test_examples(
        self, path: pathlib.Path, n_comb: int = 1, n_cands: int = 8, seed: int = 0
    ):
        print("Make test dataset.")
        np.random.seed(seed)

        test_sets = json.load(open(path / "test.json"))

        test_examples = []
        for i in range(len(test_sets)):
            example = {}

            lst = np.delete(np.arange(len(test_sets)), i)
            others = np.random.choice(lst, n_comb - 1, replace=False).tolist()
            target = [i] + others

            setX_items, setY_items = [], []
            for j in target:
                items = [
                    str(item["item_id"]) + ".json.gz" for item in test_sets[j]["items"]
                ]
                items = np.array(items)

                y_size = len(items) // 2

                xy_mask = [True] * (len(items) - y_size) + [False] * y_size
                xy_mask = np.random.permutation(xy_mask)
                setX_items.extend(items[xy_mask].tolist())
                setY_items.extend(items[~xy_mask].tolist())

            example["query"] = setX_items

            answers = [setY_items]
            for j in range(n_cands - 1):
                lst = np.delete(np.arange(len(test_sets)), target)
                negatives = np.random.choice(lst, n_comb, replace=False).tolist()
                assert len(set(target) & set(negatives)) == 0
                target += negatives  # avoid double-selecting

                setY_items = []
                for k in negatives:
                    items = [
                        str(item["item_id"]) + ".json.gz"
                        for item in test_sets[k]["items"]
                    ]
                    items = np.random.permutation(items)

                    y_size = len(items) // 2
                    setY_items.extend(items[:y_size].tolist())

                answers.append(setY_items)

            example["answers"] = answers

            test_examples.append(example)

        with open(
            path / f"test_examples_ncomb_{n_comb}_ncands_{n_cands}.json", "w"
        ) as f:
            json.dump(test_examples, f, indent=2)
