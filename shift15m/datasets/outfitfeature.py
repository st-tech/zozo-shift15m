import datetime
import gzip
import json
import os
import pathlib
import subprocess
from typing import Optional, Union

import numpy as np
import pandas as pd
import shift15m.constants as C
import torch
from set_matching.datasets.transforms import FeatureListTransform
from shift15m.constants import Keys as K
from sklearn.model_selection import train_test_split

TEST_DATA_TARFILES = "https://research.zozo.com/data_release/shift15m/set_matching/set_matching_test_data_splits/filelist.txt"
OUTFIT_JSON_URL = (
    "https://research.zozo.com/data_release/shift15m/label/iqon_outfits.json"
)


def get_train_val_loader(
    train_year: Union[str, int],
    valid_year: Union[str, int],
    batch_size: int,
    root: str = C.ROOT,
    num_workers: Optional[int] = None,
):
    label_dir_name = f"{train_year}-{valid_year}"

    iqon_outfits = IQONOutfits(root=root)

    train, valid = iqon_outfits.get_trainval_data(label_dir_name)
    feature_dir = iqon_outfits.feature_dir
    return _get_loader(
        train, feature_dir, batch_size, num_workers=num_workers, is_train=True
    ), _get_loader(
        valid, feature_dir, batch_size, num_workers=num_workers, is_train=False
    )


def get_test_loader(
    train_year: Union[str, int],
    valid_year: Union[str, int],
    batch_size: int,
    root: str = C.ROOT,
    num_workers: Optional[int] = None,
):
    label_dir_name = f"{train_year}-{valid_year}"

    iqon_outfits = IQONOutfits(root=root)

    test = iqon_outfits.get_test_data(label_dir_name)
    feature_dir = iqon_outfits.feature_dir
    return _get_loader(
        test, feature_dir, batch_size, num_workers=num_workers, is_train=True
    )


def _get_loader(
    sets,
    feature_dir,
    batch_size,
    n_sets=1,
    n_drops=None,
    num_workers=None,
    is_train=True,
):
    return torch.utils.data.DataLoader(
        OutfitMultiset(sets, feature_dir, n_sets, n_drops=n_drops),
        shuffle=is_train,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers if num_workers else os.cpu_count(),
        drop_last=is_train,
    )


class OutfitMultiset(torch.utils.data.Dataset):
    def __init__(self, sets, root, n_sets, n_drops=None, max_elementnum_per_set=8):
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
                path = os.path.join(self.feat_dir, feat_name)
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

    def _load_feature(self, path):
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

        self.label_dir = self.root / "set_matching" / "labels"
        if not self.label_dir.exists():
            self.label_dir.mkdir(parents=True, exist_ok=True)
            self._make_trainval_dataset()

        self.feature_dir = self.root / "features"
        if not self.feature_dir.exists():
            # download feature jsons
            self._download_features()

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
        print(f"Download {OUTFIT_JSON_URL} to {str(self.root)}")
        cmd = f"wget {OUTFIT_JSON_URL} -P {str(self.root)}"
        res = subprocess.run(cmd, shell=True, text=True)
        res.check_returncode()

        # download test data
        # tmp_dir = task_root / "set_matching_test_data_splits"
        # tmp_dir.mkdir(parents=True, exist_ok=True)
        # cmd = f"wget -i {TEST_DATA_TARFILES} -P {str(tmp_dir)}"
        # res = subprocess.run(cmd, shell=True, text=True)
        # res.check_returncode()
        # cmd = f"cat {str(tmp_dir)}/set_matching_test_data.tar.gz-* > {str(task_root)}/set_matching_test_data.tar.gz"
        # res = subprocess.run(cmd, shell=True, text=True)
        # res.check_returncode()
        # cmd = f"tar zxf {str(task_root)}/set_matching_test_data.tar.gz -C {str(task_root)}"
        # res = subprocess.run(cmd, shell=True, text=True)
        # res.check_returncode()
        # (task_root / "set_matching_test_data.tar.gz").unlink()
        # shutil.rmtree(str(tmp_dir))

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

            out_dir = self.label_dir / str_year
            out_dir.mkdir(exist_ok=True)
            df_train.to_json(str(out_dir / "train.json"), orient="records", indent=2)
            df_val.to_json(str(out_dir / "valid.json"), orient="records", indent=2)
            df_test.to_json(str(out_dir / "test.json"), orient="records", indent=2)

    def get_trainval_data(self, label_dir_name):
        path = self.label_dir / label_dir_name
        train = json.load(open(path / "train.json"))
        valid = json.load(open(path / "valid.json"))
        return train, valid

    def get_test_data(self, label_dir_name):
        path = self.label_dir / label_dir_name
        test = json.load(open(path / "test.json"))
        return test
