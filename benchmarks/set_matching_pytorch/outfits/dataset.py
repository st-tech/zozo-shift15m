import gzip
import json
import os

import numpy as np
import torch
from set_matching.datasets.transforms import FeatureListTransform


def get_train_val_loader(feature_dir, label_dir, batch_size, num_workers=None):
    train = json.load(open(os.path.join(label_dir, "train.json")))
    valid = json.load(open(os.path.join(label_dir, "valid.json")))

    return _get_loader(
        train, feature_dir, batch_size, num_workers=num_workers, is_train=True
    ), _get_loader(
        valid, feature_dir, batch_size, num_workers=num_workers, is_train=False
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
