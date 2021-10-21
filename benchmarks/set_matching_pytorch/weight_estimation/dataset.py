import gzip
import json
import os

import numpy as np
import torch


def _get_items(sets):
    item_ids = []
    for set_ in sets:
        item_ids.extend([item["item_id"] for item in set_["items"]])
    return item_ids


def get_train_val_loader(feature_dir, label_dir, batch_size, num_workers=None):
    train = json.load(open(os.path.join(label_dir, "train.json")))
    train = _get_items(train)

    valid = json.load(open(os.path.join(label_dir, "valid.json")))
    valid = _get_items(valid)

    common_id = set(train) & set(valid)
    traindiff_id = [d for d in train if d not in common_id]
    validdiff_id = [d for d in valid if d not in common_id]
    sampling_num = len(traindiff_id) - len(validdiff_id)
    duplicated_ind = np.random.choice(len(validdiff_id), sampling_num, replace=True)
    duplicated_valid_id = [validdiff_id[i] for i in duplicated_ind]
    validdiff_id = validdiff_id + duplicated_valid_id
    train_dataset = [{"item_id": d, "label": np.int32(1)} for d in traindiff_id] + [
        {"item_id": d, "label": np.int32(0)} for d in validdiff_id
    ]
    print("train data created")

    test = json.load(open(os.path.join(label_dir, "test.json")))
    test = _get_items(test)
    common_id = set(train) & set(test)
    testdiff_id = [d for d in test if d not in common_id]
    removed = np.random.choice(len(traindiff_id), len(testdiff_id), replace=False)
    removed_train_id = [traindiff_id[i] for i in removed]
    test_dataset = [{"item_id": d, "label": np.int32(1)} for d in removed_train_id] + [
        {"item_id": d, "label": np.int32(0)} for d in testdiff_id
    ]
    print("test data created")

    return _get_loader(
        train_dataset, feature_dir, batch_size, num_workers, is_train=True
    ), _get_loader(test_dataset, feature_dir, batch_size, num_workers, is_train=False)


def _get_loader(data, root, batch_size, num_workers=None, is_train=False):
    return torch.utils.data.DataLoader(
        FeatureLabelDataset(data, root),
        shuffle=is_train,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers if num_workers else os.cpu_count(),
        drop_last=is_train,
    )


class FeatureLabelDataset(torch.utils.data.Dataset):
    def __init__(self, data, root) -> None:
        self.data = data
        self.feature_dir = os.path.join(root, "features")

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
