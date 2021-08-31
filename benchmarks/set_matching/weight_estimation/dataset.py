import json
import numpy as np
import os
import shutil
import tarfile
import gzip
import chainer

from chainer.datasets import TransformDataset


def _extract_tarfiles(data_dir):
    path = os.path.join(data_dir, "features")
    if not os.path.isdir(path):
        os.mkdir(path)

        feature_tar_files = (
            open(os.path.join(data_dir, "tar_files.txt")).read().strip().split("\n")
        )
        feature_tar_files = [os.path.join(data_dir, s) for s in feature_tar_files]
        for fpath in feature_tar_files:
            with tarfile.open(fpath, "r") as tf:
                tf.extractall(data_dir)

            tmp_dir = fpath[:-7]
            for featname in os.listdir(tmp_dir):
                src = os.path.join(tmp_dir, featname)
                dst = os.path.join(data_dir, "features", featname)
                shutil.move(src, dst)


def get_train_val_dataset(feature_dir, label_dir):
    _extract_tarfiles(feature_dir)

    train = json.load(open(os.path.join(label_dir, "train.json")))
    train = [i["items"] for i in train]
    train = [i for items in train for i in items]
    train_id = [d["item_id"] for d in train]
    valid = json.load(open(os.path.join(label_dir, "valid.json")))
    valid = [i["items"] for i in valid]
    valid = [i for items in valid for i in items]
    valid_id = [d["item_id"] for d in valid]
    common_id = set(train_id) & set(valid_id)
    traindiff_id = [d for d in train_id if d not in common_id]
    validdiff_id = [d for d in valid_id if d not in common_id]
    sampling_num = len(traindiff_id) - len(validdiff_id)
    duplicated_ind = np.random.choice(len(validdiff_id), sampling_num, replace=True)
    duplicated_valid_id = [validdiff_id[i] for i in duplicated_ind]
    validdiff_id = validdiff_id + duplicated_valid_id
    train_dataset = [{"item_id": d, "label": np.int32(1)} for d in traindiff_id] + [
        {"item_id": d, "label": np.int32(0)} for d in validdiff_id
    ]
    print("train data created")

    test = json.load(open(os.path.join(label_dir, "test.json")))
    test = [i["items"] for i in test]
    test = [i for items in test for i in items]
    test_id = [d["item_id"] for d in test]
    common_id = set(train_id) & set(test_id)
    testdiff_id = [d for d in test_id if d not in common_id]
    removed = np.random.choice(len(traindiff_id), len(testdiff_id), replace=False)
    removed_train_id = [traindiff_id[i] for i in removed]
    test_dataset = [{"item_id": d, "label": np.int32(1)} for d in removed_train_id] + [
        {"item_id": d, "label": np.int32(0)} for d in testdiff_id
    ]
    print("test data created")

    return _get_dataset(train_dataset, feature_dir, is_train=True), _get_dataset(
        test_dataset, feature_dir, is_train=False
    )


def _get_dataset(sets, feature_dir, is_train=False):
    return TransformDataset(LoadData(sets, feature_dir), TransformData(is_train))


class TransformData(object):
    def __init__(self, is_train):
        self.is_train = is_train

    def __call__(self, in_data):
        feat, label = in_data
        feat, label = self._transform(feat, label)
        return feat, label

    def _transform(self, feat, label):
        feat = self._rescale(feat)
        return feat, label

    def _rescale(self, feat):
        return feat


class LoadData(chainer.dataset.DatasetMixin):
    def __init__(self, sets, root):
        self.sets = sets
        self.feat_dir = os.path.join(root, "features")

    def __len__(self):
        return len(self.sets)

    def get_example(self, i):
        item = self.sets[i]
        feat_name = str(item["item_id"]) + ".json.gz"
        path = os.path.join(self.feat_dir, feat_name)
        feature = self._read_feature(path)
        label = item["label"]

        return feature, label

    def _read_feature(self, path):
        with gzip.open(path, mode="rt", encoding="utf-8") as f:
            feature = json.loads(f.read())
        return np.array(feature, dtype=np.float32)
