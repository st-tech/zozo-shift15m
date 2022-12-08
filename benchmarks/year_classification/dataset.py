import json
import numpy as np
import os
import shutil
import tarfile
import gzip
import random
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
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tf, data_dir)

            tmp_dir = fpath[:-7]
            for featname in os.listdir(tmp_dir):
                src = os.path.join(tmp_dir, featname)
                dst = os.path.join(data_dir, "features", featname)
                shutil.move(src, dst)


def get_train_val_dataset(feature_dir, label_dir, year_pos, year_neg, split):
    _extract_tarfiles(feature_dir)

    train_pos = json.load(
        open(os.path.join(label_dir, str(year_pos), split, "train.json"))
    )
    train_neg = json.load(
        open(os.path.join(label_dir, str(year_neg), split, "train.json"))
    )
    train_dataset = [{"item_id": str(i), "label": np.int32(1)} for i in train_pos] + [
        {"item_id": str(i), "label": np.int32(0)} for i in train_neg
    ]
    random.shuffle(train_dataset)
    print(
        "train.json loaded. pos: "
        + str(year_pos)
        + " neg: "
        + str(year_neg)
        + " split: "
        + split
    )

    valid_pos = json.load(
        open(os.path.join(label_dir, str(year_pos), split, "valid.json"))
    )
    valid_neg = json.load(
        open(os.path.join(label_dir, str(year_neg), split, "valid.json"))
    )
    valid_dataset = [{"item_id": str(i), "label": np.int32(1)} for i in valid_pos] + [
        {"item_id": str(i), "label": np.int32(0)} for i in valid_neg
    ]
    print(
        "valid.json loaded. pos: "
        + str(year_pos)
        + " neg: "
        + str(year_neg)
        + " split: "
        + split
    )

    return (
        _get_dataset(train_dataset, feature_dir, is_train=True),
        _get_dataset(valid_dataset, feature_dir, is_train=False),
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
