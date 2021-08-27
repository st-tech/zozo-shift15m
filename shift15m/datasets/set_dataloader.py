import json
import numpy as np
import os
import gzip
import argparse
import shutil
import tarfile


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


class get_dataset(object):
    def __init__(self, sets, feature_dir):
        self.sets = sets
        self.feat_dir = os.path.join(feature_dir, "features")

    def __len__(self):
        return len(self.sets)

    def _load_feature(self, path):
        with gzip.open(path, mode="rt", encoding="utf-8") as f:
            feature = json.loads(f.read())
        return np.array(feature, dtype=np.float32)

    def get_example(self, i):
        return np.array(
            [
                self._load_feature(
                    os.path.join(self.feat_dir, j["item_id"] + ".json.gz")
                )
                for j in self.sets[i]["items"]
            ]
        )


def get_train_val_dataset(feature_dir, label_dir, download=True):
    if download == True:
        pass
        # _download_features()
        # _download_labels()
        # _extract_tarfiles(feature_dir)

    train = json.load(open(os.path.join(label_dir, "train.json")))
    valid = json.load(open(os.path.join(label_dir, "valid.json")))

    return get_dataset(train, feature_dir), get_dataset(valid, feature_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", "-f", type=str)
    parser.add_argument("--label_dir", "-l", type=str)
    args = parser.parse_args()

    train, valid = get_train_val_dataset(args.feature_dir, args.label_dir)
