import json
import numpy as np
import os
import shutil
import tarfile
import gzip
import chainer

from chainer.datasets import TransformDataset


def _extract_tarfiles(data_dir):
    path = os.path.join(data_dir, 'features')
    if not os.path.isdir(path):
        os.mkdir(path)

        feature_tar_files = open(os.path.join(data_dir, 'tar_files.txt')).read().strip().split('\n')
        feature_tar_files = [os.path.join(data_dir, s) for s in feature_tar_files]
        for fpath in feature_tar_files:
            with tarfile.open(fpath, 'r') as tf:
                tf.extractall(data_dir)
    
            tmp_dir = fpath[:-7]
            for featname in os.listdir(tmp_dir):
                src = os.path.join(tmp_dir, featname)
                dst = os.path.join(data_dir, 'features', featname)
                shutil.move(src, dst)


def get_train_val_dataset(feature_dir, label_dir):
    train = json.load(open(os.path.join(label_dir, 'train.json')))
    valid = json.load(open(os.path.join(label_dir, 'valid.json')))
    
    return _get_dataset(train, feature_dir, is_train=True, is_padding=True), \
           _get_dataset(valid, feature_dir, is_train=False, is_padding=True)
  

def _get_dataset(sets, feature_dir, is_train=False,
                 is_padding=False, n_sets=1, n_drops=None):
    return TransformDataset(
            OutfitMultiset(sets, feature_dir, n_sets, n_drops=n_drops),
            TransformMultiset(is_train, is_padding, n_sets, n_drops=n_drops))


class TransformMultiset(object):
    def __init__(self, is_train, is_padding, n_sets, n_drops=None,
                 max_elementnum=8):
        self.is_train = is_train
        self.is_padding = is_padding

        if n_drops is None:
            n_drops = max_elementnum // 2
        self.setX_size = (max_elementnum - n_drops) * n_sets
        self.setY_size = n_drops * n_sets

    def __call__(self, in_data):
        setX_features, setX_ids, setY_features, setY_ids = in_data

        setX_features, setX_ids = self._transform(setX_features, setX_ids, self.setX_size)
        setY_features, setY_ids = self._transform(setY_features, setY_ids, self.setY_size)
        return setX_features, np.array(setX_ids), setY_features, np.array(setY_ids)

    def _transform(self, features, ids, size):
        processed = list()
        for _feat in features:
            feat = _feat
            feat = self._rescale(feat)
            processed.append(feat)

        perm = np.arange(len(ids))
        perm = np.random.permutation(perm)
        perm_features, perm_ids = list(), list()
        for ix in perm:
            perm_features.append(processed[ix])
            perm_ids.append(ids[ix])
        
        if len(perm_features) > size:
            perm_features = perm_features[:size]
            perm_ids = perm_ids[:size]

        if self.is_padding:
            while len(perm_features) < size:
                perm_features.append(perm_features[-1])
                perm_ids.append(-1)

        return perm_features, perm_ids

    def _rescale(self, feature):
        return feature


class OutfitMultiset(chainer.dataset.DatasetMixin):
    def __init__(self, sets, root, n_sets, n_drops=None):
        self.sets = sets
        self.feat_dir = os.path.join(root, 'features')
        self.n_sets = n_sets
        self.split_half = (n_drops is None)
        self.n_drops = n_drops

    def __len__(self):
        return len(self.sets)

    def get_example(self, i):
        if self.n_sets > 1: # you can conduct "superset matching" by using n_sets > 1
            indices = np.delete(np.arange(len(self.sets)), i)
            indices = np.random.choice(indices, self.n_sets - 1, replace=False)
            indices = [i] + list(indices)
        else:
            indices = [i]

        setX_features, setX_ids = [], []
        setY_features, setY_ids = [], []
        for j in indices:
            set_ = self.sets[j]
            items = set_['items']
            features = []
            for item in items:
                feat_name = str(item['item_id']) + '.json.gz'
                path = os.path.join(self.feat_dir, feat_name)
                features.append(self._load_feature(path))
            features = np.array(features)

            n_features = len(features)
            if self.split_half:
                n_drops = n_features // 2
            else:
                n_drops = self.n_drops

            xy_mask = [True] * (n_features - n_drops) + [False] * n_drops
            xy_mask = np.random.permutation(xy_mask)
            setX_features.extend(list(features[xy_mask, :]))
            setY_features.extend(list(features[~xy_mask, :]))
            setX_ids.extend([j] * (n_features - n_drops))
            setY_ids.extend([j] * n_drops)

        return setX_features, setX_ids, setY_features, setY_ids

    def _load_feature(self, path):
        with gzip.open(path, mode='rt', encoding='utf-8') as f:
            feature = json.loads(f.read())
        return np.array(feature, dtype=np.float32)


class TransformFIMBsDataset(object):
    def __init__(self, is_padding, max_elementnum=5):
        self.is_padding = is_padding
        self.size = max_elementnum

    def __call__(self, in_data):
        setX_features, setY_features = in_data

        setX_features, setX_ids = self._transform(setX_features, self.size)
        answer_features, answer_ids = [], []
        for features in setY_features:
            setY_features, setY_ids = self._transform(features, self.size)
            answer_features.append(setY_features)
            answer_ids.append(setY_ids)

        return setX_features, np.array(setX_ids), answer_features, np.array(answer_ids)

    def _transform(self, features, size):
        if len(features) > size:
            features = features[:size]
        ids = [1] * len(features)

        if self.is_padding:
            while len(features) < size:
                features.append(features[-1])
                ids.append(-1)

        return features, ids