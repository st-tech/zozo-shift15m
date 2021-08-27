import gzip
import json
import numpy as np
import os
from tqdm import tqdm

INPUT_DIR = "vgg16-features/features"
LBL_DIR = [
    "labels/2013-2013-label1",
    "labels/2013-2013-label2",
    "labels/2013-2013-label3",
    "labels/2013-2014-label1",
    "labels/2013-2014-label2",
    "labels/2013-2014-label3",
    "labels/2013-2015-label1",
    "labels/2013-2015-label2",
    "labels/2013-2015-label3",
    "labels/2013-2016-label1",
    "labels/2013-2016-label2",
    "labels/2013-2016-label3",
    "labels/2013-2017-label1",
    "labels/2013-2017-label2",
    "labels/2013-2017-label3",
]
OUT_DIR = "outputs"


def _read_feature(path):
    with gzip.open(path, mode="rt", encoding="utf-8") as f:
        feature = json.loads(f.read())
    return feature


def main(label_dir, n_comb=1, n_cands=8):
    test_data = json.load(open(os.path.join(label_dir, "test.json")))

    for i in range(len(test_data)):
        data = {}

        lst = np.delete(np.arange(len(test_data)), i)
        others = np.random.choice(lst, n_comb - 1, replace=False).tolist()
        target = [i] + others

        setX_images, setY_images = [], []
        for j in target:
            items = test_data[j]["items"]
            images = []
            for img in items:
                path = os.path.join(INPUT_DIR, str(img["item_id"]) + ".json.gz")
                images.append(_read_feature(path))
            images = np.array(images)

            y_size = len(images) // 2

            xy_mask = [True] * (len(images) - y_size) + [False] * y_size
            xy_mask = np.random.permutation(xy_mask)
            setX_images.extend(images[xy_mask].tolist())
            setY_images.extend(images[~xy_mask].tolist())

        data["query"] = setX_images
        answers = [setY_images]

        for j in range(n_cands - 1):
            lst = np.delete(np.arange(len(test_data)), target)
            negatives = np.random.choice(lst, n_comb, replace=False).tolist()
            assert len(set(target) & set(negatives)) == 0
            target += negatives  # avoid double-selecting

            setY_images = []
            for k in negatives:
                items = test_data[k]["items"]
                images = []
                for img in items:
                    path = os.path.join(INPUT_DIR, str(img["item_id"]) + ".json.gz")
                    images.append(_read_feature(path))
                images = np.random.permutation(images)

                y_size = len(images) // 2
                setY_images.extend(images[:y_size].tolist())

            answers.append(setY_images)

        data["answers"] = answers

        path = os.path.join(
            OUT_DIR, "test_ncand{}".format(n_cands), os.path.basename(label_dir)
        )
        if not os.path.isdir(path):
            os.makedirs(path)
        path = os.path.join(path, "{0:05d}.json.gz".format(i))
        data = json.dumps(data)
        with gzip.open(path, mode="wt") as f:
            f.write(data)


if __name__ == "__main__":
    for lbl in tqdm(LBL_DIR):
        main(lbl, n_comb=1, n_cands=4)
        main(lbl, n_comb=1, n_cands=8)
