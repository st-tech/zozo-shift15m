import json
import numpy as np
import os
import itertools
from tqdm import tqdm

output_root = "year_classification_labels"
outfits = json.load(open("iqon_outfits.json"))
like = [int(t["like_num"]) for t in outfits]
year = [int(t["publish_date"][:4]) for t in outfits]

sort_ind = np.argsort(like)[::-1]
outfits = np.array(outfits)[sort_ind].tolist()
year = np.array(year)[sort_ind].tolist()

years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
train = 3733
val = 466
test = 466

exist_item = set()

for y in tqdm(years):
    ind_y = np.array(year) == y
    outfits_y = np.array(outfits)[ind_y].tolist()
    items_y = [i["item_id"] for o in outfits_y for i in o["items"]]
    likes_y = [[o["like_num"]] * len(o["items"]) for o in outfits_y]
    likes_y = list(itertools.chain.from_iterable(likes_y))
    unq_ind = [True if ix not in exist_item else False for ix in items_y]
    items_y = np.array(items_y)[unq_ind].tolist()
    likes_y = np.array(likes_y)[unq_ind].tolist()

    items_target = list(dict.fromkeys(items_y))
    items_target = items_target[: train + val + test]
    exist_item |= set(items_y)

    np.random.seed(0)
    data_perm = np.random.permutation(items_target)
    data_tr = data_perm[0:train]
    data_vl = data_perm[train : train + val]
    data_te = data_perm[train + val : train + val + test]
    data_tr = data_tr.tolist()
    data_vl = data_vl.tolist()
    data_te = data_te.tolist()
    output_dir = os.path.join(output_root, str(y), "label1")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.json"), "w") as f:
        json.dump(data_tr, f, indent=4, ensure_ascii=True)
    with open(os.path.join(output_dir, "valid.json"), "w") as f:
        json.dump(data_vl, f, indent=4, ensure_ascii=True)
    with open(os.path.join(output_dir, "test.json"), "w") as f:
        json.dump(data_te, f, indent=4, ensure_ascii=True)

    np.random.seed(1)
    data_perm = np.random.permutation(items_target)
    data_tr = data_perm[0:train]
    data_vl = data_perm[train : train + val]
    data_te = data_perm[train + val : train + val + test]
    data_tr = data_tr.tolist()
    data_vl = data_vl.tolist()
    data_te = data_te.tolist()
    output_dir = os.path.join(output_root, str(y), "label2")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.json"), "w") as f:
        json.dump(data_tr, f, indent=4, ensure_ascii=True)
    with open(os.path.join(output_dir, "valid.json"), "w") as f:
        json.dump(data_vl, f, indent=4, ensure_ascii=True)
    with open(os.path.join(output_dir, "test.json"), "w") as f:
        json.dump(data_te, f, indent=4, ensure_ascii=True)

    np.random.seed(2)
    data_perm = np.random.permutation(items_target)
    data_tr = data_perm[0:train]
    data_vl = data_perm[train : train + val]
    data_te = data_perm[train + val : train + val + test]
    data_tr = data_tr.tolist()
    data_vl = data_vl.tolist()
    data_te = data_te.tolist()
    output_dir = os.path.join(output_root, str(y), "label3")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.json"), "w") as f:
        json.dump(data_tr, f, indent=4, ensure_ascii=True)
    with open(os.path.join(output_dir, "valid.json"), "w") as f:
        json.dump(data_vl, f, indent=4, ensure_ascii=True)
    with open(os.path.join(output_dir, "test.json"), "w") as f:
        json.dump(data_te, f, indent=4, ensure_ascii=True)

    np.random.seed(3)
    data_perm = np.random.permutation(items_target)
    data_tr = data_perm[0:train]
    data_vl = data_perm[train : train + val]
    data_te = data_perm[train + val : train + val + test]
    data_tr = data_tr.tolist()
    data_vl = data_vl.tolist()
    data_te = data_te.tolist()
    output_dir = os.path.join(output_root, str(y), "label4")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.json"), "w") as f:
        json.dump(data_tr, f, indent=4, ensure_ascii=True)
    with open(os.path.join(output_dir, "valid.json"), "w") as f:
        json.dump(data_vl, f, indent=4, ensure_ascii=True)
    with open(os.path.join(output_dir, "test.json"), "w") as f:
        json.dump(data_te, f, indent=4, ensure_ascii=True)

    np.random.seed(4)
    data_perm = np.random.permutation(items_target)
    data_tr = data_perm[0:train]
    data_vl = data_perm[train : train + val]
    data_te = data_perm[train + val : train + val + test]
    data_tr = data_tr.tolist()
    data_vl = data_vl.tolist()
    data_te = data_te.tolist()
    output_dir = os.path.join(output_root, str(y), "label5")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.json"), "w") as f:
        json.dump(data_tr, f, indent=4, ensure_ascii=True)
    with open(os.path.join(output_dir, "valid.json"), "w") as f:
        json.dump(data_vl, f, indent=4, ensure_ascii=True)
    with open(os.path.join(output_dir, "test.json"), "w") as f:
        json.dump(data_te, f, indent=4, ensure_ascii=True)
