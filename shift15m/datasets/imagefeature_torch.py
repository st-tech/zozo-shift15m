import gzip
import json
import os
import pathlib
from typing import Any, List, Tuple

import numpy as np
import torch

label_id = {"category": 1, "subcategory": 2}


def get_loader(
    itemlist_path: str,
    data_dir: str,
    target: str,
    batch_size: int,
    is_train: bool = True,
    num_workers: int = None,
) -> torch.utils.data.DataLoader:
    items = [s.split(" ") for s in open(itemlist_path).read().strip().split("\n")]
    items = [(item[0], item[label_id[target]]) for item in items]
    root = pathlib.Path(data_dir)

    dataset = ImageFeatureDataset(items, root)
    loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=is_train,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers if num_workers else os.cpu_count(),
        drop_last=is_train,
    )
    return loader


class ImageFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, items: List, root: pathlib.Path):
        self.items = []
        self.labels = {}
        for item, label in items:
            if (root / f"{item}.json.gz").exists():
                self.items.append((item, label))
                if label not in self.labels:
                    self.labels[label] = len(self.labels)
        self.root = root

    @property
    def category_size(self) -> int:
        return len(self.labels)

    @property
    def category_count(self) -> List:
        count = [0] * self.category_size
        for _, label in self.items:
            count[self.labels[label]] += 1
        return count

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        item, label = self.items[idx]
        with gzip.open(self.root / f"{item}.json.gz", "r") as f:
            feature = json.load(f)

        feature = np.array(feature, dtype=np.float32)
        label = self.labels[label]
        return feature, label
