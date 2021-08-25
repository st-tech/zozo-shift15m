import gzip
import json
import os
import pathlib
import random
from typing import Any, List, Tuple

import numpy as np
import shift15m.constants as C
import torch
from sklearn.model_selection import train_test_split

label_id = {"category": 1, "subcategory": 2}
label_name = {"category": C.CATEGORIES, "subcategory": C.SUB_CATEGORIES}


def get_loader(
    items: List[Tuple[str, str]],
    target: str,
    data_dir: str,
    batch_size: int,
    is_train: bool = True,
    num_workers: int = None,
) -> torch.utils.data.DataLoader:
    dataset = ImageFeatureDataset(items, pathlib.Path(data_dir), target)
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
    def __init__(self, items: List, root: pathlib.Path, target: str):
        self.items = []
        self.labels = {name: i for i, name in enumerate(label_name[target])}
        for item, label in items:
            if (root / f"{item}.json.gz").exists():
                self.items.append((item, label))
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


class ItemCatalog:
    def __init__(self, catalog_path: str) -> None:
        self.items = [
            s.split(" ") for s in open(catalog_path).read().strip().split("\n")
        ]

    def _validate(self, items: List[Tuple], year: str):
        if len(items) == 0:
            raise ValueError(f"No items with year {year}.")

    def get_train_valid_test_items(
        self,
        target: str,
        train_valid_year: str,
        test_year: str,
        train_size: int,
        valid_size: int,
        test_size: int,
    ) -> Any:
        train_valid = [
            (item[0], item[label_id[target]])
            for item in self.items
            if item[3] == train_valid_year
        ]
        self._validate(train_valid, train_valid_year)
        test = [
            (item[0], item[label_id[target]])
            for item in self.items
            if item[3] == test_year
        ]
        self._validate(test, test_year)

        # split train valid
        if train_valid_year == test_year:
            train_valid_test_size = train_size + valid_size + test_size
            if train_valid_test_size > len(train_valid):
                train_size = int(train_size / train_valid_test_size)
                valid_size = int(valid_size / train_valid_test_size)
                test_size = int(test_size / train_valid_test_size)

            train, valid_test = train_test_split(
                train_valid, train_size=train_size, test_size=valid_size + test_size
            )
            valid, test = train_test_split(
                valid_test, train_size=valid_size, test_size=test_size
            )

        else:
            train_valid_size = train_size + valid_size
            if train_valid_size > len(train_valid):
                train_size = int(train_size / train_valid_size)
                valid_size = int(valid_size / train_valid_size)

            train, valid = train_test_split(
                train_valid, train_size=train_size, test_size=valid_size
            )
            test = random.sample(test, min(test_size, len(test)))

        return train, valid, test
