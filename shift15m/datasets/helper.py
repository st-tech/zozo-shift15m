import json
import pathlib

from sklearn.model_selection import train_test_split

import shift15m.constants as C
from shift15m.constants import Keys as K


def make_item_catalog(inp: str, test_size: float = 0.2, output_dir: str = C.ROOT):
    data = json.load(open(inp))
    items = {}
    for _set in data:
        for item in _set[K.ITEMS]:
            items[item[K.ITEM_ID]] = (item[K.CATEGORY_ID], item[K.SUBCATEGORY_ID])

    items = [f"{k} {v[0]} {v[1]}" for k, v in items.items()]
    train, test = train_test_split(items, test_size=test_size)

    with open(pathlib.Path(output_dir) / "train.txt", "w") as f:
        f.write("\n".join(train))
    with open(pathlib.Path(output_dir) / "test.txt", "w") as f:
        f.write("\n".join(test))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="path to the input file")
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Ratio of test dataset"
    )
    parser.add_argument(
        "--output_dir", type=str, default=".", help="Path to the output directory"
    )
    args = parser.parse_args()
    make_item_catalog(**vars(args))
