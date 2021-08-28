import json
import pathlib
from datetime import datetime

import shift15m.constants as C
from shift15m.constants import Keys as K


def make_item_catalog(inp: str, output_dir: str = C.ROOT):
    data = json.load(open(inp))
    items_mst = {}
    for _set in data:
        datestr = _set[K.PUBLISH_DATE]
        year = datetime.strptime(datestr, "%Y-%m-%d").year
        for item in _set[K.ITEMS]:
            item_id = item[K.ITEM_ID]
            if item_id in items_mst:
                items_mst[item_id]["year"] = min(items_mst[item_id]["year"], year)
            else:
                items_mst[item_id] = {
                    K.CATEGORY_ID: item[K.CATEGORY_ID],
                    K.SUBCATEGORY_ID: item[K.SUBCATEGORY_ID],
                    "year": year,
                }

    items = [
        f"{k} {v[K.CATEGORY_ID]} {v[K.SUBCATEGORY_ID]} {v['year']}"
        for k, v in items_mst.items()
    ]

    with open(pathlib.Path(output_dir) / "item_catalog.txt", "w") as f:
        f.write("\n".join(items))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("inp", type=str, help="path to the input file")
    args = parser.parse_args()
    make_item_catalog(**vars(args))
