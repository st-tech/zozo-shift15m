import json
import pathlib
from datetime import datetime

import shift15m.constants as C
from shift15m.constants import Keys as K


def make_item_catalog(inp: str, output_dir: str = C.ROOT):
    if not pathlib.Path(inp).exists():
        import requests

        url = "https://research.zozo.com/data_release/shift15m/label/iqon_outfits.json"
        print(f"download {url} to {inp}.")
        r = requests.get(url)
        with open(inp, "w") as f:
            json.dump(r.json(), f, indent=2)

    data = json.load(open(inp))
    items_mst = {}
    for _set in data:
        datestr = _set[K.PUBLISH_DATE]
        year = datetime.strptime(datestr, "%Y-%m-%d").year
        for item in _set[K.ITEMS]:
            item_id = item[K.ITEM_ID]
            if item_id in items_mst:
                items_mst[item_id][C.ItemCatalog.YEAR] = min(
                    items_mst[item_id][C.ItemCatalog.YEAR], year
                )
            else:
                items_mst[item_id] = {
                    K.CATEGORY_ID: item[K.CATEGORY_ID],
                    K.SUBCATEGORY_ID: item[K.SUBCATEGORY_ID],
                    C.ItemCatalog.YEAR: year,
                }

    items = [
        f"{k} {v[K.CATEGORY_ID]} {v[K.SUBCATEGORY_ID]} {v[C.ItemCatalog.YEAR]}"
        for k, v in items_mst.items()
    ]

    with open(pathlib.Path(output_dir) / C.ItemCatalog.DEFAULT_FILE, "w") as f:
        f.write("\n".join(items))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("inp", type=str, help="path to the input file")
    args = parser.parse_args()
    make_item_catalog(**vars(args))
