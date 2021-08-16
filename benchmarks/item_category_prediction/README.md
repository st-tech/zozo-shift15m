# Item category prediction

## Install extra requires

```bash
poetry install -E pytorch
```

## Run

```bash
poetry run python benchmarks/item_category_prediction/main.py 
```

If use `--make_dataset` option, it makes two files `train.txt` and `test.txt` and puts them in the data root. These have 3 columns.
The first column represents the item ID, which corresponds to the file name of the feature file. The second and third are the category and subcategory of this item, respectively.

