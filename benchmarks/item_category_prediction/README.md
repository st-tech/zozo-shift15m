# Item category prediction

## Install extra requires

```bash
poetry install -E pytorch
```

## Prepare datasets for category prediction task

TBD

```bash
poetry run python make_dataset.py <path to json file>
```

This command makes two files `train.txt` and `test.txt`. These have 3 columns.
The first column represents the item ID, which corresponds to the file name of the feature file. The second and third are the category and subcategory of this item, respectively.


## Run

```bash
poetry run python main.py <path to train.txt> <path to test.txt> <path to feature files directory> --target <target label (category / subcategory)> 
```