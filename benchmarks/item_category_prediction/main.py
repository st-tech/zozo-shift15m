try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as e:
    msg = (
        "This benchmark requires PyTorch.\n"
        "Please install extra pachages as follows:\n"
        "  poetry install -E pytorch"
    )
    raise ModuleNotFoundError(msg) from e
import itertools
import pathlib
import random
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import shift15m.constants as C
from shift15m.datasets import get_imagefeature_dataloader
from shift15m.datasets.helper import make_item_catalog
from shift15m.datasets.imagefeature_torch import ItemCatalog
from tqdm import tqdm


def load_item_catalog(args: Any) -> ItemCatalog:
    if args.make_dataset:
        inp = next(
            pathlib.Path(C.ROOT).glob("*.json")
        )  # Assume that the source json file is located in `C.ROOT`
        make_item_catalog(inp)
    catalog_path = pathlib.Path(C.ROOT) / "item_catalog.txt"
    return ItemCatalog(catalog_path)


def get_model(n_outputs: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(4096, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, n_outputs),
    )


def _plot_matrix(results: List[Dict]):
    cm = np.zeros((len(C.YEAES), len(C.YEAES)))
    years_rev = {y: i for i, y in enumerate(C.YEAES)}
    for res in results:
        i, j = years_rev[res["train_year"]], years_rev[res["test_year"]]
        cm[i, j] = res["test_acc"]

    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(C.YEAES))
    plt.xticks(tick_marks, C.YEAES, rotation=45)
    plt.yticks(tick_marks, C.YEAES)

    thresh = cm.max() / 1.3
    for res in results:
        i, j = years_rev[res["train_year"]], years_rev[res["test_year"]]
        plt.text(
            i,
            j,
            f"{res['train_acc']:.1f}/\n{res['test_acc']:.1f}",
            horizontalalignment="center",
            verticalalignment="center",
            color="white" if cm[i, j] > thresh else "gray",
            size=8,
        )

    plt.ylabel("test year")
    plt.xlabel("train year")
    plt.tight_layout()
    plt.savefig(
        "classification_acc_by_year.png", orientation="portrait", pad_inches=0.1
    )


def train(
    loader: torch.utils.data.DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    weight: torch.Tensor = None,
):
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    with tqdm(loader) as pbar:
        for x, label in pbar:
            pbar.set_description(f"[Epoch {epoch + 1}]")

            x, label = x.to(device), label.to(device)
            optimizer.zero_grad()
            y = model(x)
            loss = loss_fn(y, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            pbar.set_postfix({"loss": loss.item()})


def test(
    loader: torch.utils.data.DataLoader, model: nn.Module, device: str, name: str = None
) -> float:
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = 0
    correct = 0
    with torch.no_grad():
        for x, label in loader:
            x, label = x.to(device), label.to(device)
            y = model(x)
            loss += loss_fn(y, label).sum().item()
            pred = y.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

    loss /= len(loader.dataset)

    if name:
        print(
            f"{name} set: Average loss: {loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({100. * correct / len(loader.dataset):.0f}%)"
        )

    return correct / len(loader.dataset)


def run(
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    args: Any,
) -> Tuple[float, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    count = train_loader.dataset.category_count
    print(
        f"category size: {train_loader.dataset.category_size}, category count: {count}"
    )

    count = torch.tensor(count, dtype=torch.float32, device=device)
    weight = count.max() / count

    model = get_model(n_outputs=train_loader.dataset.category_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00004)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    for epoch in range(args.epochs):
        train(train_loader, model, optimizer, device, epoch, weight=weight)
        _ = test(valid_loader, model, device, "Valid")
        _ = test(test_loader, model, device, "Test")
        scheduler.step()

    return test(valid_loader, model, device), test(test_loader, model, device)


def main(args: Any):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    item_catalog = load_item_catalog(args)

    results = []
    for train_val_year, test_year in itertools.product(C.YEAES, C.YEAES):
        if train_val_year == test_year:
            continue

        train_items, valid_items, test_items = item_catalog.get_train_valid_test_items(
            args.target,
            train_valid_year=train_val_year,
            test_year=test_year,
            train_size=args.train_size,
            valid_size=args.val_test_size,
            test_size=args.val_test_size,
        )

        train_loader = get_imagefeature_dataloader(
            train_items,
            args.target,
            args.data_dir,
            args.batch_size,
            is_train=True,
        )
        valid_loader = get_imagefeature_dataloader(
            valid_items,
            args.target,
            args.data_dir,
            args.batch_size,
            is_train=False,
        )
        test_loader = get_imagefeature_dataloader(
            test_items,
            args.target,
            args.data_dir,
            args.batch_size,
            is_train=False,
        )
        print(
            f"\nstart training: trainval year {train_val_year} / test year {test_year}\n"
        )
        trainyear_acc, testyear_acc = run(train_loader, valid_loader, test_loader, args)
        results.append(
            {
                "train_year": train_val_year,
                "test_year": test_year,
                "train_acc": trainyear_acc * 100,
                "test_acc": testyear_acc * 100,
            }
        )

    _plot_matrix(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--make_dataset",
        action="store_true",
        help="it true, make train and test catalogs.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=C.FEATURE_ROOT,
        help="path to the feature directory",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["category", "subcategory"],
        default="category",
        help="target label (categoy / subcategory)",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=3500,
        help="train dataset size",
    )
    parser.add_argument(
        "--val_test_size",
        type=int,
        default=500,
        help="valid and test dataset size",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.005,
        help="learning rate (default: 0.05)",
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")

    args = parser.parse_args()
    main(args)
