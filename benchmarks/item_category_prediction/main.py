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
import pathlib
from typing import Any

from tqdm import tqdm

import shift15m.constants as C
from shift15m.datasets import get_imagefeature_dataloader
from shift15m.datasets.helper import make_item_catalog


def get_model(n_outputs: int):
    return nn.Sequential(
        nn.Linear(4096, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, n_outputs),
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


def test(loader: torch.utils.data.DataLoader, model: nn.Module, device: str):
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

    print(
        f"\nTest set: Average loss: {loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({100. * correct / len(loader.dataset):.0f}%)\n"
    )


def main(args: Any):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.make_dataset:
        inp = next(
            pathlib.Path(C.ROOT).glob("*.json")
        )  # Assume that the source json file is located in `C.ROOT`
        make_item_catalog(inp)
        inp_train, inp_test = (
            pathlib.Path(C.ROOT) / "train.txt",
            pathlib.Path(C.ROOT) / "test.txt",
        )
    else:
        assert args.inp_train is not None and args.inp_test is not None
        inp_train, inp_test = args.inp_train, args.inp_test

    train_loader = get_imagefeature_dataloader(
        inp_train, args.data_dir, args.target, args.batch_size, is_train=True
    )
    test_loader = get_imagefeature_dataloader(
        inp_test, args.data_dir, args.target, args.batch_size, is_train=False
    )
    assert train_loader.dataset.category_size == test_loader.dataset.category_size
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
        test(test_loader, model, device)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--make_dataset",
        action="store_true",
        help="it true, make train and test catalogs.",
    )
    parser.add_argument(
        "--inp_train", type=str, default=None, help="path to the input file for train"
    )
    parser.add_argument(
        "--inp_test", type=str, default=None, help="path to the input file for test"
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
        "--batch_size",
        type=int,
        default=128,
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
        default=0.05,
        help="learning rate (default: 0.05)",
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )

    args = parser.parse_args()
    main(args)
