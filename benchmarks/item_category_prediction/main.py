import torch
import torch.nn.functional as F
from dataset import get_loader
from model import Net
from tqdm import tqdm


def train(loader, model, optimizer, device, epoch, weight=None):
    model.train()
    with tqdm(loader) as pbar:
        for x, label in pbar:
            pbar.set_description(f"[Epoch {epoch + 1}]")

            x, label = x.to(device), label.to(device)
            optimizer.zero_grad()
            y = model(x)
            loss = F.nll_loss(y, label, weight=weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            pbar.set_postfix({"loss": loss.item()})


def test(loader, model, device):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for x, label in loader:
            x, label = x.to(device), label.to(device)
            y = model(x)
            loss += F.nll_loss(y, label, reduction="sum").item()
            pred = y.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

    loss /= len(loader.dataset)

    print(
        f"\nTest set: Average loss: {loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({100. * correct / len(loader.dataset):.0f}%)\n"
    )


def main(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = get_loader(
        args.inp_train, args.data_dir, args.target, args.batch_size, is_train=True
    )
    test_loader = get_loader(
        args.inp_test, args.data_dir, args.target, args.batch_size, is_train=False
    )
    assert train_loader.dataset.category_size == test_loader.dataset.category_size
    count = train_loader.dataset.category_count
    print(
        f"category size: {train_loader.dataset.category_size}, category count: {count}"
    )

    count = torch.tensor(count, dtype=torch.float32, device=device)
    weight = count.max() / count

    model = Net(n_outputs=train_loader.dataset.category_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00004)

    for epoch in range(args.epochs):
        train(train_loader, model, optimizer, device, epoch, weight=weight)
        test(test_loader, model, device)

    if args.save_model:
        torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("inp_train", type=str, help="path to the input file for train")
    parser.add_argument("inp_test", type=str, help="path to the input file for test")
    parser.add_argument("data_dir", type=str, help="path to the feature directory")
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
