import torch
import torch.nn.functional as F
from tqdm import tqdm

from .model import Net


def train(loader, model, optimizer, device, epoch):
    model.train()
    with tqdm(loader) as pbar:
        for x, label in pbar:
            pbar.set_description(f"[Epoch {epoch + 1}]")

            x, label = x.to(device), label.to(device)
            optimizer.zero_grad()
            y = model(x)
            loss = F.nll_loss(y, label)
            loss.backward()
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

    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(args.epochs):
        train(train_loader, model, optimizer, device, epoch)
        test(test_loader, model, device)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=14, metavar="N", help="number of epochs to train (default: 14)")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)")
    parser.add_argument(
        "--gamma", type=float, default=0.7, metavar="M", help="Learning rate step gamma (default: 0.7)"
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument("--save-model", action="store_true", default=False, help="For Saving the current Model")

    args = parser.parse_args()
    main(args)
