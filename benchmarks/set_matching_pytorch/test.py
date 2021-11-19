import argparse
import json
import pathlib

import torch
from set_matching.models.set_matching import SetMatching
from shift15m.datasets.outfitfeature import get_test_loader
from tqdm import tqdm

from model import SetMatchingCov

MODELS = {
    "set_matching_sim": SetMatching,
    "cov_mean": SetMatchingCov,
    "cov_max": SetMatchingCov,
}


def model_fn(model_dir, device):
    model_name = json.load(open(pathlib.Path(model_dir) / "args.json"))["model"]
    model_config = json.load(open(pathlib.Path(model_dir) / "model_config.json"))

    model = MODELS[model_name](**model_config)

    with open(pathlib.Path(model_dir) / "model.pt", "rb") as f:
        model.load_state_dict(torch.load(f))
    model.eval()
    model.to(device)

    return model


def predict_fn(inputs, model):
    query, q_mask, candidates, c_mask = inputs
    query_set_size = query.shape[1]
    (
        batch,
        n_candidates,
        cand_set_size,
    ) = candidates.shape[:3]

    query = (
        torch.broadcast_to(query, (n_candidates,) + query.shape)
        .permute((1, 0) + tuple(range(2, len(candidates.shape))))
        .reshape((-1, query_set_size) + candidates.shape[3:])
    )
    q_mask = (
        torch.broadcast_to(q_mask, (n_candidates,) + q_mask.shape)
        .permute(1, 0, 2)
        .reshape(-1, query_set_size)
    )
    candidates = candidates.view((-1, cand_set_size) + candidates.shape[3:])
    c_mask = c_mask.view(-1, cand_set_size)

    score = model(query, q_mask, candidates, c_mask)  # (batch*n_cands, batch*n_cands)
    score = torch.diagonal(score, 0).view(batch, n_candidates)
    pred = score.argmax(dim=1)

    return pred, torch.softmax(score, dim=1)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model_fn(args.model_dir, device)
    test_loader = get_test_loader(args.train_year, args.valid_year, 4, 32)

    correct, count = 0, 0
    with torch.inference_mode():
        for batch in tqdm(test_loader):
            batch = tuple(map(lambda x: x.to(device), batch))
            pred, _ = predict_fn(batch, model)
            correct += pred.eq(torch.zeros_like(pred)).sum().item()
            count += len(pred)

    print("\naccuracy: " + str(correct / count * 100) + " %")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_year", type=int)
    parser.add_argument("--valid_year", type=int)
    parser.add_argument("--model_dir", "-d", type=str)
    args = parser.parse_args()

    main(args)
