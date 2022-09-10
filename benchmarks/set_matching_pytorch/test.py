import argparse
import json
import pathlib
from typing import Any, Optional, Union

import shift15m.constants as C
import torch
from set_matching.models.set_matching import SetMatching
from shift15m.datasets.outfitfeature import FINBsDataset, IQONOutfits, get_loader
from tqdm import tqdm

from model import SetMatchingCov

MODELS = {
    "set_matching_sim": SetMatching,
    "cov_mean": SetMatchingCov,
    "cov_max": SetMatchingCov,
}


def get_test_loader(
    train_year: Union[str, int],
    valid_year: Union[str, int],
    split: int,
    n_cand_sets: int,
    batch_size: int,
    root: str = C.ROOT,
    num_workers: Optional[int] = None,
) -> torch.utils.data.DataLoader:
    label_dir_name = f"{train_year}-{valid_year}-split{split}"

    iqon_outfits = IQONOutfits(root=root, split=split)

    test_examples = iqon_outfits.get_fitb_data(label_dir_name, n_comb=args.n_comb)
    feature_dir = iqon_outfits.feature_dir
    dataset = FINBsDataset(
        test_examples,
        feature_dir,
        n_cand_sets=n_cand_sets,
        max_set_size_query=6,
        max_set_size_answer=6,
    )
    return get_loader(dataset, batch_size, num_workers=num_workers, is_train=False)


def model_fn(model_dir: str, device: str) -> Any:
    model_name = json.load(open(pathlib.Path(model_dir) / "args.json"))["model"]
    model_config = json.load(open(pathlib.Path(model_dir) / "model_config.json"))

    model = MODELS[model_name](**model_config)

    with open(pathlib.Path(model_dir) / "model.pt", "rb") as f:
        model.load_state_dict(torch.load(f))
    model.eval()
    model.to(device)

    return model


def predict_fn(inputs: Any, model: Any) -> Any:
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

    out = model(query, q_mask, candidates, c_mask)  # (batch*n_cands, batch*n_cands)
    if isinstance(out, tuple):
        score = out[0]
    else:
        score = out
    score = torch.diagonal(score, 0).view(batch, n_candidates)
    pred = score.argmax(dim=1)

    return pred, torch.softmax(score, dim=1)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model_fn(args.model_dir, device)
    test_loader = get_test_loader(args.train_year, args.valid_year, args.split, 4, 32)

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
    parser.add_argument("--split", type=int, choices=[0, 1, 2])
    parser.add_argument("--model_dir", "-d", type=str)
    parser.add_argument("--n_comb", type=int, default=1)
    args = parser.parse_args()

    main(args)
