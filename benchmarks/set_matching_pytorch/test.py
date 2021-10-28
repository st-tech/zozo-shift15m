import argparse
import gzip
import json
import pathlib

import numpy as np
import torch
from set_matching.datasets.transforms import FeatureListTransform
from set_matching.models.set_matching import SetMatching
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

    transform = FeatureListTransform(
        max_set_size=5,
        apply_shuffle=False,
        apply_padding=True,
    )

    return model, transform


def input_fn(request_body):
    input_object = json.loads(request_body)

    query, answers = [], []
    for feature in input_object["query"]:
        query.append(np.array(feature, dtype=np.float32))

    for cand in input_object["answers"]:
        c_feature = []
        for feature in cand:
            c_feature.append(np.array(feature, dtype=np.float32))
        answers.append(c_feature)

    return query, answers


def _to_tensor(x):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.tensor(x[None, :]).to(device)


def _predict_fn(inputs, model):
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


def predict_fn(input_object, model):
    query, target = input_object
    model, transform = model

    query, _, q_mask = transform(query, [0] * len(query))
    answers, a_masks = [], []
    for answer in target:
        ans, _, a_mask = transform(answer, [0] * len(answer))
        answers.append(ans)
        a_masks.append(a_mask)
    x = tuple(map(_to_tensor, [query, q_mask, np.array(answers), np.array(a_masks)]))

    with torch.inference_mode():
        _, prediction = _predict_fn(x, model)
    return prediction.cpu().detach().numpy()[0]


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model_fn(args.model_dir, device)
    testdata_path = pathlib.Path(args.input_dir).glob("*.json.gz")

    ans_list = []
    for path in tqdm(testdata_path):
        with gzip.open(path, mode="rt", encoding="utf-8") as f:
            data = f.read()

        input_objects = input_fn(data)
        pred = predict_fn(input_objects, model)
        ans = np.argmax(pred) == 0
        ans_list.append(ans)

    print("\naccuracy: " + str(np.mean(ans_list) * 100) + "%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", type=str)
    parser.add_argument("--model_dir", "-d", type=str)
    args = parser.parse_args()

    main(args)
