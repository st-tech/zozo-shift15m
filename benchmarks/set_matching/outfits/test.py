import os
import json
import numpy as np
import argparse
import gzip

from glob import glob
from tqdm import tqdm

from outfits.deploy import model_fn
from outfits.deploy import predict_fn
from outfits.deploy import input_fn
from outfits.deploy import output_fn


def inference(model, data, device):
    result, _ = output_fn(
        predict_fn(input_fn(data, "application/json", device), model), accept=True
    )
    return result


def load_data(path):
    with gzip.open(path, mode="rt", encoding="utf-8") as f:
        data = f.read()
    return data


def is_correct_answer(result):
    return (
        np.argmax(json.loads(result)) == 0
    )  # correct answer exists in the first position of each defined test data


def test(args):
    model = model_fn(args.model_dir, args.device)
    testdata_path = glob(os.path.join(args.input_dir, "*.json.gz"))

    ans_list = []
    for p in tqdm(testdata_path):
        data = load_data(p)
        result = inference(model, data, args.device)
        ans = is_correct_answer(result)
        ans_list.append(ans)
    print("\naccuracy: " + str(np.mean(ans_list) * 100) + "%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", type=str)
    parser.add_argument("--model_dir", "-d", type=str)
    parser.add_argument(
        "--device", "-gpu", type=int
    )  # -1 for cpu or indicate gpu id >= 0
    args = parser.parse_args()

    test(args)
