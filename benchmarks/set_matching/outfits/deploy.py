import json
import numpy as np
import os
import chainer
from chainer.backend import cuda
from chainer.dataset import concat_examples

from outfits.config import get_model_conf
from outfits.dataset import TransformFIMBsDataset


def model_fn(model_dir, device):
    model_name = json.load(open(os.path.join(model_dir, "args.json")))["model"]
    model_config = get_model_conf(model_name)

    if model_name == "set_matching_sim":
        from outfits.model import SetMatching

        model = SetMatching(**model_config)
    elif model_name in ["cov_mean", "cov_max"]:
        from outfits.model import SetMatchingCov

        model = SetMatchingCov(**model_config)
    else:
        raise ValueError("unknown model.")

    chainer.serializers.load_npz(os.path.join(model_dir, "model.npz"), model)
    if device >= 0:
        model.to_gpu(device)

    return model


def input_fn(request_body, request_content_type, device):
    assert request_content_type == "application/json"
    input_object = json.loads(request_body)

    query, answers = [], []
    for feature in input_object["query"]:
        query.append(np.array(feature, dtype=np.float32))

    for cand in input_object["answers"]:
        c_feature = []
        for feature in cand:
            c_feature.append(np.array(feature, dtype=np.float32))
        answers.append(c_feature)

    transformer = TransformFIMBsDataset(True)
    input_object = transformer((query, answers))
    input_object = concat_examples([input_object], device)
    return input_object


def predict_fn(input_object, model):
    with chainer.using_config("train", False), chainer.no_backprop_mode():
        prob = model.predict(*input_object)
    prediction = cuda.to_cpu(prob.data)
    return prediction


def output_fn(prediction, accept):
    return json.dumps(list(prediction.astype(np.float64))), accept
