import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import set_matching.extensions as exfn
import shift15m.constants as C
import torch
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping, ModelCheckpoint
from ignite.metrics import Loss, RunningAverage
from shift15m.datasets.outfitfeature import FeatureLabelDataset, IQONOutfits, get_loader
from tensorboardX import SummaryWriter

from metrics import BinaryAccuracy
from model import TwoLayeredLinear


def _get_items(sets: List[Dict]) -> List:
    item_ids = []
    for set_ in sets:
        item_ids.extend([item["item_id"] for item in set_["items"]])
    return item_ids


def get_train_val_loader(
    train_year: Union[str, int],
    valid_year: Union[str, int],
    split: int,
    batch_size: int,
    root: str = C.ROOT,
    num_workers: Optional[int] = None,
) -> Tuple[Any, Any]:
    iqon_outfits = IQONOutfits(
        train_year=train_year, valid_year=valid_year, split=split, root=root
    )

    train, valid = iqon_outfits.get_trainval_data()
    feature_dir = iqon_outfits.feature_dir

    train = _get_items(train)
    valid = _get_items(valid)

    common_id = set(train) & set(valid)
    traindiff_id = list(set(train) - common_id)
    validdiff_id = list(set(valid) - common_id)
    sampling_num = len(traindiff_id) - len(validdiff_id)
    duplicated_ind = np.random.choice(len(validdiff_id), sampling_num, replace=True)
    duplicated_valid_id = [validdiff_id[i] for i in duplicated_ind]
    validdiff_id = validdiff_id + duplicated_valid_id
    train_data = [{"item_id": d, "label": np.int32(1)} for d in traindiff_id] + [
        {"item_id": d, "label": np.int32(0)} for d in validdiff_id
    ]
    print("train data created")

    test = iqon_outfits.get_test_data()
    test = _get_items(test)
    common_id = set(train) & set(test)
    testdiff_id = list(set(test) - common_id)
    removed = np.random.choice(len(traindiff_id), len(testdiff_id), replace=False)
    removed_train_id = [traindiff_id[i] for i in removed]
    valid_data = [{"item_id": d, "label": np.int32(1)} for d in removed_train_id] + [
        {"item_id": d, "label": np.int32(0)} for d in testdiff_id
    ]
    print("test data created")

    train_dataset = FeatureLabelDataset(train_data, feature_dir)
    valid_dataset = FeatureLabelDataset(valid_data, feature_dir)
    return (
        get_loader(train_dataset, batch_size, num_workers, is_train=True),
        get_loader(valid_dataset, batch_size, num_workers, is_train=False),
    )


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model
    model = TwoLayeredLinear(n_units=512)
    model.to(device)
    loss_fn = torch.nn.BCELoss().to(device)

    # optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.001, momentum=0.5, weight_decay=0.00004
    )

    # dataset
    train_loader, valid_loader = get_train_val_loader(
        args.train_year, args.valid_year, args.split, args.batchsize
    )

    # logger
    writer = SummaryWriter(logdir=args.log_dir)

    def train_process(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, t = tuple(map(lambda x: x.to(device), batch))
        score = model(x)
        loss = loss_fn(score, t.float())
        loss.backward()
        optimizer.step()
        return loss.item()

    def eval_process(engine, batch):
        model.eval()
        with torch.inference_mode():
            x, t = tuple(map(lambda x: x.to(device), batch))
            score = model(x)
            return score, t.float()

    trainer = Engine(train_process)
    train_evaluator = Engine(eval_process)
    valid_evaluator = Engine(eval_process)
    train_history = {"loss": [-1], "acc": [-1], "iteration": [-1], "epoch": [-1]}
    valid_history = {"loss": [-1], "acc": [-1], "iteration": [-1], "epoch": [-1]}

    # metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    BinaryAccuracy().attach(train_evaluator, "acc")
    Loss(loss_fn).attach(train_evaluator, "loss")
    BinaryAccuracy().attach(valid_evaluator, "acc")
    Loss(loss_fn).attach(valid_evaluator, "loss")

    # early stopping
    handler = EarlyStopping(
        patience=5,
        score_function=exfn.stopping_score_function,
        trainer=trainer,
    )
    valid_evaluator.add_event_handler(Events.COMPLETED, handler)

    # lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.7)
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        exfn.lr_step,
        lr_scheduler,
    )

    # logging
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=1000),
        exfn.log_training_loss,
        train_loader,
        writer,
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=1),
        exfn.log_training_results,
        "Training",
        train_evaluator,
        train_loader,
        train_history,
        writer,
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=1),
        exfn.log_training_results,
        "Validation",
        valid_evaluator,
        valid_loader,
        valid_history,
        writer,
    )

    # checkpoints
    objects_to_checkpoint = {
        "trainer": trainer,
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
    }
    trainer_checkpointer = Checkpoint(
        to_save=objects_to_checkpoint,
        save_handler=DiskSaver(args.log_dir, require_empty=False),
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=args.checkpoint_interval),
        trainer_checkpointer,
    )

    model_checkpointer = ModelCheckpoint(
        args.log_dir,
        "modelckpt",
        n_saved=1,
        create_dir=True,
        require_empty=False,
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=args.checkpoint_interval),
        model_checkpointer,
        {"model": model},
    )

    # kick everything off
    trainer.run(train_loader, max_epochs=args.epochs)

    writer.close()

    torch.save(model.cpu().state_dict(), os.path.join(args.log_dir, "model.pt"))
    with open(os.path.join(args.log_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", "-b", type=int, default=128)
    parser.add_argument("--epochs", "-e", type=int, default=16)
    parser.add_argument(
        "--log_dir", "-o", type=str, default="/tmp/ml/weight_estimation"
    )
    parser.add_argument("--checkpoint_interval", type=int, default=2)
    # channel
    parser.add_argument("--train_year", type=int)
    parser.add_argument("--valid_year", type=int)
    parser.add_argument("--split", type=int, choices=[0, 1, 2])
    args = parser.parse_args()

    np.random.seed(0)  # used for train/test splitting
    main(args)
