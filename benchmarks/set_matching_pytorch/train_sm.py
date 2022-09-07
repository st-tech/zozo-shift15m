import json
import os
from typing import Any, Optional, Tuple, Union

import set_matching.extensions as exfn
import shift15m.constants as C
import torch
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping, ModelCheckpoint
from ignite.metrics import Loss, RunningAverage
from set_matching.metrics import NPairsAccuracy
from shift15m.datasets.outfitfeature import (
    IQONOutfits,
    MultisetSplitDataset,
    get_loader,
)
from tensorboardX import SummaryWriter

from config import get_model_conf


def get_train_val_loader(
    train_year: Union[str, int],
    valid_year: Union[str, int],
    split: int,
    batch_size: int,
    root: str = C.ROOT,
    num_workers: Optional[int] = None,
) -> Tuple[Any, Any]:
    label_dir_name = f"{train_year}-{valid_year}-split{split}"

    iqon_outfits = IQONOutfits(root=root)

    train, valid = iqon_outfits.get_trainval_data(label_dir_name)
    feature_dir = iqon_outfits.feature_dir
    train_dataset = MultisetSplitDataset(train, feature_dir, n_sets=1, n_drops=None)
    valid_dataset = MultisetSplitDataset(valid, feature_dir, n_sets=1, n_drops=None)
    return (
        get_loader(train_dataset, batch_size, num_workers=num_workers, is_train=True),
        get_loader(valid_dataset, batch_size, num_workers=num_workers, is_train=False),
    )


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model
    model_config = get_model_conf(args.model)
    if args.model == "set_matching_sim":
        from set_matching.models.set_matching import SetMatching

        model = SetMatching(**model_config)
        loss_fn = torch.nn.CrossEntropyLoss(reduce=True).to(device)
        loss_fn_eval = torch.nn.CrossEntropyLoss(reduce=True).to(device)
    elif args.model in ["cov_mean", "cov_max"]:
        from model import SetMatchingCov

        model_config["pretrained_weight"] = args.weight_path
        model = SetMatchingCov(**model_config)
        loss_fn = torch.nn.CrossEntropyLoss(reduce=False).to(device)
        loss_fn_eval = torch.nn.CrossEntropyLoss(reduce=True).to(device)
    else:
        raise ValueError("unknown model.")
    model.to(device)

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
        batch = tuple(map(lambda x: x.to(device), batch))
        out = model(*batch)
        if isinstance(out, tuple):
            score, importance = out
        else:
            score, importance = out, None
        loss = loss_fn(score, torch.arange(score.size()[0]).to(device))
        if importance is not None:
            loss = torch.mean(loss * importance)
        loss.backward()
        optimizer.step()
        return loss.item()

    def eval_process(engine, batch):
        model.eval()
        with torch.inference_mode():
            batch = tuple(map(lambda x: x.to(device), batch))
            out = model(*batch)
            if isinstance(out, tuple):
                score, _ = out
            else:
                score, _ = out, None
            return score, torch.arange(score.size()[0]).to(device)

    trainer = Engine(train_process)
    train_evaluator = Engine(eval_process)
    valid_evaluator = Engine(eval_process)
    train_history = {"loss": [-1], "acc": [-1], "iteration": [-1], "epoch": [-1]}
    valid_history = {"loss": [-1], "acc": [-1], "iteration": [-1], "epoch": [-1]}

    # metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    NPairsAccuracy().attach(train_evaluator, "acc")
    Loss(loss_fn_eval).attach(train_evaluator, "loss")
    NPairsAccuracy().attach(valid_evaluator, "acc")
    Loss(loss_fn_eval).attach(valid_evaluator, "loss")

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
        Events.EPOCH_COMPLETED(every=4),
        exfn.log_training_results,
        "Training",
        train_evaluator,
        train_loader,
        train_history,
        writer,
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=4),
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
    with open(os.path.join(args.log_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=2)
    with open(os.path.join(args.log_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        choices=[
            "set_matching_sim",
            "cov_mean",
            "cov_max",
        ],
        default="cov_max",
    )
    parser.add_argument("--batchsize", "-b", type=int, default=32)
    parser.add_argument("--epochs", "-e", type=int, default=10)
    parser.add_argument("--log_dir", "-o", type=str, default="/tmp/ml/set_matching/")
    parser.add_argument("--checkpoint_interval", type=int, default=2)

    parser.add_argument("--train_year", type=int)
    parser.add_argument("--valid_year", type=int)
    parser.add_argument("--split", type=int, choices=[0, 1, 2])
    parser.add_argument("--weight_path", "-w", type=str, default=None)

    args = parser.parse_args()

    main(args)
