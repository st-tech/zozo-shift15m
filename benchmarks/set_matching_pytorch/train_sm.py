import json
import os

import set_matching.extensions as exfn
import shift15m.constants as C
import torch
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping, ModelCheckpoint
from ignite.metrics import Loss, RunningAverage
from set_matching.metrics import NPairsAccuracy
from tensorboardX import SummaryWriter

from outfits.config import get_model_conf
from outfits.dataset import get_train_val_loader


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model
    model_config = get_model_conf(args.model)
    if args.model == "set_matching_sim":
        from set_matching.models.set_matching import SetMatching

        model = SetMatching(**model_config)
        loss_fn = torch.nn.CrossEntropyLoss(reduce=True).to(device)
    elif args.model in ["cov_mean", "cov_max"]:
        from model import SetMatchingCov

        model_config["pretrained_weight"] = args.weight_path
        model = SetMatchingCov(**model_config)
        loss_fn = torch.nn.CrossEntropyLoss(reduce=False).to(device)
    else:
        raise ValueError("unknown model.")
    model.to(device)

    # optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.001, momentum=0.5, weight_decay=0.00004
    )

    # dataset
    train_loader, valid_loader = get_train_val_loader(
        args.input_dir, args.label_dir, args.batchsize
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
            score = model(*batch)
            return score, torch.arange(score.size()[0]).to(device)

    trainer = Engine(train_process)
    train_evaluator = Engine(eval_process)
    valid_evaluator = Engine(eval_process)
    train_history = {"loss": [-1], "acc": [-1], "iteration": [-1], "epoch": [-1]}
    valid_history = {"loss": [-1], "acc": [-1], "iteration": [-1], "epoch": [-1]}

    # metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    NPairsAccuracy().attach(train_evaluator, "acc")
    Loss(loss_fn).attach(train_evaluator, "loss")
    NPairsAccuracy().attach(valid_evaluator, "acc")
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
    parser.add_argument("--batchsize", "-b", type=int, default=4)
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
    # trainer
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--log_dir", type=str, default="/tmp/ml/set_matching/")
    parser.add_argument("--checkpoint_interval", type=int, default=2)
    # channel
    parser.add_argument("--input_dir", "-i", type=str, default=C.ROOT)
    parser.add_argument("--label_dir", "-l", type=str)
    parser.add_argument("--weight_path", "-w", type=str)

    args = parser.parse_args()

    main(args)
