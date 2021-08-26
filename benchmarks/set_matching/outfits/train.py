import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import argparse
import chainer

from chainer import training
from chainer.training import extensions
from outfits.config import get_trainer_conf, get_model_conf
from outfits.dataset import get_train_val_dataset


def main(args):
    device = args.device
    
    # model
    model_config = get_model_conf(args.model)
    if args.model == 'set_matching_sim':
        from outfits.model import SetMatching
        model = SetMatching(**model_config)
    elif args.model in ['cov_mean', 'cov_max']:
        from outfits.model import SetMatchingCov
        model = SetMatchingCov(**model_config)
        model.load_weight(os.path.join(args.weight, 'model.npz'))
    else:
        raise ValueError('unknown model.')
    if device >= 0:
        model.to_gpu(device)

    # optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=0.001, momentum=0.5)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.00004))

    # dataset
    train_data, valid_data = get_train_val_dataset(args.input_dir, args.label_dir)

    # iterator
    train_iter = chainer.iterators.MultiprocessIterator(train_data,
            batch_size=args.batchsize, repeat=True, shuffle=True)
    valid_iter = chainer.iterators.MultiprocessIterator(valid_data,
            batch_size=args.batchsize, repeat=False, shuffle=False)

    # updater
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=device)

    trainer_config = get_trainer_conf()
    stop_trigger = (args.epoch, 'epoch')
    val_interval = trainer_config['val_interval']
    log_interval = trainer_config['log_interval']
    print_metrics = trainer_config['print_metrics']

    trainer = training.Trainer(updater, stop_trigger, out=args.out_dir)

    trainer.extend(
            extensions.Evaluator(valid_iter, model, device=device),
            trigger=val_interval)

    if 'lr_decay' in trainer_config:
        ld_conf = trainer_config['lr_decay']
        lr_shift_interval = ld_conf['freq_lr_shift']
        rate_lr_shift = ld_conf['rate_lr_shift']
        trainer.extend(
                extensions.ExponentialShift('lr', rate_lr_shift),
                trigger=lr_shift_interval)

    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport(
        ['iteration', 'epoch', 'elapsed_time', 'lr'] + print_metrics),
        trigger=log_interval)

    trainer.run()

    # save final results
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)
    chainer.serializers.save_npz(os.path.join(args.out_dir, "model.npz"), model)
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=4)
    parser.add_argument('--epoch', '-e', type=int, default=1)
    parser.add_argument(
        '--model',
        '-m', 
        choices=[
            "set_matching_sim",
            "cov_mean",
            "cov_max",
        ],
        default='cov_max',
    )
    # channel
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--label_dir', type=str)
    parser.add_argument('--weight', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--device', type=int, default=-1) # -1 for cpu or indicate gpu id >= 0
    args = parser.parse_args()

    main(args)
