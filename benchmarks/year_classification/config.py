trainer_conf = {
    "val_interval": (4, 'epoch'),
    "log_interval": (4, 'epoch'),
    "lr_decay": {
        "freq_lr_shift": (4, 'epoch'),
        "rate_lr_shift": 0.7
    },
    "print_metrics": [
        "main/loss",
        "main/acc",
        "validation/main/loss",
        "validation/main/acc",],
    "plot_report": {
        "plot_loss": {
            "metrics": ["main/loss", "validation/main/loss"],
            "axis": "epoch"
        },
        "plot_acc": {
            "metrics": ["main/acc", "validation/main/acc"],
            "axis": "epoch"
        },
    }
}

model_conf = {
    "two_layered_cnn": {"n_units": 512},
}

def get_trainer_conf():
    return trainer_conf

def get_model_conf(model_name):
    return model_conf[model_name]
