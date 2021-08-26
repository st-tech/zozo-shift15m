trainer_conf = {
    "val_interval": (4, 'epoch'),
    "log_interval": (50, 'iteration'),
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
    "set_matching_sim": {
        "n_units": 512,
        "n_encoder_layer": 1,
        "n_decoder_layer": 1,
        "h": 8,
        "n_iterative": 2,
        "enc_apply_ln": True,
        "dec_apply_ln": True,  # only for MAB layer
        "dec_component": "MHSim",
    },
    "cov_mean": {
        "n_units": 512,
        "n_encoder_layer": 1,
        "n_decoder_layer": 1,
        "h": 8,
        "n_iterative": 2,
        "enc_apply_ln": True,
        "dec_apply_ln": True,  # only for MAB layer
        "dec_component": "MHSim",
        "weight": "mean",
        "logits": True,
    },
    "cov_max": {
        "n_units": 512,
        "n_encoder_layer": 1,
        "n_decoder_layer": 1,
        "h": 8,
        "n_iterative": 2,
        "enc_apply_ln": True,
        "dec_apply_ln": True,  # only for MAB layer
        "dec_component": "MHSim",
        "weight": "max",
        "logits": True,
    },
}

def get_trainer_conf():
    return trainer_conf

def get_model_conf(model_name):
    return model_conf[model_name]
