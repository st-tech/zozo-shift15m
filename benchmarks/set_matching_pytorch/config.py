model_conf = {
    "set_matching_sim": {
        "n_units": 512,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "n_heads": 8,
        "n_iterative": 2,
        "enc_apply_ln": True,
        "dec_apply_ln": True,  # only for MAB layer
        "dec_component": "MHSim",
        "embedder_arch": "linear",
    },
    "cov_mean": {
        "n_units": 512,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "n_heads": 8,
        "n_iterative": 2,
        "enc_apply_ln": True,
        "dec_apply_ln": True,  # only for MAB layer
        "dec_component": "MHSim",
        "embedder_arch": "linear",
        "weight": "mean",
        "logits": True,
    },
    "cov_max": {
        "n_units": 512,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "n_heads": 8,
        "n_iterative": 2,
        "enc_apply_ln": True,
        "dec_apply_ln": True,  # only for MAB layer
        "dec_component": "MHSim",
        "embedder_arch": "linear",
        "weight": "max",
        "logits": True,
    },
}


def get_model_conf(model_name):
    return model_conf[model_name]
