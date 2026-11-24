from src import config

R0 = {
    "name": "r0", "architecture": "cnn", "blocks": 3, "base_filters": 32, "convs_per_block": 2,
    "batch_norm": True, "dropout": 0.3, "l2": 1e-4, "pooling": "gap", "dense_units": 128,
    "optimizer": "adam", "learning_rate": 1e-3, "weight_decay": 0.0, "batch_size": 32,
    "image_size": 128, "augment": "medium", "class_weights": True, "oversample": False,
    "lr_schedule": True, "label_smoothing": 0.0, "shuffle_labels": False, "train_fraction": 1.0,
    "max_epochs": config.MAX_EPOCHS, "early_stop_patience": config.EARLY_STOP_PATIENCE,
}

BASELINE = {
    **R0, "name": "baseline", "architecture": "baseline", "blocks": 2, "convs_per_block": 1,
    "batch_norm": False, "dropout": 0.0, "l2": 0.0, "pooling": "flatten", "augment": "none",
    "class_weights": False,
}

R0_SHUFFLED_LABELS = {**R0, "name": "r0_shuffled_labels", "shuffle_labels": True}

OVERFIT_CHECK = {**R0, "name": "r0_overfit_one_batch", "augment": "none", "dropout": 0.0, "l2": 0.0,
                 "class_weights": False}


def variant(base, name, **changes):
    unknown = set(changes) - set(base)
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}")
    return {**base, **changes, "name": name}


def section_groups(section):
    groups = {
        3: {
            "A": {"configs": [R0_SHUFFLED_LABELS], "seeds": [11], "overfit_check": True},
            "B": {"configs": [BASELINE], "seeds": config.SEEDS},
            "C": {"configs": [R0], "seeds": config.SEEDS},
        },
    }
    if section not in groups:
        raise KeyError(f"no grid defined for section {section}")
    return groups[section]
