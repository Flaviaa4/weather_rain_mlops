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

SECTION_BUDGET_MINUTES = {5: 50}
ABLATION_STEPS = {
    "batch_norm": {"batch_norm": R0["batch_norm"]},
    "dropout": {"dropout": R0["dropout"]},
    "l2": {"l2": R0["l2"]},
    "class_weights": {"class_weights": R0["class_weights"]},
    "augment": {"augment": R0["augment"]},
    "gap": {"pooling": R0["pooling"]},
}


def variant(base, name, **changes):
    unknown = set(changes) - set(base) - set(config.OPTIONAL_CONFIG_DEFAULTS)
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}")
    return {**base, **changes, "name": name}


def ablation_configs():
    return [variant(BASELINE, f"baseline_plus_{name}", **changes) for name, changes in ABLATION_STEPS.items()]


def stall_fix_configs():
    return [variant(R0, "r0_bn_momentum_0.9", bn_momentum=0.9), variant(R0, "r0_bn_momentum_0.8", bn_momentum=0.8)]


def stopping_rule_configs():
    return [variant(R0, "r0_patience30_min20", early_stop_patience=30, min_epochs=20)]


def optimizer_and_augment_configs():
    return [variant(R0, "r0_lr3e-4", learning_rate=3e-4), variant(R0, "r0_warmup5", warmup_epochs=5),
            variant(R0, "r0_augment_light", augment="light")]


# chosen in Section 5 by the rule in experiments/DECISION.md, fixed before the test set was touched
def final_config():
    return stopping_rule_configs()[0]


def smoke_configs(section):
    if section == 3:
        return [BASELINE, R0, R0_SHUFFLED_LABELS]
    if section == 7:
        return [variant(final_config(), "smoke_final")]
    return [variant(R0, "smoke_r0_options", bn_momentum=0.9, warmup_epochs=2, min_epochs=1, early_stop_patience=30,
                    learning_rate=3e-4, augment="light"),
            variant(BASELINE, "smoke_baseline_options", **{k: v for step in ABLATION_STEPS.values() for k, v in step.items()})]


def section_groups(section):
    groups = {
        3: {
            "A": {"configs": [R0_SHUFFLED_LABELS], "seeds": [11], "overfit_check": True},
            "B": {"configs": [BASELINE], "seeds": config.SEEDS},
            "C": {"configs": [R0], "seeds": config.SEEDS},
        },
        5: {
            "A": {"configs": stall_fix_configs(), "seeds": config.SEEDS},
            "B": {"configs": stopping_rule_configs(), "seeds": config.SEEDS},
            "C": {"configs": ablation_configs(), "seeds": config.SEEDS[:3]},
            "D": {"configs": optimizer_and_augment_configs(), "seeds": config.SEEDS},
        },
        7: {
            "final": {"configs": [final_config()], "seeds": config.SEEDS, "final_export": True},
        },
    }
    if section not in groups:
        raise KeyError(f"no grid defined for section {section}")
    return groups[section]
