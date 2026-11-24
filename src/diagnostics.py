import logging
from pathlib import Path

import numpy as np

from src import config, experiments, grids, metrics, run_report
from src.preprocessing import preprocess_image_file

LOGGER = logging.getLogger("experiments")
DIAG_STEPS = 300
DIAG_PER_CLASS = 8
MODE_GAP_TOLERANCE = 0.07
SAME_ACCURACY_TOLERANCE = 0.04
SAME_PROBABILITY_TOLERANCE = 1e-3
CHANNEL_MEAN_TOLERANCE = 0.05
CHECK_VARIANT = "overfit_check_config"


def diag_variants():
    return [
        ("baseline", grids.BASELINE, None),
        ("r0", grids.R0, None),
        ("r0_no_batch_norm", grids.variant(grids.R0, "r0_no_batch_norm", batch_norm=False), None),
        ("r0_dropout0_no_augment", grids.variant(grids.R0, "r0_dropout0_no_augment", dropout=0.0, augment="none"), None),
        ("r0_bn_momentum_0.9", grids.R0, 0.9),
        (CHECK_VARIANT, grids.OVERFIT_CHECK, None),
    ]


def diag_batch(data_dir, per_class, seed, size=config.IMAGE_SIZE):
    images, labels, paths = experiments.load_split_cached("train", size, data_dir)
    keep = experiments.stratified_indices(labels, per_class, seed)
    return images[keep], labels[keep], [paths[i] for i in keep]


def scores(probabilities, labels):
    result = metrics.classification_metrics(probabilities, labels)
    return {"accuracy": result["accuracy"], "loss": result["loss"]}


def forward(model, images, training):
    import keras
    return keras.ops.convert_to_numpy(model(images, training=training))


def build_diag_model(cfg, momentum):
    from src import model as model_lib
    built = model_lib.build_model(cfg)
    if momentum is not None:
        model_lib.set_batch_norm_momentum(built, momentum)
    return model_lib.compile_model(built, cfg)


def train_on_batch(model, cfg, images, labels, steps, seed):
    from src.preprocessing import build_dataset
    targets = metrics.one_hot(labels, len(config.CLASS_NAMES))
    dataset = build_dataset(images, targets, len(images), training=True, strength=cfg["augment"], seed=seed)
    history = model.fit(dataset, epochs=steps, verbose=0)
    return {"accuracy": history.history["accuracy"][-1], "loss": history.history["loss"][-1]}


def inference_paths(model, images, labels):
    targets = metrics.one_hot(labels, len(config.CLASS_NAMES))
    evaluated = model.evaluate(images, targets, batch_size=len(images), verbose=0, return_dict=True)
    predicted = model.predict(images, batch_size=len(images), verbose=0)
    manual = forward(model, images, False)
    return {"evaluate": {"accuracy": evaluated["accuracy"], "loss_with_regularization": evaluated["loss"]},
            "predict": scores(predicted, labels), "manual": scores(manual, labels),
            "predict_vs_manual_max_abs_diff": float(np.abs(predicted - manual).max())}


def batch_norm_means(model):
    import keras
    from src import model as model_lib
    return [(keras.ops.convert_to_numpy(layer.moving_mean), keras.ops.convert_to_numpy(layer.moving_variance))
            for layer in model_lib.batch_norm_layers(model)]


def largest_mean_gap(before, after):
    gaps = [np.max(np.abs(old_mean - new_mean) / np.sqrt(new_var + 1e-3)) for (old_mean, _), (new_mean, new_var) in zip(before, after)]
    return float(max(gaps)) if gaps else 0.0


def refreshed_scores(model, images, labels):
    from src import model as model_lib
    before = batch_norm_means(model)
    model_lib.refresh_batch_norm_statistics(model, images)
    return scores(forward(model, images, False), labels), largest_mean_gap(before, batch_norm_means(model))


def measure_trained_model(model, images, labels):
    trained = model.get_weights()
    inference = inference_paths(model, images, labels)
    model.set_weights(trained)
    training_mode = scores(forward(model, images, True), labels)
    model.set_weights(trained)
    refreshed, gap = refreshed_scores(model, images, labels)
    return {"training_mode": training_mode, "inference": inference, "inference_after_bn_refresh": refreshed,
            "max_bn_mean_gap_in_std": gap}


def diagnose_variant(name, cfg, momentum, batch, steps, seed):
    experiments.set_seeds(seed)
    images, labels, _ = batch
    model = build_diag_model(cfg, momentum)
    fit_last = train_on_batch(model, cfg, images, labels, steps, seed)
    row = {"variant": name, "batch_norm": cfg["batch_norm"], "dropout": cfg["dropout"], "augment": cfg["augment"],
           "l2": cfg["l2"], "bn_momentum": momentum if momentum is not None else (0.99 if cfg["batch_norm"] else None),
           "steps": steps, "fit_last_step": fit_last, **measure_trained_model(model, images, labels)}
    LOGGER.info("Diag %s: train mode %.3f, inference %.3f, after BN refresh %.3f", name, row["training_mode"]["accuracy"],
                row["inference"]["manual"]["accuracy"], row["inference_after_bn_refresh"]["accuracy"])
    return row


def dataset_tensors(images, labels, training, seed):
    from src.preprocessing import build_dataset
    dataset = build_dataset(images, labels, len(images), training=training, strength="none", seed=seed)
    return np.concatenate([batch.numpy() for batch, _ in dataset])


def independent_channel_means(path, size):
    import tensorflow as tf
    decoded = tf.io.decode_image(tf.io.read_file(str(path)), channels=3, expand_animations=False)
    resized = tf.image.resize(tf.cast(decoded, tf.float32) / 255.0, (size, size))
    return resized.numpy().mean(axis=(0, 1))


def sorted_rows(array):
    return sorted(row.tobytes() for row in array)


def channel_order_gap(batch, data_dir, size):
    images, _, paths = batch
    reference = np.stack([independent_channel_means(Path(data_dir) / path, size) for path in paths])
    ours = images.mean(axis=(1, 2))
    return float(np.abs(ours - reference).max()), float(np.abs(ours[:, ::-1] - reference).max())


def pipeline_check(batch, data_dir, seed, size=config.IMAGE_SIZE):
    images, labels, paths = batch
    eval_x, train_x = dataset_tensors(images, labels, False, seed), dataset_tensors(images, labels, True, seed)
    inference_x = np.stack([preprocess_image_file(Path(data_dir) / path, size) for path in paths])
    rgb_gap, bgr_gap = channel_order_gap(batch, data_dir, size)
    result = {"dtype": str(eval_x.dtype), "shape": list(eval_x.shape[1:]), "min": float(eval_x.min()), "max": float(eval_x.max()),
              "eval_equals_inference_path": bool(np.array_equal(eval_x, inference_x)),
              "train_equals_eval_ignoring_order": sorted_rows(train_x) == sorted_rows(eval_x),
              "channel_mean_gap_rgb": rgb_gap, "channel_mean_gap_if_bgr": bgr_gap}
    result["passed"] = (result["dtype"] == "float32" and result["shape"] == [size, size, config.CHANNELS]
                        and 0.0 <= result["min"] and result["max"] <= 1.0 and result["eval_equals_inference_path"]
                        and result["train_equals_eval_ignoring_order"] and rgb_gap < CHANNEL_MEAN_TOLERANCE
                        and rgb_gap <= bgr_gap)
    return result


def paths_agree(row):
    inference = row["inference"]
    accuracies = [inference["evaluate"]["accuracy"], inference["predict"]["accuracy"], inference["manual"]["accuracy"]]
    return (max(accuracies) - min(accuracies) <= SAME_ACCURACY_TOLERANCE
            and inference["predict_vs_manual_max_abs_diff"] < SAME_PROBABILITY_TOLERANCE)


def mode_gap(row, key="inference"):
    value = row[key]["manual"]["accuracy"] if key == "inference" else row[key]["accuracy"]
    return row["training_mode"]["accuracy"] - value


def has_gap(row, key="inference"):
    return mode_gap(row, key) > MODE_GAP_TOLERANCE


def verdict(pipeline, rows):
    by_name = {row["variant"]: row for row in rows}
    check, no_bn = by_name[CHECK_VARIANT], by_name["r0_no_batch_norm"]
    if not pipeline["passed"] or not all(paths_agree(row) for row in rows):
        return "c", "The input pipeline or the inference paths disagree, so there is a real preprocessing or eval-path bug."
    if check["training_mode"]["accuracy"] < experiments.OVERFIT_PASS_ACCURACY:
        return "c", "The check configuration cannot fit 32 images even in training mode."
    if has_gap(check) and not has_gap(check, "inference_after_bn_refresh") and not has_gap(no_bn):
        return "a", ("Only batch norm differs between modes: refreshing the moving statistics on the same batch closes "
                     "the gap, and the model without batch norm has no gap.")
    if has_gap(check):
        return "c", "The gap in the check configuration survives a batch norm refresh, so it is not batch norm lag."
    if has_gap(by_name["r0"]) and not has_gap(by_name["r0_dropout0_no_augment"]):
        return "b", "The gap appears only with dropout or augmentation, which the check configuration does not use."
    return "not_reproduced", "No train/inference gap in the check configuration in this run."


def run_batch_diagnostics(ctx, steps=DIAG_STEPS, per_class=DIAG_PER_CLASS, seed=config.SEEDS[0]):
    batch = diag_batch(ctx["data_dir"], per_class, seed)
    pipeline = pipeline_check(batch, ctx["data_dir"], seed)
    rows = [diagnose_variant(name, cfg, momentum, batch, steps, seed) for name, cfg, momentum in diag_variants()]
    cause, reason = verdict(pipeline, rows)
    result = {"steps": steps, "per_class": per_class, "seed": seed, "pipeline": pipeline, "variants": rows,
              "cause": cause, "reason": reason, **experiments.runtime_info()}
    run_report.save_json(result, Path(ctx["dir"]) / "sanity" / "diag_overfit_one_batch.json")
    LOGGER.info("Diag verdict: (%s) %s", cause, reason)
    if not pipeline["passed"] and not ctx["smoke"]:
        raise RuntimeError(f"Training and evaluation inputs differ for the same files: {pipeline}")
    return result
