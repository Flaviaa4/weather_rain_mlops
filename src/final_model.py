import argparse
import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src import config, metrics, run_report
from src.preprocessing import list_labeled_files, load_split

LOGGER = logging.getLogger("experiments")
FINAL_MODEL_NAME = "weather_cnn"
FINAL_DIR = "final_model"
INSTALLED_FILES = ["h5", "onnx", "class_names", "metrics"]
PARITY_TOLERANCE = 1e-4
MIN_PARITY_IMAGES = 20
BOOTSTRAP_DRAWS = 2000
LOCAL_CHECK_TOLERANCE = 1e-4


def output_paths(experiments_dir):
    folder = Path(experiments_dir) / FINAL_DIR
    return {"dir": folder, "h5": folder / f"{FINAL_MODEL_NAME}.h5", "onnx": folder / f"{FINAL_MODEL_NAME}.onnx",
            "class_names": folder / "class_names.json", "metrics": folder / "metrics.json",
            "test_touch": folder / "test_touch.json", "test_predictions": folder / "test_predictions.csv",
            "test_confusion": folder / "test_confusion_matrix.png"}


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def config_rows(cfg, ctx):
    from src import experiments
    rows = experiments.section_rows(ctx)
    rows = rows if rows.empty else rows[rows["config_hash"] == experiments.config_hash(cfg)]
    if rows.empty:
        raise RuntimeError(f"no finished runs of {cfg['name']} in section {ctx['section']}, run the grid first")
    return rows


def seed_summary(rows):
    f1 = rows["val_macro_f1"]
    return {"seeds": int(len(rows)), "val_macro_f1_mean": float(f1.mean()), "val_macro_f1_std": float(f1.std(ddof=1)),
            "val_macro_f1_min": float(f1.min()), "val_macro_f1_max": float(f1.max()), "run_ids": sorted(rows["run_id"])}


def chosen_run(rows):
    from src import experiments
    run_id = experiments.best_run_id(rows)
    return run_id, int(rows.loc[rows["run_id"] == run_id, "seed"].iloc[0])


def load_run_model(ctx, run_id):
    import keras
    return keras.models.load_model(Path(ctx["dir"]) / "runs" / run_id / "model.keras", compile=False)


def export_model(model, paths, run_id):
    from src.onnx_export import export_keras_to_onnx
    paths["dir"].mkdir(parents=True, exist_ok=True)
    model.save(str(paths["h5"]))
    metadata = {"model_name": FINAL_MODEL_NAME, "run_id": run_id, "placeholder": "false"}
    return export_keras_to_onnx(model, paths["onnx"], metadata)


# parity compares CPU with CPU: GPU kernels alone moved Section 5 probabilities by 1.5e-4
def keras_probabilities(model, images):
    import keras
    with keras.device("cpu:0"):
        return np.asarray(model.predict(images, batch_size=config.BATCH_SIZE, verbose=0), dtype=np.float64)


def onnx_probabilities(onnx_path, images):
    from src import prediction
    probabilities = prediction.predict_probabilities(images, onnx_path)
    prediction.drop_session(onnx_path)
    return probabilities


def parity_report(keras_probs, onnx_probs, tolerance=PARITY_TOLERANCE, minimum=MIN_PARITY_IMAGES):
    difference = float(np.abs(np.asarray(keras_probs) - np.asarray(onnx_probs)).max())
    same = float(np.mean(np.argmax(keras_probs, axis=1) == np.argmax(onnx_probs, axis=1)))
    count = int(len(keras_probs))
    return {"n_images": count, "max_abs_diff": difference, "same_class_share": same, "tolerance": tolerance,
            "min_images": minimum, "passed": count >= minimum and difference < tolerance and same == 1.0}


def check_parity(report, split):
    if not report["passed"]:
        raise RuntimeError(f"ONNX parity failed on {split}: {report}")
    return report


def evaluate_arrays(model, onnx_path, images, labels):
    keras_probs, onnx_probs = keras_probabilities(model, images), onnx_probabilities(onnx_path, images)
    return {"metrics": metrics.classification_metrics(onnx_probs, labels), "parity": parity_report(keras_probs, onnx_probs),
            "probabilities": onnx_probs, "labels": labels}


def guard_test_once(paths):
    if paths["test_touch"].exists():
        raise RuntimeError(f"The test set was already evaluated for this final model ({paths['test_touch']}). "
                           "Rule 5 allows one evaluation, so it is not repeated.")


def record_test_touch(paths, run_id, split, n_images):
    record = {"split": split, "run_id": run_id, "n_images": int(n_images), "at": now_iso(),
              "onnx_sha256": hashlib.sha256(paths["onnx"].read_bytes()).hexdigest()}
    return run_report.save_json(record, paths["test_touch"])


def evaluate_test_once(model, paths, cfg, ctx, run_id, split):
    from src import experiments
    guard_test_once(paths)
    images, labels, files = load_split(split, cfg["image_size"], ctx["data_dir"])
    record_test_touch(paths, run_id, split, len(labels))
    LOGGER.info("Test set touched once: split %s, %d images, model %s", split, len(labels), run_id)
    files = experiments.relative_paths(files, ctx["data_dir"])
    return {**evaluate_arrays(model, paths["onnx"], images, labels), "files": files}


def bootstrap_matrices(true, predicted, n_classes):
    draws = true.shape[0]
    codes = np.arange(draws)[:, None] * n_classes ** 2 + true * n_classes + predicted
    return np.bincount(codes.ravel(), minlength=draws * n_classes ** 2).reshape(draws, n_classes, n_classes)


def matrices_f1(matrices):
    matrices = matrices.astype(float)
    hits = np.diagonal(matrices, axis1=1, axis2=2)
    denominator = matrices.sum(axis=1) + matrices.sum(axis=2)
    return np.divide(2 * hits, denominator, out=np.zeros_like(hits), where=denominator > 0)


def interval(values):
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def bootstrap_ci(probabilities, labels, draws=BOOTSTRAP_DRAWS, seed=config.SAMPLE_SEED, class_names=config.CLASS_NAMES):
    labels, predicted = np.asarray(labels), np.asarray(probabilities).argmax(axis=1)
    picks = np.random.default_rng(seed).integers(0, len(labels), size=(draws, len(labels)))
    matrices = bootstrap_matrices(labels[picks], predicted[picks], len(class_names))
    f1 = matrices_f1(matrices)
    accuracy = np.trace(matrices, axis1=1, axis2=2) / len(labels)
    return {"draws": draws, "seed": seed, "accuracy": interval(accuracy), "macro_f1": interval(f1.mean(axis=1)),
            "per_class_f1": {name: interval(f1[:, k]) for k, name in enumerate(class_names)}}


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_fingerprint(split_dir):
    files = sorted(list_labeled_files(split_dir), key=lambda item: Path(item[0]).relative_to(split_dir).as_posix())
    digest = hashlib.sha256()
    for path, label in files:
        digest.update(f"{Path(path).relative_to(split_dir).as_posix()}:{label}:{file_digest(path)}\n".encode("utf-8"))
    return {"n_images": len(files), "sha256": digest.hexdigest()}


def dataset_fingerprint(data_dir):
    return {split: split_fingerprint(Path(data_dir) / split) for split in config.SPLITS if (Path(data_dir) / split).is_dir()}


def metrics_record(cfg, rows, run_id, method, validation, test, ctx):
    from src import experiments
    seed = chosen_run(rows)[1]
    return {"model_name": FINAL_MODEL_NAME, "version": f"{FINAL_MODEL_NAME}_{run_id}", "created_at": now_iso(),
            "run_id": run_id, "seed": seed, "config": cfg, "config_hash": experiments.config_hash(cfg),
            "selection": seed_summary(rows), "dataset": dataset_fingerprint(ctx["data_dir"]), "export_method": method,
            "runtime": experiments.runtime_info(), "class_names": list(config.CLASS_NAMES),
            "parity": {"validation": validation["parity"], "test": test["parity"]},
            "validation": {**validation["metrics"], "split": "valid"}, "test": {**test["metrics"], "split": test["split"]},
            "test_bootstrap_95ci": bootstrap_ci(test["probabilities"], test["labels"])}


def save_test_outputs(test, paths, run_id):
    run_report.save_predictions(test["files"], test["labels"], test["probabilities"], paths["test_predictions"])
    title = f"{run_id}: {test['split']}, macro F1 {test['metrics']['macro_f1']:.3f}"
    return run_report.plot_confusion_matrix(test["metrics"]["confusion_matrix"], paths["test_confusion"], title)


def log_final(record):
    validation, test, ci = record["validation"], record["test"], record["test_bootstrap_95ci"]
    LOGGER.info("Final model %s (seed %d): validation macro F1 %.4f, %s macro F1 %.4f (95%% CI %.3f to %.3f), "
                "accuracy %.4f", record["run_id"], record["seed"], validation["macro_f1"], test["split"],
                test["macro_f1"], ci["macro_f1"][0], ci["macro_f1"][1], test["accuracy"])
    LOGGER.info("ONNX parity: validation %s, %s %s", record["parity"]["validation"], test["split"], record["parity"]["test"])


def validated_export(cfg, ctx, paths):
    rows = config_rows(cfg, ctx)
    run_id = chosen_run(rows)[0]
    model = load_run_model(ctx, run_id)
    method = export_model(model, paths, run_id)
    images, labels, _ = load_split("valid", cfg["image_size"], ctx["data_dir"])
    validation = evaluate_arrays(model, paths["onnx"], images, labels)
    check_parity(validation["parity"], "valid")
    return rows, run_id, model, method, validation


def run_final(cfg, ctx, test_split="test"):
    from src import experiments
    cfg, paths = experiments.effective_config(cfg, ctx), output_paths(ctx["dir"])
    guard_test_once(paths)
    rows, run_id, model, method, validation = validated_export(cfg, ctx, paths)
    run_report.save_json(list(config.CLASS_NAMES), paths["class_names"])
    test = {**evaluate_test_once(model, paths, cfg, ctx, run_id, test_split), "split": test_split}
    record = metrics_record(cfg, rows, run_id, method, validation, test, ctx)
    run_report.save_json(record, paths["metrics"])
    save_test_outputs(test, paths, run_id)
    log_final(record)
    check_parity(test["parity"], test_split)
    return record


def check_final_files(paths):
    missing = [paths[key].name for key in INSTALLED_FILES + ["test_touch"] if not paths[key].exists()]
    if missing:
        raise FileNotFoundError(f"{paths['dir']} is missing {missing}")
    touched = json.loads(paths["test_touch"].read_text(encoding="utf-8"))["onnx_sha256"]
    if touched != file_digest(paths["onnx"]):
        raise ValueError("the ONNX file is not the one the test set was evaluated with")
    return paths


def check_record(record):
    problems = [key for key in ["validation", "test"] if key not in record]
    problems += [] if record.get("model_name") == FINAL_MODEL_NAME else ["model_name"]
    problems += [f"parity_{split}" for split in ["validation", "test"] if not record.get("parity", {}).get(split, {}).get("passed")]
    if problems:
        raise ValueError(f"metrics.json is not a valid final model record: {problems}")
    return record


def check_onnx_locally(paths, record, data_dir):
    from src import prediction
    images, labels, _ = load_split("valid", record["config"]["image_size"], data_dir)
    local = prediction.evaluate_onnx_model(images, labels, paths["onnx"])
    prediction.drop_session(paths["onnx"])
    gap = abs(local["macro_f1"] - record["validation"]["macro_f1"])
    if gap > LOCAL_CHECK_TOLERANCE:
        raise ValueError(f"local ONNX validation macro F1 {local['macro_f1']:.4f} differs from metrics.json by {gap:.4f}")
    return local["macro_f1"]


def install_final_model(experiments_dir, models_dir=config.MODELS_DIR, data_dir=config.DATA_DIR):
    paths = check_final_files(output_paths(experiments_dir))
    record = check_record(json.loads(paths["metrics"].read_text(encoding="utf-8")))
    local_f1 = check_onnx_locally(paths, record, data_dir)
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    installed = [shutil.copy2(paths[key], Path(models_dir) / paths[key].name) for key in INSTALLED_FILES]
    return {"installed": [str(path) for path in installed], "local_validation_macro_f1": local_f1,
            "run_id": record["run_id"], "test_macro_f1": record["test"]["macro_f1"]}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", required=True, help="experiments folder that holds final_model/ from Kaggle")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(install_final_model(parse_args().install), indent=2))
