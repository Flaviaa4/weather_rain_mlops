import logging
import os
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src import config, database, prediction
from src.preprocessing import list_labeled_files, load_split, preprocess_image_file
from src.run_report import save_json
from src.uploads import upload_file_path

LOGGER = logging.getLogger("retrain")
JOB_LOCK = threading.Lock()
STATUS_LOCK = threading.Lock()
STATUS = {"run_id": None, "stage": None, "started_at": None, "message": None}
BASE_SPLITS = {"train": "training", "valid": "validation"}


class RetrainRunning(Exception):
    pass


class BaseDataMissing(RuntimeError):
    pass


def split_class_counts(split_dir):
    labels = [label for _, label in list_labeled_files(split_dir)]
    return {name: labels.count(index) for index, name in enumerate(config.CLASS_NAMES)}


def base_data_status(data_dir=None):
    data_dir = Path(data_dir or config.DATA_DIR)
    return {split: {"path": str(data_dir / split), "per_class": split_class_counts(data_dir / split)} for split in BASE_SPLITS}


def split_problem(split, info):
    total, empty = sum(info["per_class"].values()), [name for name, count in info["per_class"].items() if count == 0]
    if total == 0:
        return f"base {BASE_SPLITS[split]} data not found at {info['path']}, {total} images"
    return f"base {BASE_SPLITS[split]} data at {info['path']} has no images for {', '.join(empty)}" if empty else None


def base_data_report(data_dir=None):
    status = base_data_status(data_dir)
    problems = [problem for split, info in status.items() if (problem := split_problem(split, info))]
    splits = {split: {"path": info["path"], "n_images": sum(info["per_class"].values()), "per_class": info["per_class"]}
              for split, info in status.items()}
    return {"ok": not problems, "problems": problems, **splits}


def check_base_data(data_dir=None):
    report = base_data_report(data_dir)
    if not report["ok"]:
        raise BaseDataMissing("; ".join(report["problems"]) + ". Retraining needs data/train and data/valid, "
                              "one folder per class, in the data folder mounted into the trainer.")
    return report


def model_paths():
    return {"h5": config.H5_PATH, "onnx": config.ONNX_PATH, "metrics": config.METRICS_PATH}


def set_stage(run_id, stage, message=None, db_path=None):
    with STATUS_LOCK:
        STATUS.update({"run_id": run_id, "stage": stage, "message": message})
    database.update_run(run_id, db_path, stage=stage)
    LOGGER.info("Retrain run %s: %s", run_id, stage)


def get_status(db_path=None):
    with STATUS_LOCK:
        current = dict(STATUS)
    runs = database.list_runs(1, db_path)
    return {"running": JOB_LOCK.locked(), "current": current if JOB_LOCK.locked() else None,
            "last_run": runs[0] if runs else None, "unused_uploads": database.count_unused_uploads(db_path),
            "threshold": config.RETRAIN_UPLOAD_THRESHOLD, "auto_retrain": config.RETRAIN_AUTO}


def should_retrain(unused_count, threshold=None):
    return unused_count >= (config.RETRAIN_UPLOAD_THRESHOLD if threshold is None else threshold)


def should_promote(candidate_f1, current_f1, tolerance=None):
    tolerance = config.PROMOTION_TOLERANCE if tolerance is None else tolerance
    return current_f1 is None or candidate_f1 >= current_f1 - tolerance


def retrain_config():
    from src import grids
    return {**grids.R0, "name": "retrain", "learning_rate": config.RETRAIN_LEARNING_RATE,
            "max_epochs": config.RETRAIN_MAX_EPOCHS, "early_stop_patience": config.RETRAIN_PATIENCE}


def load_one_upload(row, size, uploads_dir):
    try:
        return preprocess_image_file(upload_file_path(row, uploads_dir), size)
    except Exception:
        LOGGER.warning("Upload %s could not be read, marking it rejected", row["id"])
        return None


def load_upload_arrays(rows, size=config.IMAGE_SIZE, uploads_dir=None):
    arrays, labels, ids, rejected = [], [], [], []
    for row in rows:
        array = load_one_upload(row, size, uploads_dir)
        if array is None:
            rejected.append(row["id"])
            continue
        arrays.append(array), labels.append(config.CLASS_NAMES.index(row["label"])), ids.append(row["id"])
    images = np.stack(arrays) if arrays else np.zeros((0, size, size, config.CHANNELS), np.float32)
    return {"images": images, "labels": np.array(labels, dtype=np.int32), "ids": ids, "rejected": rejected}


def merge_training_data(base, uploads):
    images = np.concatenate([base[0], uploads["images"]]) if len(uploads["ids"]) else base[0]
    labels = np.concatenate([base[1], uploads["labels"]]) if len(uploads["ids"]) else base[1]
    return images, labels


def prepare_retraining_data(db_path=None, data_dir=None, uploads_dir=None, size=config.IMAGE_SIZE):
    data_dir = data_dir or config.DATA_DIR
    check_base_data(data_dir)
    base = load_split("train", size, data_dir)
    uploads = load_upload_arrays(database.all_usable_uploads(db_path), size, uploads_dir)
    images, labels = merge_training_data(base, uploads)
    valid = load_split("valid", size, data_dir)
    return {"train_images": images, "train_labels": labels, "val_images": valid[0], "val_labels": valid[1],
            "upload_ids": uploads["ids"], "rejected_ids": uploads["rejected"], "n_base": len(base[1])}


def ensure_base_model():
    if config.IS_PLACEHOLDER:
        from src.placeholder import ensure_placeholder_files
        ensure_placeholder_files(with_keras=True)
    if not config.H5_PATH.exists():
        raise FileNotFoundError(f"{config.H5_PATH} is missing. Retraining continues from our saved model, so it is required.")
    return config.H5_PATH


def fine_tune_model(data, cfg, history_path, h5_path=None):
    import keras
    from src import experiments
    from src import model as model_lib
    keras.utils.set_random_seed(config.RETRAIN_SEED)
    model = model_lib.compile_model(keras.models.load_model(str(h5_path or config.H5_PATH), compile=False), cfg)
    train, valid = experiments.build_datasets(cfg, config.RETRAIN_SEED, data)
    class_weight = model_lib.compute_class_weights(data["train_labels"]) if cfg["class_weights"] else None
    callbacks = model_lib.get_callbacks(cfg, history_path)
    model_lib.train_model(model, train, valid, cfg, callbacks, class_weight)
    return model


def candidate_paths(run_id):
    folder = config.CANDIDATE_DIR / f"run_{run_id}"
    folder.mkdir(parents=True, exist_ok=True)
    return {"dir": folder, "h5": folder / "candidate.h5", "onnx": folder / "candidate.onnx",
            "history": folder / "history.csv", "metrics": folder / "metrics.json"}


def export_candidate(model, paths, run_id):
    from src.onnx_export import export_keras_to_onnx
    model.save(str(paths["h5"]))
    metadata = {"model_name": config.MODEL_NAME, "retrain_run": run_id, "placeholder": str(config.IS_PLACEHOLDER).lower()}
    return export_keras_to_onnx(model, paths["onnx"], metadata)


def evaluate_model_file(onnx_path, data):
    if not Path(onnx_path).exists():
        return None
    result = prediction.evaluate_onnx_model(data["val_images"], data["val_labels"], onnx_path)
    prediction.drop_session(onnx_path)
    return result


def archive_current_model(run_id, archive_dir=None):
    archive_dir = archive_dir or config.ARCHIVE_DIR
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = Path(archive_dir) / f"{stamp}_before_run{run_id}"
    target.mkdir(parents=True, exist_ok=True)
    for path in model_paths().values():
        if Path(path).exists():
            shutil.copy2(path, target / Path(path).name)
    return target


def atomic_replace(source, target, attempts=50):
    for attempt in range(attempts):
        try:
            return os.replace(source, target)
        except PermissionError:
            # Windows refuses to replace a file another thread is reading, Linux never gets here
            if attempt == attempts - 1:
                raise
            time.sleep(0.02)


def replace_file(source, target):
    partial = Path(target).with_name(Path(target).name + ".incoming")
    shutil.copy2(source, partial)
    atomic_replace(partial, target)
    return Path(target)


def promotion_record(run_id, candidate, current, data):
    return {"model_name": config.MODEL_NAME, "placeholder_base": config.IS_PLACEHOLDER, "split": "valid",
            "retrain_run": run_id, "promoted_at": database.now_iso(), "n_train_images": int(len(data["train_labels"])),
            "n_uploads": len(data["upload_ids"]), "previous_macro_f1": current["macro_f1"] if current else None,
            "validation": candidate}


def promote_model(paths, record):
    replace_file(paths["h5"], config.H5_PATH)
    save_json(record, paths["metrics"])
    replace_file(paths["metrics"], config.METRICS_PATH)
    replace_file(paths["onnx"], config.ONNX_PATH)
    return config.ONNX_PATH


def per_class_f1(result):
    return {name: round(values["f1"], 4) for name, values in result["per_class"].items()} if result else None


def comparison_details(current, candidate, method, archive):
    return {"export_method": method, "archive": str(archive) if archive else None,
            "current": {"macro_f1": current["macro_f1"] if current else None, "accuracy": current["accuracy"] if current else None,
                        "per_class_f1": per_class_f1(current)},
            "candidate": {"macro_f1": candidate["macro_f1"], "accuracy": candidate["accuracy"],
                          "per_class_f1": per_class_f1(candidate)}}


def decide_and_apply(run_id, paths, current, candidate, data):
    promoted = should_promote(candidate["macro_f1"], current["macro_f1"] if current else None)
    archive = None
    if promoted:
        set_stage(run_id, "promoting")
        archive = archive_current_model(run_id)
        promote_model(paths, promotion_record(run_id, candidate, current, data))
    return promoted, archive


def finish_success(run_id, started, data, current, candidate, promoted, details):
    verdict = "promoted, the API reloads it on the next request" if promoted else "rejected, the current model stays"
    database.update_run(run_id, status="completed", stage="done", finished_at=database.now_iso(),
                        duration_seconds=round(time.time() - started, 1), n_uploads=len(data["upload_ids"]),
                        n_train_images=int(len(data["train_labels"])), promoted=int(promoted),
                        current_macro_f1=current["macro_f1"] if current else None,
                        candidate_macro_f1=candidate["macro_f1"], details=details,
                        message=f"Candidate macro F1 {candidate['macro_f1']:.4f}: {verdict}.",
                        model_version=prediction.file_digest(config.ONNX_PATH))


def mark_uploads_after_run(run_id, data):
    new_ids = {row["id"] for row in database.unused_uploads()}
    database.mark_uploads([i for i in data["upload_ids"] if i in new_ids], run_id, "used")
    database.mark_uploads(data["rejected_ids"], run_id, "rejected")


def run_retraining(run_id):
    started = time.time()
    set_stage(run_id, "loading data")
    data = prepare_retraining_data()
    ensure_base_model()
    set_stage(run_id, "evaluating current model")
    current = evaluate_model_file(config.ONNX_PATH, data)
    set_stage(run_id, "training")
    paths = candidate_paths(run_id)
    model = fine_tune_model(data, retrain_config(), paths["history"])
    set_stage(run_id, "exporting to ONNX")
    method = export_candidate(model, paths, run_id)
    set_stage(run_id, "evaluating candidate")
    candidate = evaluate_model_file(paths["onnx"], data)
    promoted, archive = decide_and_apply(run_id, paths, current, candidate, data)
    mark_uploads_after_run(run_id, data)
    finish_success(run_id, started, data, current, candidate, promoted, comparison_details(current, candidate, method, archive))
    shutil.rmtree(paths["dir"], ignore_errors=True)


def finish_failure(run_id, error):
    LOGGER.exception("Retrain run %s failed", run_id)
    database.update_run(run_id, status="failed", stage="failed", finished_at=database.now_iso(),
                        message=f"{type(error).__name__}: {error}"[:500])


def guarded_run(run_id, runner):
    try:
        runner(run_id)
    except Exception as error:
        finish_failure(run_id, error)
    finally:
        with STATUS_LOCK:
            STATUS.update({"run_id": None, "stage": None, "started_at": None, "message": None})
        JOB_LOCK.release()


def start_retraining(trigger="manual", runner=None, wait=False):
    if not JOB_LOCK.acquire(blocking=False):
        raise RetrainRunning("A retraining job is already running.")
    try:
        run_id = database.start_run(trigger)
    except Exception:
        JOB_LOCK.release()
        raise
    with STATUS_LOCK:
        STATUS.update({"run_id": run_id, "stage": "queued", "started_at": database.now_iso()})
    thread = threading.Thread(target=guarded_run, args=(run_id, runner or run_retraining), daemon=True)
    thread.start()
    if wait:
        thread.join()
    return run_id


def maybe_auto_retrain(runner=None):
    if not config.RETRAIN_AUTO or JOB_LOCK.locked() or not should_retrain(database.count_unused_uploads()):
        return None
    base_data = base_data_report()
    if not base_data["ok"]:
        LOGGER.warning("Upload threshold reached but no retrain started: %s", base_data["problems"])
        return None
    try:
        return start_retraining("auto", runner)
    except RetrainRunning:
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    database.init_db()
    print(database.get_run(start_retraining("script", wait=True)))
