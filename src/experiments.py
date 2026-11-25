import argparse
import hashlib
import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import config, grids, metrics, run_report
from src.preprocessing import list_labeled_files, load_split

LOGGER = logging.getLogger("experiments")
MODEL_FILE = "model.keras"
RUN_FILES = ["config.json", "history.csv", "curves.png", "val_metrics.json", "confusion_matrix.png",
             "val_predictions.csv"]
KEEP_MODEL_CONFIGS = 3
SMOKE_PER_CLASS = {"train": 8, "valid": 4}
SMOKE_EPOCHS = 2
OVERFIT_STEPS = 300
OVERFIT_PASS_ACCURACY = 0.97
TYPICAL_EPOCHS = 60
RUN_OVERHEAD_SECONDS = 20
COLAB_UNITS_PER_HOUR = float(os.environ.get("COLAB_UNITS_PER_HOUR", "1.96"))
UNIT_BUDGET = 0.6 * float(os.environ.get("COLAB_UNITS_REMAINING", "74"))
DATA_CACHE = {}
DIAG_SECTIONS = {3}


def config_hash(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()[:8]


def make_run_id(section, cfg, seed):
    return f"s{section}_{cfg['name']}_seed{seed}"


def load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))["config"]


def results_path(experiments_dir):
    return Path(experiments_dir) / "results.csv"


def read_results(experiments_dir):
    path = results_path(experiments_dir)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def already_done(run_id, cfg, experiments_dir):
    results = read_results(experiments_dir)
    if results.empty or run_id not in set(results["run_id"]):
        return False
    stored = results.loc[results["run_id"] == run_id, "config_hash"].iloc[-1]
    if stored != config_hash(cfg):
        raise ValueError(f"{run_id} is already in results.csv with a different config ({stored})")
    return True


def log_result_row(row, experiments_dir):
    path = results_path(experiments_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.concat([read_results(experiments_dir), pd.DataFrame([row])], ignore_index=True)
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)
    return path


def setup_logging(experiments_dir, name):
    log_dir = Path(experiments_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in [logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8"), logging.StreamHandler()]:
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)
    return LOGGER


def set_seeds(seed):
    import keras
    keras.utils.set_random_seed(seed)


def relative_paths(paths, data_dir):
    return [Path(path).relative_to(data_dir).as_posix() for path in paths]


def load_split_cached(split, size, data_dir):
    key = (split, size, str(data_dir))
    if key not in DATA_CACHE:
        images, labels, paths = load_split(split, size, data_dir)
        DATA_CACHE[key] = (images, labels, relative_paths(paths, data_dir))
    return DATA_CACHE[key]


def take_per_class(split_data, per_class):
    images, labels, paths = split_data
    keep = np.concatenate([np.flatnonzero(labels == k)[:per_class] for k in np.unique(labels)])
    return images[keep], labels[keep], [paths[i] for i in keep]


def check_supported(cfg):
    if cfg["oversample"] or cfg["train_fraction"] != 1.0:
        raise NotImplementedError("oversampling and train fractions are added in Sections 5 and 6")


def prepare_data(cfg, seed, ctx):
    check_supported(cfg)
    train = load_split_cached("train", cfg["image_size"], ctx["data_dir"])
    valid = load_split_cached("valid", cfg["image_size"], ctx["data_dir"])
    if ctx["smoke"]:
        train, valid = take_per_class(train, SMOKE_PER_CLASS["train"]), take_per_class(valid, SMOKE_PER_CLASS["valid"])
    train_labels = np.random.default_rng(seed).permutation(train[1]) if cfg["shuffle_labels"] else train[1]
    return {"train_images": train[0], "train_labels": train_labels, "val_images": valid[0],
            "val_labels": valid[1], "val_paths": valid[2]}


def effective_config(cfg, ctx):
    return {**cfg, "max_epochs": SMOKE_EPOCHS} if ctx["smoke"] else cfg


def backup_dir_for(run_dir, ctx):
    return run_dir / "backup" if ctx["backup"] else None


def prepare_run_dir(run_dir, ctx):
    resuming = ctx["backup"] and (run_dir / "backup").exists()
    if run_dir.exists() and not resuming:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return resuming


def build_datasets(cfg, seed, data):
    from src.preprocessing import build_dataset
    n_classes = len(config.CLASS_NAMES)
    train = build_dataset(data["train_images"], metrics.one_hot(data["train_labels"], n_classes),
                          cfg["batch_size"], training=True, strength=cfg["augment"], seed=seed)
    valid = build_dataset(data["val_images"], metrics.one_hot(data["val_labels"], n_classes), cfg["batch_size"])
    return train, valid


def fit_run(cfg, seed, data, run_dir, ctx):
    from src import model as model_lib
    model = model_lib.compile_model(model_lib.build_model(cfg), cfg)
    train, valid = build_datasets(cfg, seed, data)
    class_weight = model_lib.compute_class_weights(data["train_labels"]) if cfg["class_weights"] else None
    callbacks = model_lib.get_callbacks(cfg, run_dir / "history.csv", backup_dir_for(run_dir, ctx))
    model_lib.train_model(model, train, valid, cfg, callbacks, class_weight)
    return model


def collect_predictions(model, cfg, data):
    from src import model as model_lib
    val_metrics, probabilities = model_lib.evaluate_model(model, data["val_images"], data["val_labels"], cfg["batch_size"])
    targets = metrics.one_hot(data["train_labels"], len(config.CLASS_NAMES))
    train_eval = model.evaluate(data["train_images"], targets, batch_size=cfg["batch_size"], verbose=0, return_dict=True)
    return val_metrics, probabilities, train_eval


def save_run_artifacts(run_dir, run_id, record, history, val_metrics, probabilities, data, model):
    run_report.save_json(record, run_dir / "config.json")
    run_report.plot_curves(history, run_dir / "curves.png", run_id, config.config_setting(record["config"], "min_epochs"))
    run_report.save_json(val_metrics, run_dir / "val_metrics.json")
    run_report.plot_confusion_matrix(val_metrics["confusion_matrix"], run_dir / "confusion_matrix.png",
                                     f"{run_id}: validation, macro F1 {val_metrics['macro_f1']:.3f}")
    run_report.save_predictions(data["val_paths"], data["val_labels"], probabilities, run_dir / "val_predictions.csv")
    from src import model as model_lib
    model_lib.save_without_optimizer(model, run_dir / MODEL_FILE)
    return run_dir


def history_fields(history, cfg):
    best = run_report.best_epoch_index(history, config.config_setting(cfg, "min_epochs"))
    row = history.iloc[best]
    return {"epochs_run": len(history), "best_epoch": best + 1, "train_loss": row["loss"],
            "train_accuracy": row["accuracy"], "history_val_loss": row["val_loss"],
            "history_val_accuracy": row["val_accuracy"], "stopped_early": len(history) < cfg["max_epochs"],
            "epoch_seconds_median": float(history["epoch_seconds"].median()),
            "final_learning_rate": float(history["learning_rate"].iloc[-1])}


def runtime_info():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    gpu = tf.config.experimental.get_device_details(gpus[0]).get("device_name", "GPU") if gpus else "none"
    return {"tf_version": tf.__version__, "gpu": gpu}


def build_result_row(run_id, ctx, cfg, seed, history, val_metrics, train_eval, extra):
    row = {"run_id": run_id, "section": ctx["section"], "group": ctx["group"], "seed": seed,
           "config_hash": config_hash(cfg), **{f"cfg_{key}": value for key, value in cfg.items()}}
    row.update(history_fields(history, cfg))
    row.update({"train_eval_loss": train_eval["loss"], "train_eval_accuracy": train_eval["accuracy"]})
    row.update(metrics.flat_metric_fields(val_metrics))
    row.update(extra)
    return row


def run_one(cfg, seed, ctx):
    cfg = effective_config(cfg, ctx)
    run_id = make_run_id(ctx["section"], cfg, seed)
    run_dir = Path(ctx["dir"]) / "runs" / run_id
    resumed = prepare_run_dir(run_dir, ctx)
    started = time.perf_counter()
    set_seeds(seed)
    data = prepare_data(cfg, seed, ctx)
    model = fit_run(cfg, seed, data, run_dir, ctx)
    val_metrics, probabilities, train_eval = collect_predictions(model, cfg, data)
    history = run_report.read_history(run_dir / "history.csv")
    record = {"run_id": run_id, "seed": seed, "config_hash": config_hash(cfg), "config": cfg, **runtime_info()}
    save_run_artifacts(run_dir, run_id, record, history, val_metrics, probabilities, data, model)
    extra = {"params": int(model.count_params()), "minutes": (time.perf_counter() - started) / 60, "resumed": resumed,
             "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), **runtime_info()}
    row = build_result_row(run_id, ctx, cfg, seed, history, val_metrics, train_eval, extra)
    log_result_row(row, ctx["dir"])
    return row


def pending_runs(configs, seeds, ctx):
    planned = [(effective_config(cfg, ctx), seed) for cfg in configs for seed in seeds]
    return [(cfg, seed) for cfg, seed in planned
            if not already_done(make_run_id(ctx["section"], cfg, seed), cfg, ctx["dir"])]


def estimate_total_time(n_runs, minutes_per_run):
    return n_runs * minutes_per_run


def initial_minutes_per_run(ctx):
    path = Path(ctx["dir"]) / "timing.json"
    return json.loads(path.read_text())["typical_minutes_per_run"] if path.exists() else None


def log_progress(index, total, row, eta_minutes):
    LOGGER.info("[%d/%d] %s: val macro F1 %.3f, rainy F1 %.3f, %d epochs (best %d), %.1f min. ETA %.0f min",
                index, total, row["run_id"], row["val_macro_f1"], row["val_f1_rainy"], row["epochs_run"],
                row["best_epoch"], row["minutes"], eta_minutes)


def log_plan(planned, todo, ctx):
    guess = initial_minutes_per_run(ctx)
    LOGGER.info("%d runs planned, %d already done, %d to run. Initial ETA: %s", planned, planned - len(todo), len(todo),
                f"{estimate_total_time(len(todo), guess):.0f} min" if guess else "after the first run")


def section_deadline(section):
    minutes = float(os.environ.get("RUN_BUDGET_MINUTES", grids.SECTION_BUDGET_MINUTES.get(section, 0)))
    return time.monotonic() + minutes * 60 if minutes > 0 else None


def past_deadline(ctx):
    return ctx.get("deadline") is not None and time.monotonic() > ctx["deadline"]


def log_budget_stop(left):
    LOGGER.info("Time budget reached: %d runs not started (%s). Attach this version's output and run the same GROUP "
                "again to finish them.", len(left), ", ".join(f"{cfg['name']} seed {seed}" for cfg, seed in left))


def best_run_id(rows):
    return rows.sort_values(["val_macro_f1", "val_loss"], ascending=[False, True])["run_id"].iloc[0]


def section_rows(ctx):
    results = read_results(ctx["dir"])
    return results if results.empty else results[results["section"] == ctx["section"]]


def top_config_hashes(rows, top=KEEP_MODEL_CONFIGS):
    stats = rows.groupby("config_hash")["val_macro_f1"].agg(["mean", "std"]).fillna(0.0)
    return list(stats.sort_values(["mean", "std"], ascending=[False, True]).index[:top])


def best_runs_of(rows, hashes):
    return {best_run_id(rows[rows["config_hash"] == value]) for value in hashes}


def delete_models(run_ids, ctx):
    removed = []
    for run_id in run_ids:
        path = Path(ctx["dir"]) / "runs" / run_id / MODEL_FILE
        if path.exists():
            path.unlink()
            removed.append(run_id)
    return removed


def prune_seed_models(cfg, ctx):
    rows = section_rows(ctx)
    rows = rows if rows.empty else rows[rows["config_hash"] == config_hash(cfg)]
    if rows.empty:
        return []
    keep = best_run_id(rows)
    return delete_models([run_id for run_id in rows["run_id"] if run_id != keep], ctx)


def prune_models(ctx, top=KEEP_MODEL_CONFIGS):
    rows = section_rows(ctx)
    if rows.empty:
        return []
    keep = best_runs_of(rows, top_config_hashes(rows, top))
    removed = delete_models([run_id for run_id in rows["run_id"] if run_id not in keep], ctx)
    LOGGER.info("Models kept for the top %d configs by mean val macro F1: %s (%d removed)", top, ", ".join(sorted(keep)),
                len(removed))
    return removed


def run_grid(configs, seeds, ctx):
    todo = pending_runs(configs, seeds, ctx)
    log_plan(len(configs) * len(seeds), todo, ctx)
    durations = []
    for index, (cfg, seed) in enumerate(todo, start=1):
        if past_deadline(ctx):
            log_budget_stop(todo[index - 1:])
            break
        row = run_one(cfg, seed, ctx)
        durations.append(row["minutes"])
        prune_seed_models(cfg, ctx)
        log_progress(index, len(todo), row, estimate_total_time(len(todo) - index, float(np.mean(durations))))
    prune_models(ctx)
    return len(durations)


def stratified_indices(labels, per_class, seed):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(np.flatnonzero(labels == k), per_class, replace=False) for k in np.unique(labels)])


def stratified_batch(images, labels, per_class, seed):
    keep = stratified_indices(labels, per_class, seed)
    return images[keep], labels[keep]


def overfit_result(history, raw, refreshed, steps):
    accuracy = history.history["accuracy"]
    first = next((i + 1 for i, value in enumerate(accuracy) if value >= 1.0), None)
    return {"steps": steps, "first_step_at_100pct_train_accuracy": first, "final_train_loss": history.history["loss"][-1],
            "final_train_accuracy": accuracy[-1], "inference_mode_accuracy_stale_bn": raw["accuracy"],
            "inference_mode_loss_stale_bn": raw["loss"], "inference_mode_accuracy": refreshed["accuracy"],
            "inference_mode_loss": refreshed["loss"], "bn_statistics": "refreshed on the batch before the inference test",
            "passed": accuracy[-1] >= OVERFIT_PASS_ACCURACY and refreshed["accuracy"] >= OVERFIT_PASS_ACCURACY}


def run_overfit_check(ctx, seed=config.SEEDS[0], steps=OVERFIT_STEPS):
    from src import model as model_lib
    path = Path(ctx["dir"]) / "sanity" / "overfit_one_batch.json"
    if path.exists() and json.loads(path.read_text())["passed"]:
        return LOGGER.info("Overfit check already passed, skipping")
    set_seeds(seed)
    images, labels = stratified_batch(*load_split_cached("train", grids.OVERFIT_CHECK["image_size"], ctx["data_dir"])[:2], 8, seed)
    targets = metrics.one_hot(labels, len(config.CLASS_NAMES))
    model = model_lib.compile_model(model_lib.build_model(grids.OVERFIT_CHECK), grids.OVERFIT_CHECK)
    history = model.fit(images, targets, batch_size=len(images), epochs=steps, shuffle=False, verbose=0)
    raw = model.evaluate(images, targets, verbose=0, return_dict=True)
    model_lib.refresh_batch_norm_statistics(model, images)
    result = overfit_result(history, raw, model.evaluate(images, targets, verbose=0, return_dict=True), steps)
    run_report.save_json({**result, "config": grids.OVERFIT_CHECK, "seed": seed}, path)
    LOGGER.info("Overfit one batch: %s", result)
    if not result["passed"] and not ctx["smoke"]:
        raise RuntimeError("overfit-one-batch check failed: the pipeline cannot fit 32 images")


def time_config(cfg, ctx, epochs=3):
    timing_ctx = {**ctx, "smoke": False, "backup": False}
    run_dir = Path(ctx["dir"]) / "timing_runs" / cfg["name"]
    prepare_run_dir(run_dir, timing_ctx)
    set_seeds(config.SEEDS[0])
    fit_run({**cfg, "max_epochs": epochs}, config.SEEDS[0], prepare_data(cfg, config.SEEDS[0], timing_ctx), run_dir, timing_ctx)
    seconds = run_report.read_history(run_dir / "history.csv")["epoch_seconds"].iloc[1:].median()
    shutil.rmtree(run_dir)
    return float(seconds)


def group_minutes(spec, epoch_seconds, epochs):
    runs = [(cfg["architecture"], seed) for cfg in spec["configs"] for seed in spec["seeds"]]
    return sum((epochs * epoch_seconds[arch] + RUN_OVERHEAD_SECONDS) / 60 for arch, _ in runs)


def estimate_section(section, epoch_seconds):
    rows = []
    for name, spec in grids.section_groups(section).items():
        typical, worst = group_minutes(spec, epoch_seconds, TYPICAL_EPOCHS), group_minutes(spec, epoch_seconds, config.MAX_EPOCHS)
        rows.append({"group": name, "runs": len(spec["configs"]) * len(spec["seeds"]), "typical_minutes": typical,
                     "worst_minutes": worst, "worst_units": worst / 60 * COLAB_UNITS_PER_HOUR})
    return rows


def run_timing(ctx):
    epoch_seconds = {"cnn": time_config(grids.R0, ctx), "baseline": time_config(grids.BASELINE, ctx)}
    rows = estimate_section(ctx["section"], epoch_seconds)
    typical = (TYPICAL_EPOCHS * epoch_seconds["cnn"] + RUN_OVERHEAD_SECONDS) / 60
    summary = {"epoch_seconds": epoch_seconds, "typical_epochs": TYPICAL_EPOCHS, "typical_minutes_per_run": typical,
               "units_per_hour": COLAB_UNITS_PER_HOUR, "groups": rows, **runtime_info()}
    run_report.save_json(summary, Path(ctx["dir"]) / "timing.json")
    LOGGER.info("Epoch seconds: %s", epoch_seconds)
    for row in rows:
        LOGGER.info("Group %s: %d runs, typical %.0f min, worst case %.0f min = %.1f compute units", row["group"],
                    row["runs"], row["typical_minutes"], row["worst_minutes"], row["worst_units"])
    worst = sum(row["worst_units"] for row in rows)
    LOGGER.info("Section %s worst case total: %.1f compute units (budget without asking: %.0f)", ctx["section"], worst, UNIT_BUDGET)
    if worst > UNIT_BUDGET:
        raise SystemExit("Worst-case estimate is above 60% of the remaining compute units. Stop and ask before running.")
    return summary


def check_run_artifacts(run_dir, with_model=True):
    required = RUN_FILES + ([MODEL_FILE] if with_model else [])
    missing = [name for name in required if not (Path(run_dir) / name).exists()]
    if missing:
        raise RuntimeError(f"{run_dir} is missing {missing}")
    return True


def kept_models(experiments_dir):
    return sorted(Path(experiments_dir).glob(f"runs/*/{MODEL_FILE}"))


def check_kept_models(ctx, n_configs):
    expected = min(n_configs, KEEP_MODEL_CONFIGS)
    found = len(kept_models(ctx["dir"]))
    if found != expected:
        raise RuntimeError(f"model pruning check failed: {found} model files kept, expected {expected}")
    return found


def folder_files(path):
    return [item for item in Path(path).rglob("*") if item.is_file()]


def log_folder_size(path):
    files = folder_files(path)
    sizes = {item: item.stat().st_size for item in files}
    models = [item for item in files if item.name == MODEL_FILE]
    LOGGER.info("Folder %s before zipping: %.1f MB in %d files, %d model files (%.1f MB)", path,
                sum(sizes.values()) / 1e6, len(files), len(models), sum(sizes[item] for item in models) / 1e6)
    largest = sorted(files, key=sizes.get, reverse=True)[:5]
    LOGGER.info("Largest files: %s", ", ".join(f"{item.relative_to(path)} {sizes[item] / 1e6:.1f} MB" for item in largest))
    return sum(sizes.values())


def run_smoke(ctx):
    smoke_ctx = {**ctx, "dir": Path(ctx["dir"]) / "smoke", "smoke": True}
    shutil.rmtree(smoke_ctx["dir"], ignore_errors=True)
    configs = grids.smoke_configs(ctx["section"])
    run_grid(configs, [config.SEEDS[0]], smoke_ctx)
    for cfg in configs:
        check_run_artifacts(Path(smoke_ctx["dir"]) / "runs" / make_run_id(ctx["section"], cfg, config.SEEDS[0]), False)
    check_kept_models(smoke_ctx, len(configs))
    if pending_runs(configs, [config.SEEDS[0]], smoke_ctx):
        raise RuntimeError("resume check failed: finished smoke runs are not skipped")
    run_overfit_check(smoke_ctx, steps=20)
    if ctx["section"] in DIAG_SECTIONS:
        run_diag(smoke_ctx, steps=3)
    LOGGER.info("Smoke test passed: runs, artifacts, results.csv, resume skip and overfit check work (plus diag in Section 3)")


def require_gpu():
    if runtime_info()["gpu"] == "none" and os.environ.get("ALLOW_CPU") != "1":
        raise SystemExit("No GPU visible. Colab: Runtime > Change runtime type > T4 GPU. Kaggle: Settings > Accelerator > GPU T4 x2.")


def run_group(section, group, ctx):
    ctx = {**ctx, "group": group}
    if group == "smoke":
        return run_smoke(ctx)
    require_gpu()
    if group == "timing":
        return run_timing(ctx)
    if group == "diag":
        return run_diag(ctx)
    spec = grids.section_groups(section)[group]
    if spec.get("overfit_check"):
        run_overfit_check(ctx)
    return run_grid(spec["configs"], spec["seeds"], ctx)


def run_diag(ctx, **kwargs):
    from src.diagnostics import run_batch_diagnostics
    return run_batch_diagnostics(ctx, **kwargs)


def group_order(section):
    diag = ["diag"] if section in DIAG_SECTIONS else []
    return ["smoke", "timing"] + diag + list(grids.section_groups(section))


def next_group_message(section, last_group):
    order = group_order(section)
    position = order.index(last_group) + 1
    return f"Next: --section {section} --group {order[position]}" if position < len(order) else f"Section {section} grid complete."


def run_section(section, groups, data_dir, experiments_dir):
    setup_logging(experiments_dir, f"section{section}")
    ctx = {"section": section, "data_dir": Path(data_dir), "dir": Path(experiments_dir), "smoke": False,
           "backup": os.environ.get("USE_BACKUP") == "1", "deadline": section_deadline(section)}
    try:
        run_groups(section, groups, ctx)
    finally:
        finish_section(ctx)


def run_groups(section, groups, ctx):
    for group in groups:
        LOGGER.info("Section %s, group %s: start", section, group)
        run_group(section, group, ctx)
        LOGGER.info("Section %s, group %s: done. %s", section, group, next_group_message(section, group))


def finish_section(ctx):
    prune_models(ctx)
    return log_folder_size(ctx["dir"])


def split_labels(split, data_dir):
    return np.array([label for _, label in list_labeled_files(Path(data_dir) / split)])


def run_reference_baselines(data_dir=config.DATA_DIR, experiments_dir=config.EXPERIMENTS_DIR):
    train, valid = split_labels("train", data_dir), split_labels("valid", data_dir)
    n_classes = len(config.CLASS_NAMES)
    result = {"split": "valid", "majority": metrics.majority_baseline(train, valid),
              "random_train_prior": metrics.random_baseline(np.bincount(train, minlength=n_classes) / len(train), valid),
              "random_uniform": metrics.random_baseline(np.full(n_classes, 1 / n_classes), valid)}
    return run_report.save_json(result, Path(experiments_dir) / "sanity" / "reference_baselines.json")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", type=int)
    parser.add_argument("--group", default="smoke")
    parser.add_argument("--reference-baselines", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.reference_baselines:
        print(run_reference_baselines())
    else:
        run_section(args.section, [g.strip() for g in args.group.split(",")], config.DATA_DIR, config.EXPERIMENTS_DIR)
