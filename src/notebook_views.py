import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src import config, metrics, selection
from src.eda import CLASS_COLORS, INK, INK_SECONDARY, SURFACE, apply_style, style_axis
from src.preprocessing import list_labeled_files, normalize_array, open_image_bytes, resize_image, to_array, to_rgb
from src.run_report import annotate_cells

SELECTED_NAME, BASELINE_NAME = "r0_patience30_min20", "baseline"
SELECTED_COLOR, BASELINE_COLOR = "#2a78d6", "#eb6834"
CHANNEL_COLORS = {"red": "#eb6834", "green": "#1baf7a", "blue": "#2a78d6"}
STEPS_PER_EPOCH = 23
FINAL_METRICS_PATH = config.MODELS_DIR / "metrics.json"
FINAL_ONNX_PATH = config.MODELS_DIR / "weather_cnn.onnx"
METRIC_NAMES = ["accuracy", "loss", "macro_precision", "macro_recall", "macro_f1", "roc_auc_macro"]
TECHNIQUES = [
    ("Batch norm momentum 0.9 (instead of 0.99)", "r0_bn_momentum_0.9", "r0"),
    ("Batch norm momentum 0.8", "r0_bn_momentum_0.8", "r0"),
    ("Early stopping patience 30, no stop before epoch 21", "r0_patience30_min20", "r0"),
    ("Learning rate 3e-4 (instead of 1e-3)", "r0_lr3e-4", "r0"),
    ("5-epoch learning rate warmup", "r0_warmup5", "r0"),
    ("Light augmentation (instead of medium)", "r0_augment_light", "r0"),
    ("Batch norm added to the baseline", "baseline_plus_batch_norm", "baseline"),
    ("Dropout 0.3 added to the baseline", "baseline_plus_dropout", "baseline"),
    ("L2 1e-4 added to the baseline", "baseline_plus_l2", "baseline"),
    ("Class weights added to the baseline", "baseline_plus_class_weights", "baseline"),
    ("Medium augmentation added to the baseline", "baseline_plus_augment", "baseline"),
    ("Global average pooling added to the baseline", "baseline_plus_gap", "baseline"),
]


def finish(figure):
    figure.tight_layout()
    plt.close(figure)
    return figure


def split_counts():
    return pd.read_csv(config.AUDIT_DIR / "after_split_class_counts.csv").set_index("split")


def plot_split_counts(counts):
    apply_style()
    splits, width = ["train", "valid", "test"], 0.2
    figure, axis = plt.subplots(figsize=(7.5, 3.6))
    for k, name in enumerate(config.CLASS_NAMES):
        bars = axis.bar(np.arange(3) + (k - 1.5) * width, counts.loc[splits, name], width, color=CLASS_COLORS[name],
                        edgecolor=SURFACE, linewidth=2, label=name)
        axis.bar_label(bars, padding=2, color=INK_SECONDARY, fontsize=8)
    axis.set_xticks(range(3), [f"{split} ({counts.loc[split, 'total']})" for split in splits])
    axis.set_ylabel("images")
    axis.set_title("Images per class and split after duplicate cleanup", loc="left")
    axis.legend(frameon=False, ncol=4, loc="upper right")
    style_axis(axis)
    return finish(figure)


def example_image(split="train", class_name="rainy", index=0):
    files = [path for path, label in list_labeled_files(config.DATA_DIR / split) if config.CLASS_NAMES[label] == class_name]
    return sorted(files, key=lambda path: Path(path).name)[index]


def describe(step, function, value):
    if isinstance(value, Image.Image):
        return {"step": step, "function": function, "shape": f"{value.size[0]} x {value.size[1]} (width x height)",
                "dtype": f"PIL mode {value.mode}", "min": None, "max": None}
    return {"step": step, "function": function, "shape": str(value.shape), "dtype": str(value.dtype),
            "min": float(value.min()), "max": float(value.max())}


def preprocessing_steps(path, size=config.IMAGE_SIZE):
    opened = open_image_bytes(Path(path).read_bytes())
    rgb = to_rgb(opened)
    resized = resize_image(rgb, size)
    array = to_array(resized)
    normalized = normalize_array(array)
    steps = [("1 decode the file", "open_image_bytes", opened), ("2 fix orientation, force RGB", "to_rgb", rgb),
             ("3 resize", f"resize_image(size={size})", resized), ("4 to an array", "to_array", array),
             ("5 scale to 0..1", "normalize_array", normalized)]
    table = pd.DataFrame([describe(*step) for step in steps]).set_index("step")
    return table, {"original": rgb, "resized": resized, "normalized": normalized}


def plot_preprocessing(images):
    apply_style()
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.6, 1, 1.3]})
    width, height = images["original"].size
    axes[0].imshow(images["original"]), axes[0].set_title(f"original {width} x {height}", loc="left")
    axes[1].imshow(images["resized"]), axes[1].set_title("resized 128 x 128", loc="left")
    [axis.axis("off") for axis in axes[:2]]
    for k, (name, color) in enumerate(CHANNEL_COLORS.items()):
        axes[2].hist(images["normalized"][..., k].ravel(), bins=40, range=(0, 1), histtype="step", linewidth=2,
                     color=color, label=name)
    axes[2].set_title("pixel values after scaling", loc="left"), axes[2].set_xlabel("value"), axes[2].set_ylabel("pixels")
    axes[2].legend(frameon=False)
    style_axis(axes[2])
    return finish(figure)


def load_results():
    return pd.read_csv(config.RESULTS_CSV)


def summary(results, names):
    table = selection.config_summary(results)
    table = table[table["cfg_name"].isin(names)].set_index("cfg_name").loc[names]
    columns = ["seeds", "f1_mean", "f1_std", "f1_min", "f1_max", "rainy_f1_mean", "rainy_f1_min", "best_epoch_mean",
               "epochs_mean", "minutes_mean"]
    return table[columns].round(3)


def config_color(name):
    return {SELECTED_NAME: SELECTED_COLOR, BASELINE_NAME: BASELINE_COLOR}.get(name, INK_SECONDARY)


def seed_legend_handles():
    from matplotlib.lines import Line2D
    return [Line2D([], [], marker="o", linestyle="", color=INK_SECONDARY, alpha=0.55, label="one seed"),
            Line2D([], [], color=INK_SECONDARY, linewidth=2.5, label="mean over seeds")]


def plot_seed_scores(results, names, title):
    apply_style()
    height = 0.45 * len(names) + 0.6
    figure, axis = plt.subplots(figsize=(8, height + 1.2))
    for row, name in enumerate(names):
        scores = results.loc[results["cfg_name"] == name, "val_macro_f1"]
        axis.scatter(scores, [row] * len(scores), s=40, color=config_color(name), alpha=0.55, linewidths=0)
        axis.plot([scores.mean()] * 2, [row - 0.3, row + 0.3], color=config_color(name), linewidth=2.5)
    axis.set_yticks(range(len(names)), names), axis.invert_yaxis(), axis.set_ylim(len(names) - 0.5, -0.5)
    axis.set_xlabel("validation macro F1"), axis.set_title(title, loc="left")
    axis.legend(handles=seed_legend_handles(), frameon=False, loc="upper left", bbox_to_anchor=(0, -0.55 / height), ncol=2)
    style_axis(axis, "x")
    return finish(figure)


def history(run_id):
    return pd.read_csv(config.RUNS_DIR / run_id / "history.csv")


def plot_stall(failed_run, converged_run, epochs=30):
    apply_style()
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for run, color in [(failed_run, BASELINE_COLOR), (converged_run, SELECTED_COLOR)]:
        frame = history(run).head(epochs)
        x = np.arange(1, len(frame) + 1)
        axes[0].plot(x, frame["val_loss"], color=color, linewidth=2, label=run)
        axes[1].plot(x, frame["accuracy"], color=color, linewidth=2, label=run)
    axes[0].set_title("validation loss (inference mode)", loc="left"), axes[0].set_ylabel("loss")
    axes[1].set_title("training accuracy (training mode)", loc="left"), axes[1].set_ylabel("accuracy")
    for axis in axes:
        axis.set_xlabel("epoch"), style_axis(axis)
    axes[0].legend(frameon=False)
    return finish(figure)


def momentum_lag_table(momenta=(0.99, 0.9, 0.8), epochs=(1, 2, 5, 10, 15, 20)):
    rows = {f"momentum {m}": {f"after {e} epochs": m ** (STEPS_PER_EPOCH * e) for e in epochs} for m in momenta}
    return pd.DataFrame(rows).T.map(lambda share: f"{share:.1%}")


def collapsed_seeds(results, name):
    rows = results[results["cfg_name"] == name]
    return int((rows["best_epoch"] <= 2).sum())


def verdict(row, reference):
    if selection.is_tie(row, reference):
        return "no difference (within one std)"
    return "better" if row["f1_mean"] > reference["f1_mean"] else "worse"


def technique_table(results):
    table = selection.config_summary(results).set_index("cfg_name")
    rows = []
    for technique, name, reference in TECHNIQUES:
        row, ref = table.loc[name], table.loc[reference]
        rows.append({"technique": technique, "config": name, "seeds": int(row["seeds"]), "macro F1 mean": row["f1_mean"],
                     "std": row["f1_std"], "worst seed": row["f1_min"], "compared with": reference,
                     "difference": row["f1_mean"] - ref["f1_mean"], "collapsed seeds": collapsed_seeds(results, name),
                     "verdict": verdict(row, ref)})
    return pd.DataFrame(rows).set_index("technique").round(3)


def config_run_ids(results, name):
    return list(results.loc[results["cfg_name"] == name].sort_values("seed")["run_id"])


def run_predictions(run_id):
    return pd.read_csv(config.RUNS_DIR / run_id / "val_predictions.csv")


def run_metrics(run_id):
    frame = run_predictions(run_id)
    probabilities = frame[[f"prob_{name}" for name in config.CLASS_NAMES]].to_numpy()
    labels = frame["true"].map(config.CLASS_NAMES.index).to_numpy()
    return metrics.classification_metrics(probabilities, labels)


def seed_metrics_table(results, name):
    rows = {run_id: {key: run_metrics(run_id)[key] for key in METRIC_NAMES} for run_id in config_run_ids(results, name)}
    table = pd.DataFrame(rows).T
    return pd.concat([table, table.agg(["mean", "std"])]).round(3)


def per_class_table(results, names):
    rows = []
    for name in names:
        for run_id in config_run_ids(results, name):
            for class_name, report in run_metrics(run_id)["per_class"].items():
                rows.append({"config": name, "class": class_name, "precision": report["precision"],
                             "recall": report["recall"], "f1": report["f1"]})
    table = pd.DataFrame(rows).groupby(["class", "config"])[["precision", "recall", "f1"]].mean()
    return table.unstack("config").reindex(config.CLASS_NAMES).round(3)


def pooled_confusion(results, name):
    return sum(np.asarray(run_metrics(run_id)["confusion_matrix"]) for run_id in config_run_ids(results, name))


def draw_confusion(axis, matrix, title):
    matrix = np.asarray(matrix)
    shares = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    axis.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    annotate_cells(axis, matrix, shares)
    axis.set_xticks(range(len(config.CLASS_NAMES)), config.CLASS_NAMES)
    axis.set_yticks(range(len(config.CLASS_NAMES)), config.CLASS_NAMES)
    axis.set_xlabel("predicted"), axis.set_ylabel("true"), axis.set_title(title, loc="left")
    axis.tick_params(length=0)
    [spine.set_visible(False) for spine in axis.spines.values()]


def plot_confusions(matrices):
    apply_style()
    figure, axes = plt.subplots(1, len(matrices), figsize=(5.2 * len(matrices), 4.6))
    for axis, (title, matrix) in zip(np.atleast_1d(axes), matrices.items()):
        draw_confusion(axis, matrix, title)
    return finish(figure)


def misclassified(run_id, limit=8):
    frame = run_predictions(run_id)
    wrong = frame[frame["true"] != frame["predicted"]]
    return wrong.sort_values("confidence", ascending=False).head(limit)


def correct_examples(run_id, per_class=1):
    frame = run_predictions(run_id)
    right = frame[frame["true"] == frame["predicted"]]
    return right.groupby("true", group_keys=False).head(per_class)


def demo_rows(run_id, wrong=4, per_class=1):
    return pd.concat([correct_examples(run_id, per_class), misclassified(run_id, wrong)])


def plot_gallery(rows, columns=4, size=2.6):
    apply_style()
    lines = max(1, int(np.ceil(len(rows) / columns)))
    figure, axes = plt.subplots(lines, columns, figsize=(columns * size, lines * (size + 0.4)))
    for axis in np.ravel(axes):
        axis.axis("off")
    for axis, (_, row) in zip(np.ravel(axes), rows.iterrows()):
        axis.imshow(Image.open(config.DATA_DIR / row["path"]).convert("RGB"))
        axis.set_title(f"true {row['true']}\npredicted {row['predicted']} ({row['confidence']:.2f})", fontsize=9,
                       color=INK if row["true"] == row["predicted"] else BASELINE_COLOR, loc="left")
    return finish(figure)


def load_final_metrics(path=FINAL_METRICS_PATH):
    if not Path(path).exists():
        return None
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    return record if "test" in record and "validation" in record else None


def final_model_missing():
    return (f"{FINAL_METRICS_PATH} is not installed yet. Run Section 7 on Kaggle (SECTION = 7, GROUP = \"smoke,final\"), "
            "then python -m src.final_model --install experiments/incoming/section7, and run this notebook again.")


def final_description(record):
    keys = ["version", "run_id", "seed", "created_at", "export_method", "config_hash"]
    info = {key: record.get(key) for key in keys}
    info.update({f"{split} images": record["dataset"][split]["n_images"] for split in record.get("dataset", {})})
    return pd.Series(info, name="final model")


def with_interval(value, bounds):
    return f"{value:.3f} ({bounds[0]:.3f} to {bounds[1]:.3f})" if bounds else f"{value:.3f}"


def final_overview(record):
    intervals = record.get("test_bootstrap_95ci", {})
    rows = {name: {"validation": f"{record['validation'][name]:.3f}",
                   "test (95% CI)": with_interval(record["test"][name], intervals.get(name))}
            for name in METRIC_NAMES}
    table = pd.DataFrame(rows).T
    table["test minus validation"] = [f"{record['test'][name] - record['validation'][name]:+.3f}" for name in METRIC_NAMES]
    return table


def final_per_class(record):
    intervals = record.get("test_bootstrap_95ci", {}).get("per_class_f1", {})
    rows = {}
    for name in config.CLASS_NAMES:
        validation, test = record["validation"]["per_class"][name], record["test"]["per_class"][name]
        rows[name] = {"val precision": validation["precision"], "val recall": validation["recall"], "val f1": validation["f1"],
                      "test precision": test["precision"], "test recall": test["recall"],
                      "test f1 (95% CI)": with_interval(test["f1"], intervals.get(name)), "test images": test["support"]}
    return pd.DataFrame(rows).T


def final_confusions(record):
    return plot_confusions({"validation (final model)": record["validation"]["confusion_matrix"],
                            "test (final model, evaluated once)": record["test"]["confusion_matrix"]})


def parity_table(record):
    return pd.DataFrame(record["parity"]).T[["n_images", "max_abs_diff", "same_class_share", "tolerance", "passed"]]


def predict_examples(rows, onnx_path=FINAL_ONNX_PATH):
    from src import prediction
    from src.preprocessing import preprocess_image_file
    images = np.stack([preprocess_image_file(config.DATA_DIR / path) for path in rows["path"]])
    outputs = [prediction.format_prediction(p, config.CLASS_NAMES) for p in prediction.predict_probabilities(images, onnx_path)]
    prediction.drop_session(onnx_path)
    table = pd.DataFrame([{key: out[key] for key in ["label", "confidence", "uncertain"]} for out in outputs])
    return table.assign(true=list(rows["true"]), file=[Path(path).name for path in rows["path"]]).set_index("file")
