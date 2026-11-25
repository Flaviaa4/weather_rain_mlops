import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.eda import INK, INK_SECONDARY, SURFACE, apply_style, style_axis

TRAIN_COLOR, VAL_COLOR = "#2a78d6", "#eb6834"


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, default=float), encoding="utf-8")
    return Path(path)


def read_history(path):
    history = pd.read_csv(path)
    return history.drop_duplicates(subset="epoch", keep="last").reset_index(drop=True)


def best_epoch_index(history, start=0):
    losses = history["val_loss"]
    return int((losses.iloc[start:] if start < len(losses) else losses).idxmin())


def draw_metric(axis, history, metric, best_index):
    epochs = history["epoch"] + 1
    axis.plot(epochs, history[metric], color=TRAIN_COLOR, linewidth=2, label="train")
    axis.plot(epochs, history[f"val_{metric}"], color=VAL_COLOR, linewidth=2, label="validation")
    axis.axvline(best_index + 1, color=INK_SECONDARY, linewidth=1, linestyle="--")
    axis.set_title(metric)
    axis.set_xlabel("epoch")
    style_axis(axis)


def draw_learning_rate(axis, history):
    axis.plot(history["epoch"] + 1, history["learning_rate"], color=INK_SECONDARY, linewidth=2)
    axis.set_yscale("log")
    axis.set_title("learning rate")
    axis.set_xlabel("epoch")
    style_axis(axis)


def plot_curves(history, path, title, start=0):
    apply_style()
    best = best_epoch_index(history, start)
    figure, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    draw_metric(axes[0], history, "loss", best)
    draw_metric(axes[1], history, "accuracy", best)
    draw_learning_rate(axes[2], history)
    axes[0].legend(frameon=False)
    figure.suptitle(f"{title}: best epoch {best + 1} of {len(history)} (dashed)", x=0.01, ha="left", fontweight="bold")
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return Path(path)


def annotate_cells(axis, matrix, shares):
    for (row, column), count in np.ndenumerate(matrix):
        color = SURFACE if shares[row, column] > 0.55 else INK
        axis.text(column, row, f"{count}\n{shares[row, column]:.0%}", ha="center", va="center", color=color, fontsize=9)


def plot_confusion_matrix(matrix, path, title, class_names=config.CLASS_NAMES):
    apply_style()
    matrix = np.asarray(matrix)
    shares = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    figure, axis = plt.subplots(figsize=(5.2, 4.6))
    axis.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    annotate_cells(axis, matrix, shares)
    axis.set_xticks(range(len(class_names)), class_names)
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("predicted"), axis.set_ylabel("true")
    axis.tick_params(length=0)
    [spine.set_visible(False) for spine in axis.spines.values()]
    axis.set_title(f"{title}\ncount and share of true class", loc="left")
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return Path(path)


def prediction_frame(paths, labels, probabilities, class_names=config.CLASS_NAMES):
    predicted = probabilities.argmax(axis=1)
    frame = pd.DataFrame({"path": paths, "true": [class_names[k] for k in labels],
                          "predicted": [class_names[k] for k in predicted], "confidence": probabilities.max(axis=1)})
    for k, name in enumerate(class_names):
        frame[f"prob_{name}"] = probabilities[:, k]
    return frame


def save_predictions(paths, labels, probabilities, path):
    frame = prediction_frame(paths, labels, probabilities)
    frame.to_csv(path, index=False)
    return frame
