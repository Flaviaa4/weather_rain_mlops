import numpy as np
import pandas as pd

from src import config


def confusion_matrix(y_true, y_pred, n_classes):
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return matrix


def safe_divide(numerator, denominator):
    numerator, denominator = np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def per_class_scores(matrix):
    true_positive = np.diag(matrix)
    precision = safe_divide(true_positive, matrix.sum(axis=0))
    recall = safe_divide(true_positive, matrix.sum(axis=1))
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "support": matrix.sum(axis=1)}


def accuracy_from_matrix(matrix):
    return float(np.trace(matrix) / max(matrix.sum(), 1))


def log_loss(probabilities, y_true):
    picked = probabilities[np.arange(len(y_true)), np.asarray(y_true)]
    return float(-np.mean(np.log(np.clip(picked, 1e-7, 1.0))))


def rank_auc(scores, is_positive):
    is_positive = np.asarray(is_positive, dtype=bool)
    positives, negatives = int(is_positive.sum()), int((~is_positive).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = pd.Series(scores).rank().to_numpy()
    return float((ranks[is_positive].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def one_vs_rest_auc(probabilities, y_true):
    y_true = np.asarray(y_true)
    return np.array([rank_auc(probabilities[:, k], y_true == k) for k in range(probabilities.shape[1])])


def per_class_report(scores, aucs, class_names):
    return {name: {"precision": float(scores["precision"][k]), "recall": float(scores["recall"][k]),
                   "f1": float(scores["f1"][k]), "support": int(scores["support"][k]), "roc_auc": float(aucs[k])}
            for k, name in enumerate(class_names)}


def classification_metrics(probabilities, y_true, class_names=config.CLASS_NAMES):
    probabilities, y_true = np.asarray(probabilities, dtype=float), np.asarray(y_true)
    matrix = confusion_matrix(y_true, probabilities.argmax(axis=1), len(class_names))
    scores, aucs = per_class_scores(matrix), one_vs_rest_auc(probabilities, y_true)
    return {
        "accuracy": accuracy_from_matrix(matrix), "loss": log_loss(probabilities, y_true),
        "macro_precision": float(scores["precision"].mean()), "macro_recall": float(scores["recall"].mean()),
        "macro_f1": float(scores["f1"].mean()), "roc_auc_macro": float(np.nanmean(aucs)),
        "per_class": per_class_report(scores, aucs, class_names), "confusion_matrix": matrix.tolist(),
    }


def flat_metric_fields(metrics, prefix="val_"):
    fields = {f"{prefix}{key}": metrics[key] for key in
              ["accuracy", "loss", "macro_precision", "macro_recall", "macro_f1", "roc_auc_macro"]}
    for name, report in metrics["per_class"].items():
        fields[f"{prefix}f1_{name}"] = report["f1"]
        fields[f"{prefix}recall_{name}"] = report["recall"]
    return fields


def one_hot(labels, n_classes):
    return np.eye(n_classes, dtype=np.float32)[np.asarray(labels)]


def majority_baseline(train_labels, val_labels, n_classes=len(config.CLASS_NAMES)):
    majority = int(np.bincount(train_labels, minlength=n_classes).argmax())
    probabilities = one_hot(np.full(len(val_labels), majority), n_classes)
    return {"predicted_class": config.CLASS_NAMES[majority], **classification_metrics(probabilities, val_labels)}


def random_guess_scores(prior, val_labels, rng):
    guesses = rng.choice(len(prior), size=len(val_labels), p=prior)
    metrics = classification_metrics(one_hot(guesses, len(prior)), val_labels)
    return metrics["accuracy"], metrics["macro_f1"], metrics["per_class"]["rainy"]["f1"]


def random_baseline(prior, val_labels, draws=1000, seed=config.SAMPLE_SEED):
    rng = np.random.default_rng(seed)
    scores = np.array([random_guess_scores(np.asarray(prior), val_labels, rng) for _ in range(draws)])
    return {name: {"mean": float(scores[:, k].mean()), "std": float(scores[:, k].std())}
            for k, name in enumerate(["accuracy", "macro_f1", "f1_rainy"])}
