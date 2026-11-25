import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import config, run_report
from src.experiments import split_labels

N_CLASSES = len(config.CLASS_NAMES)
SINGLE_DRAWS = 20000
BEST_OF_GROUPS = 5000
PERMUTATION_DRAWS = 2000
QUANTILES = [0.025, 0.5, 0.975, 0.999]
SHUFFLED_RUN_ID = "s3_r0_shuffled_labels_seed11"
PARTITIONS = {"four_clusters": [[0], [1], [2], [3]], "cloudy_shiny_merged_post_hoc": [[0, 2], [1], [3]]}


def batched_confusion(true, predicted, n_classes=N_CLASSES):
    draws = predicted.shape[0]
    codes = np.arange(draws)[:, None] * n_classes ** 2 + true[None, :] * n_classes + predicted
    return np.bincount(codes.ravel(), minlength=draws * n_classes ** 2).reshape(draws, n_classes, n_classes)


def batched_macro_f1(true, predicted, n_classes=N_CLASSES):
    matrices = batched_confusion(np.asarray(true), np.atleast_2d(predicted), n_classes).astype(float)
    hits = np.diagonal(matrices, axis1=1, axis2=2)
    denominator = matrices.sum(axis=1) + matrices.sum(axis=2)
    f1 = np.divide(2 * hits, denominator, out=np.zeros_like(hits), where=denominator > 0)
    return f1.mean(axis=1)


def random_prior_f1(prior, val_labels, draws, rng):
    guesses = rng.choice(len(prior), size=(draws, len(val_labels)), p=prior)
    return batched_macro_f1(val_labels, guesses)


def best_of_k_f1(prior, val_labels, k, groups, rng):
    return random_prior_f1(prior, val_labels, k * groups, rng).reshape(groups, k).max(axis=1)


def summarize(values, observed):
    quantiles = np.quantile(values, QUANTILES)
    return {"mean": float(values.mean()), "std": float(values.std()),
            **{f"q{q * 100:g}": float(v) for q, v in zip(QUANTILES, quantiles)}, "max": float(values.max()),
            "share_at_or_above_observed": float((values >= observed).mean())}


def class_weighted_counts(true, shuffled, n_classes=N_CLASSES):
    counts = batched_confusion(np.asarray(true), np.asarray(shuffled)[None, :], n_classes)[0].astype(float)
    return counts * (len(shuffled) / (n_classes * counts.sum(axis=0)))[None, :]


def weighted_label_shares(true, shuffled, n_classes=N_CLASSES):
    weighted = class_weighted_counts(true, shuffled, n_classes)
    return weighted / weighted.sum(axis=1, keepdims=True)


def cluster_mapping(true, shuffled, clusters, n_classes=N_CLASSES):
    weighted = class_weighted_counts(true, shuffled, n_classes)
    mapping = np.zeros(n_classes, dtype=int)
    for members in clusters:
        mapping[members] = int(weighted[members].sum(axis=0).argmax())
    return mapping


def mapping_f1(mapping, val_labels):
    return float(batched_macro_f1(val_labels, mapping[np.asarray(val_labels)][None, :])[0])


def permutation_null(train_labels, val_labels, clusters, draws, rng):
    shuffles = (rng.permutation(train_labels) for _ in range(draws))
    return np.array([mapping_f1(cluster_mapping(train_labels, shuffled, clusters), val_labels) for shuffled in shuffles])


def observed_shuffled_run(experiments_dir):
    results = pd.read_csv(Path(experiments_dir) / "results.csv")
    row = results.loc[results["run_id"] == SHUFFLED_RUN_ID].iloc[-1]
    return float(row["val_macro_f1"]), int(row["epochs_run"]), int(row["seed"])


def random_prediction_bands(prior, val_labels, observed, epochs, rng):
    return {"single_draw": summarize(random_prior_f1(prior, val_labels, SINGLE_DRAWS, rng), observed),
            f"best_of_{epochs}_draws": summarize(best_of_k_f1(prior, val_labels, epochs, BEST_OF_GROUPS, rng), observed)}


def run_permutation_signal(train_labels, val_labels, seed):
    shuffled = np.random.default_rng(seed).permutation(train_labels)
    mappings = {name: cluster_mapping(train_labels, shuffled, clusters) for name, clusters in PARTITIONS.items()}
    return {"class_weighted_label_shares": np.round(weighted_label_shares(train_labels, shuffled), 4).tolist(),
            "agreement_with_true_labels": float((shuffled == train_labels).mean()),
            "mappings": {name: [config.CLASS_NAMES[k] for k in mapping] for name, mapping in mappings.items()},
            "mapping_macro_f1": {name: mapping_f1(mapping, val_labels) for name, mapping in mappings.items()}}


def build_chance_band(data_dir=config.DATA_DIR, experiments_dir=config.EXPERIMENTS_DIR, seed=config.SAMPLE_SEED):
    train_labels, val_labels = split_labels("train", data_dir), split_labels("valid", data_dir)
    observed, epochs, run_seed = observed_shuffled_run(experiments_dir)
    prior = np.bincount(train_labels, minlength=N_CLASSES) / len(train_labels)
    rng = np.random.default_rng(seed)
    nulls = {name: summarize(permutation_null(train_labels, val_labels, clusters, PERMUTATION_DRAWS, rng), observed)
             for name, clusters in PARTITIONS.items()}
    result = {"observed_val_macro_f1": observed, "epochs_run": epochs, "train_prior": prior.round(4).tolist(),
              "random_predictions": random_prediction_bands(prior, val_labels, observed, epochs, rng),
              "this_permutation": run_permutation_signal(train_labels, val_labels, run_seed),
              "permutation_null_cluster_oracle": nulls}
    return run_report.save_json(result, Path(experiments_dir) / "sanity" / "chance_band.json")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=config.SAMPLE_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    print(build_chance_band(seed=parse_args().seed).read_text())
