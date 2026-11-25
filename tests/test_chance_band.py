import numpy as np
import pytest

from src import chance_band, metrics


def test_batched_macro_f1_matches_classification_metrics():
    rng = np.random.default_rng(3)
    true, predicted = rng.integers(0, 4, 50), rng.integers(0, 4, (3, 50))
    expected = [metrics.classification_metrics(metrics.one_hot(p, 4), true)["macro_f1"] for p in predicted]
    assert chance_band.batched_macro_f1(true, predicted) == pytest.approx(expected)


def test_best_of_k_is_at_least_single_draw_level():
    prior, labels = np.full(4, 0.25), np.repeat(np.arange(4), 20)
    single = chance_band.random_prior_f1(prior, labels, 2000, np.random.default_rng(0))
    best = chance_band.best_of_k_f1(prior, labels, 10, 200, np.random.default_rng(1))
    assert 0.2 < single.mean() < 0.3 and best.mean() > np.quantile(single, 0.75)


def test_cluster_mapping_follows_class_weighted_majority():
    true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    shuffled = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    assert chance_band.cluster_mapping(true, shuffled, [[0], [1]], 2).tolist() == [1, 0]
    assert chance_band.cluster_mapping(true, true, [[0], [1]], 2).tolist() == [0, 1]
    shares = chance_band.weighted_label_shares(true, shuffled, 2)
    assert shares.sum(axis=1) == pytest.approx([1.0, 1.0]) and shares[0, 1] == pytest.approx(0.75)


def test_mapping_f1_identity_and_swap():
    labels = np.array([0, 1, 2, 3, 3])
    assert chance_band.mapping_f1(np.arange(4), labels) == 1.0
    assert chance_band.mapping_f1(np.array([1, 0, 2, 3]), labels) == pytest.approx(0.5)
