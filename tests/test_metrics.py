import numpy as np
import pytest

from src import metrics


def test_confusion_matrix_counts():
    matrix = metrics.confusion_matrix([0, 0, 1, 2, 2, 2], [0, 1, 1, 2, 2, 0], 3)
    assert matrix.tolist() == [[1, 1, 0], [0, 1, 0], [1, 0, 2]]


def test_per_class_scores_known_values():
    matrix = np.array([[8, 2], [4, 6]])
    scores = metrics.per_class_scores(matrix)
    assert scores["precision"] == pytest.approx([8 / 12, 6 / 8])
    assert scores["recall"] == pytest.approx([0.8, 0.6])
    assert scores["f1"][0] == pytest.approx(2 * (8 / 12) * 0.8 / ((8 / 12) + 0.8))


def test_missing_prediction_gives_zero_not_nan():
    scores = metrics.per_class_scores(metrics.confusion_matrix([0, 1], [0, 0], 2))
    assert scores["precision"][1] == 0.0 and scores["f1"][1] == 0.0


def test_classification_metrics_perfect_predictions():
    labels = np.array([0, 1, 2, 3, 1])
    result = metrics.classification_metrics(metrics.one_hot(labels, 4) * 0.97 + 0.0075, labels)
    assert result["accuracy"] == 1.0 and result["macro_f1"] == 1.0 and result["roc_auc_macro"] == 1.0
    assert result["loss"] == pytest.approx(-np.log(0.9775), rel=1e-4)
    assert result["per_class"]["rainy"]["support"] == 2


def test_rank_auc_handles_ties_and_single_class():
    assert metrics.rank_auc([0.5, 0.5], [True, False]) == 0.5
    assert np.isnan(metrics.rank_auc([0.1, 0.2], [True, True]))


def test_flat_metric_fields_names():
    labels = np.array([0, 1, 2, 3])
    fields = metrics.flat_metric_fields(metrics.classification_metrics(metrics.one_hot(labels, 4), labels))
    assert {"val_macro_f1", "val_f1_rainy", "val_recall_sunrise", "val_roc_auc_macro"} <= set(fields)


def test_majority_baseline_predicts_most_common_train_class():
    result = metrics.majority_baseline(np.array([3, 3, 3, 0]), np.array([0, 1, 3, 3]))
    assert result["predicted_class"] == "sunrise" and result["accuracy"] == 0.5


def test_random_baseline_uniform_is_near_chance():
    labels = np.repeat(np.arange(4), 50)
    result = metrics.random_baseline(np.full(4, 0.25), labels, draws=300)
    assert result["accuracy"]["mean"] == pytest.approx(0.25, abs=0.02)
