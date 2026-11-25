import json

import numpy as np
import pandas as pd
import pytest

from src import config, final_model, metrics, notebook_views as views, run_report


def fake_split(rng, n):
    labels = np.tile(np.arange(4), n // 4)
    probabilities = rng.random((len(labels), 4))
    probabilities[np.arange(len(labels)), labels] += 0.8
    return probabilities / probabilities.sum(axis=1, keepdims=True), labels


def fake_record():
    rng = np.random.default_rng(0)
    (val_p, val_y), (test_p, test_y) = fake_split(rng, 144), fake_split(rng, 136)
    parity = {"n_images": 144, "max_abs_diff": 1e-7, "same_class_share": 1.0, "tolerance": 1e-4, "passed": True}
    return {"version": "weather_cnn_x", "run_id": "s7_x_seed22", "seed": 22, "created_at": "2026-11-25",
            "export_method": "from_keras", "config_hash": "abc", "dataset": {"valid": {"n_images": 144}},
            "validation": metrics.classification_metrics(val_p, val_y), "test": metrics.classification_metrics(test_p, test_y),
            "test_bootstrap_95ci": final_model.bootstrap_ci(test_p, test_y, draws=200), "parity": {"validation": parity, "test": parity}}


def write_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    rng, rows = np.random.default_rng(1), []
    for seed in [11, 22, 33]:
        probabilities, labels = fake_split(rng, 40)
        folder = tmp_path / "runs" / f"s5_m_seed{seed}"
        folder.mkdir(parents=True)
        paths = [f"valid/{config.CLASS_NAMES[k]}/img{i}.jpg" for i, k in enumerate(labels)]
        run_report.save_predictions(paths, labels, probabilities, folder / "val_predictions.csv")
        rows.append({"run_id": f"s5_m_seed{seed}", "cfg_name": "m", "seed": seed})
    return pd.DataFrame(rows)


def test_final_views_read_a_metrics_record(tmp_path):
    record = fake_record()
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(record, default=float))
    loaded = views.load_final_metrics(path)
    overview = views.final_overview(loaded)
    assert list(overview.index) == views.METRIC_NAMES and "(" in overview.loc["macro_f1", "test (95% CI)"]
    assert "(" not in overview.loc["loss", "test (95% CI)"]
    assert list(views.final_per_class(loaded).index) == config.CLASS_NAMES
    assert views.parity_table(loaded)["passed"].all()
    assert views.final_description(loaded)["valid images"] == 144
    assert views.final_confusions(loaded) is not None


def test_missing_or_placeholder_metrics_give_none(tmp_path):
    assert views.load_final_metrics(tmp_path / "missing.json") is None
    (tmp_path / "placeholder.json").write_text(json.dumps({"validation": {}}))
    assert views.load_final_metrics(tmp_path / "placeholder.json") is None
    assert "src.final_model --install" in views.final_model_missing()


def test_seed_level_views_from_saved_predictions(tmp_path, monkeypatch):
    results = write_runs(tmp_path, monkeypatch)
    table = views.seed_metrics_table(results, "m")
    assert list(table.index[-2:]) == ["mean", "std"] and len(table) == 5
    assert set(views.METRIC_NAMES) <= set(table.columns)
    per_class = views.per_class_table(results, ["m"])
    assert list(per_class.index) == config.CLASS_NAMES and ("recall", "m") in per_class.columns
    assert views.pooled_confusion(results, "m").sum() == 120
    wrong = views.misclassified("s5_m_seed11")
    assert (wrong["true"] != wrong["predicted"]).all() and wrong["confidence"].is_monotonic_decreasing
    assert len(views.correct_examples("s5_m_seed11")) == 4


def test_verdict_and_collapsed_seeds():
    reference = pd.Series({"f1_mean": 0.90, "f1_std": 0.02})
    assert views.verdict(pd.Series({"f1_mean": 0.91, "f1_std": 0.01}), reference).startswith("no difference")
    assert views.verdict(pd.Series({"f1_mean": 0.95, "f1_std": 0.01}), reference) == "better"
    results = pd.DataFrame({"cfg_name": ["a", "a", "a"], "best_epoch": [1, 2, 30]})
    assert views.collapsed_seeds(results, "a") == 2


def test_preprocessing_steps_and_momentum_table(tmp_path, monkeypatch, make_tiny_dataset):
    monkeypatch.setattr(config, "DATA_DIR", make_tiny_dataset(tmp_path / "data"))
    table, images = views.preprocessing_steps(views.example_image("train", "rainy", 0))
    assert table.iloc[-1]["dtype"] == "float32" and 0.0 <= table.iloc[-1]["min"] <= table.iloc[-1]["max"] <= 1.0
    assert table.iloc[-2]["shape"] == f"({config.IMAGE_SIZE}, {config.IMAGE_SIZE}, 3)"
    assert views.plot_preprocessing(images) is not None
    assert views.momentum_lag_table().iloc[0, 0] == "79.4%"


def test_predict_examples_runs_the_onnx_model(tmp_path, monkeypatch, make_tiny_dataset):
    pytest.importorskip("onnxruntime")
    import onnx
    from src import placeholder
    data_dir = make_tiny_dataset(tmp_path / "data")
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    onnx.save(placeholder.build_placeholder_onnx(), str(tmp_path / "model.onnx"))
    rows = pd.DataFrame({"path": ["valid/rainy/rainy0.jpg", "valid/shiny/shiny1.jpg"], "true": ["rainy", "shiny"]})
    table = views.predict_examples(rows, tmp_path / "model.onnx")
    assert list(table.columns) == ["label", "confidence", "uncertain", "true"] and len(table) == 2
    assert set(table["label"]) <= set(config.CLASS_NAMES)


def test_technique_table_on_saved_results():
    if not config.RESULTS_CSV.exists() or 5 not in set(pd.read_csv(config.RESULTS_CSV)["section"]):
        pytest.skip("no Section 5 results in this checkout")
    table = views.technique_table(views.load_results())
    assert len(table) == len(views.TECHNIQUES)
    assert table.loc["Batch norm momentum 0.9 (instead of 0.99)", "verdict"] == "better"
