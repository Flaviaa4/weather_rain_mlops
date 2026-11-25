import json
import shutil

import numpy as np
import pandas as pd
import pytest

from src import experiments, final_model, grids, selection
from tests.conftest import write_tiny_dataset

pytest.importorskip("onnxruntime")
SIZE = 32


class OnnxBackedModel:
    def __init__(self, onnx_path, noise=0.0):
        self.onnx_path, self.noise = onnx_path, noise

    def predict(self, images, batch_size=32, verbose=0):
        from src import prediction
        probabilities = prediction.predict_probabilities(images, self.onnx_path)
        prediction.drop_session(self.onnx_path)
        return probabilities + self.noise


def save_onnx(path):
    import onnx
    from src import placeholder
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(placeholder.build_placeholder_onnx(size=SIZE), str(path))
    return path


def fake_export(model, paths, run_id):
    save_onnx(paths["onnx"])
    paths["h5"].write_bytes(b"h5")
    model.onnx_path = paths["onnx"]
    return "fake"


@pytest.fixture
def final_ctx(tmp_path, monkeypatch):
    data_dir = write_tiny_dataset(tmp_path / "data", per_class=6)
    shutil.copytree(data_dir / "valid", data_dir / "test")
    cfg = {**grids.final_config(), "image_size": SIZE}
    ctx = {"section": 7, "dir": tmp_path / "experiments", "data_dir": data_dir, "smoke": False}
    for seed, f1 in [(11, 0.90), (22, 0.95), (33, 0.93)]:
        row = {"run_id": experiments.make_run_id(7, cfg, seed), "section": 7, "seed": seed, "val_loss": 0.3,
               "config_hash": experiments.config_hash(cfg), "val_macro_f1": f1}
        experiments.log_result_row(row, ctx["dir"])
    monkeypatch.setattr(final_model, "load_run_model", lambda run_ctx, run_id: OnnxBackedModel(None))
    monkeypatch.setattr(final_model, "export_model", fake_export)
    monkeypatch.setattr(final_model, "keras_probabilities", lambda model, images: model.predict(images))
    monkeypatch.setattr(experiments, "runtime_info", lambda: {"tf_version": "test", "gpu": "none"})
    return cfg, ctx


def test_parity_needs_enough_images_small_difference_and_same_classes():
    probs = np.tile([[0.7, 0.1, 0.1, 0.1]], (25, 1))
    assert final_model.parity_report(probs, probs + 1e-6)["passed"]
    assert not final_model.parity_report(probs[:10], probs[:10])["passed"]
    assert not final_model.parity_report(probs, probs + 1e-3)["passed"]
    swapped = probs.copy()
    swapped[0] = [0.1, 0.7, 0.1, 0.1]
    report = final_model.parity_report(probs, swapped)
    assert not report["passed"] and report["same_class_share"] == pytest.approx(24 / 25)
    with pytest.raises(RuntimeError):
        final_model.check_parity(report, "valid")


def test_bootstrap_interval_contains_estimate_and_is_reproducible():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(4), 30)
    probabilities = rng.random((120, 4))
    probabilities[np.arange(120), labels] += 0.6
    first = final_model.bootstrap_ci(probabilities, labels, draws=500)
    point = final_model.metrics.classification_metrics(probabilities, labels)["macro_f1"]
    assert first["macro_f1"][0] <= point <= first["macro_f1"][1]
    assert first == final_model.bootstrap_ci(probabilities, labels, draws=500)
    perfect = final_model.bootstrap_ci(np.eye(4)[labels], labels, draws=200)
    assert perfect["accuracy"] == [1.0, 1.0] and perfect["per_class_f1"]["rainy"] == [1.0, 1.0]


def test_dataset_fingerprint_changes_when_an_image_changes(tmp_path):
    data_dir = write_tiny_dataset(tmp_path / "data")
    before = final_model.dataset_fingerprint(data_dir)
    assert set(before) == {"train", "valid"} and before["valid"]["n_images"] == 12
    assert final_model.dataset_fingerprint(data_dir) == before
    target = next((data_dir / "valid" / "rainy").iterdir())
    target.write_bytes(target.read_bytes() + b"0")
    assert final_model.dataset_fingerprint(data_dir)["valid"]["sha256"] != before["valid"]["sha256"]


def test_run_final_writes_files_picks_best_seed_and_touches_test_once(final_ctx):
    cfg, ctx = final_ctx
    record = final_model.run_final(cfg, ctx)
    paths = final_model.output_paths(ctx["dir"])
    assert all(paths[key].exists() for key in final_model.INSTALLED_FILES + ["test_touch", "test_predictions"])
    assert record["run_id"] == "s7_r0_patience30_min20_seed22" and record["seed"] == 22
    assert {"validation", "test", "test_bootstrap_95ci", "dataset", "parity"} <= set(record)
    assert record["test"]["split"] == "test" and record["parity"]["test"]["n_images"] == 24
    assert json.loads(paths["class_names"].read_text()) == ["cloudy", "rainy", "shiny", "sunrise"]
    assert json.loads(paths["test_touch"].read_text())["split"] == "test"
    assert not pd.read_csv(paths["test_predictions"])["path"].str.contains(":").any()
    with pytest.raises(RuntimeError, match="already evaluated"):
        final_model.run_final(cfg, ctx)


def test_failed_validation_parity_stops_before_the_test_set(final_ctx, monkeypatch):
    cfg, ctx = final_ctx
    monkeypatch.setattr(final_model, "load_run_model", lambda run_ctx, run_id: OnnxBackedModel(None, noise=0.01))
    with pytest.raises(RuntimeError, match="parity failed on valid"):
        final_model.run_final(cfg, ctx)
    assert not final_model.output_paths(ctx["dir"])["test_touch"].exists()


def test_install_checks_record_and_onnx_then_copies_files(final_ctx, tmp_path):
    cfg, ctx = final_ctx
    final_model.run_final(cfg, ctx)
    result = final_model.install_final_model(ctx["dir"], tmp_path / "models", ctx["data_dir"])
    assert sorted(path.split("models")[-1][1:] for path in result["installed"]) == [
        "class_names.json", "metrics.json", "weather_cnn.h5", "weather_cnn.onnx"]
    metrics_path = final_model.output_paths(ctx["dir"])["metrics"]
    record = json.loads(metrics_path.read_text())
    metrics_path.write_text(json.dumps({**record, "validation": {**record["validation"], "macro_f1": 0.0}}))
    with pytest.raises(ValueError, match="local ONNX"):
        final_model.install_final_model(ctx["dir"], tmp_path / "models2", ctx["data_dir"])
    metrics_path.write_text(json.dumps({key: value for key, value in record.items() if key != "test"}))
    with pytest.raises(ValueError, match="test"):
        final_model.install_final_model(ctx["dir"], tmp_path / "models3", ctx["data_dir"])


def test_install_refuses_an_onnx_file_the_test_set_never_saw(final_ctx, tmp_path):
    cfg, ctx = final_ctx
    final_model.run_final(cfg, ctx)
    onnx_path = final_model.output_paths(ctx["dir"])["onnx"]
    onnx_path.write_bytes(onnx_path.read_bytes() + b"0")
    with pytest.raises(ValueError, match="not the one"):
        final_model.install_final_model(ctx["dir"], tmp_path / "models", ctx["data_dir"])


def test_section7_group_trains_the_selected_config():
    spec = grids.section_groups(7)["final"]
    assert spec["final_export"] and spec["seeds"] == grids.config.SEEDS
    assert experiments.config_hash(spec["configs"][0]) == experiments.config_hash(grids.stopping_rule_configs()[0])
    assert [cfg["name"] for cfg in grids.smoke_configs(7)] == ["smoke_final"]
    assert experiments.group_order(7) == ["smoke", "timing", "final"]


def test_final_config_matches_the_selection_on_saved_results():
    path = experiments.results_path(grids.config.EXPERIMENTS_DIR)
    results = pd.read_csv(path) if path.exists() else pd.DataFrame({"section": []})
    if 5 not in set(results["section"]):
        pytest.skip("no Section 5 results in this checkout")
    chosen = selection.select_config(selection.config_summary(results[results["section"].isin([3, 5])]))
    assert chosen["cfg_name"] == grids.final_config()["name"]
    stored = results.loc[results["cfg_name"] == chosen["cfg_name"], "config_hash"].iloc[0]
    assert stored == experiments.config_hash(grids.final_config())
