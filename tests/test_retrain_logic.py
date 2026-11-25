import shutil
import threading

import onnx
import pytest

from src import config, database, placeholder, prediction, retrain, uploads


def fake_metrics(macro_f1):
    per_class = {name: {"precision": macro_f1, "recall": macro_f1, "f1": macro_f1, "support": 3} for name in config.CLASS_NAMES}
    return {"macro_f1": macro_f1, "accuracy": macro_f1, "per_class": per_class}


def fake_training(monkeypatch, scores=None):
    monkeypatch.setattr(retrain, "ensure_base_model", lambda: config.H5_PATH)
    monkeypatch.setattr(retrain, "fine_tune_model", lambda data, cfg, history: "trained")

    def export(model, paths, run_id):
        paths["h5"].write_bytes(b"fake keras weights")
        onnx.save(placeholder.build_placeholder_onnx(seed=run_id + 100), str(paths["onnx"]))
        return "fake"

    monkeypatch.setattr(retrain, "export_candidate", export)
    if scores:
        monkeypatch.setattr(retrain, "evaluate_model_file",
                            lambda path, data: fake_metrics(scores["candidate" if path.name == "candidate.onnx" else "current"]))


def run_and_wait(runner=None):
    return database.get_run(retrain.start_retraining("test", runner, wait=True))


@pytest.mark.parametrize("unused, threshold, expected", [(39, 40, False), (40, 40, True), (0, 0, True)])
def test_should_retrain(unused, threshold, expected):
    assert retrain.should_retrain(unused, threshold) is expected


@pytest.mark.parametrize("candidate, current, expected", [(0.80, 0.80, True), (0.81, 0.80, True), (0.79, 0.80, False),
                                                          (0.10, None, True)])
def test_should_promote_requires_match_or_better(candidate, current, expected):
    assert retrain.should_promote(candidate, current, tolerance=0.0) is expected


def test_second_job_is_refused_while_one_runs(service_env):
    release, started = threading.Event(), threading.Event()

    def slow_runner(run_id):
        started.set()
        release.wait(5)

    run_id = retrain.start_retraining("manual", slow_runner)
    started.wait(5)
    with pytest.raises(retrain.RetrainRunning):
        retrain.start_retraining("manual", slow_runner)
    assert retrain.get_status()["running"] and retrain.get_status()["current"]["run_id"] == run_id
    release.set()
    retrain.JOB_LOCK.acquire(timeout=5)
    retrain.JOB_LOCK.release()


def test_failed_run_is_logged(service_env):
    def broken(run_id):
        raise RuntimeError("disk on fire")

    run = run_and_wait(broken)
    assert run["status"] == "failed" and "disk on fire" in run["message"] and not retrain.JOB_LOCK.locked()


def test_auto_trigger_fires_at_threshold(service_env, make_jpeg, monkeypatch):
    monkeypatch.setattr(config, "RETRAIN_AUTO", True)
    monkeypatch.setattr(config, "RETRAIN_UPLOAD_THRESHOLD", 3)
    calls = []
    uploads.save_uploads([("a.jpg", make_jpeg(1)), ("b.jpg", make_jpeg(2))], "rainy")
    assert retrain.maybe_auto_retrain(calls.append) is None
    uploads.save_uploads([("c.jpg", make_jpeg(3))], "rainy")
    run_id = retrain.maybe_auto_retrain(calls.append)
    retrain.JOB_LOCK.acquire(timeout=5)
    retrain.JOB_LOCK.release()
    assert run_id is not None and calls == [run_id] and database.get_run(run_id)["trigger"] == "auto"


def test_retraining_data_merges_uploads_and_rejects_unreadable(service_env, make_jpeg):
    uploads.save_uploads([("a.jpg", make_jpeg(1)), ("b.jpg", make_jpeg(2))], "rainy")
    uploads.upload_file_path(database.unused_uploads()[1]).unlink()
    data = retrain.prepare_retraining_data(size=32)
    assert len(data["train_labels"]) == data["n_base"] + 1 and len(data["upload_ids"]) == 1
    assert len(data["rejected_ids"]) == 1 and data["train_images"].shape[1:] == (32, 32, 3)


def test_full_cycle_promotes_archives_and_hot_reloads(service_env, make_jpeg, monkeypatch):
    fake_training(monkeypatch, {"current": 0.50, "candidate": 0.55})
    uploads.save_uploads([("a.jpg", make_jpeg(1))], "rainy")
    before = prediction.model_version()["sha256"]
    run = run_and_wait()
    assert run["status"] == "completed" and run["promoted"] is True and run["candidate_macro_f1"] == 0.55
    assert prediction.model_version()["sha256"] != before
    assert config.H5_PATH.read_bytes() == b"fake keras weights"
    assert prediction.load_model_metrics()["validation"]["macro_f1"] == 0.55
    archived = list(config.ARCHIVE_DIR.glob("*_before_run*/placeholder_model.onnx"))
    assert len(archived) == 1 and database.count_unused_uploads() == 0
    assert run["details"]["current"]["per_class_f1"]["rainy"] == 0.5


def test_worse_candidate_is_rejected_and_model_kept(service_env, monkeypatch):
    fake_training(monkeypatch, {"current": 0.60, "candidate": 0.40})
    before = prediction.model_version()["sha256"]
    run = run_and_wait()
    assert run["status"] == "completed" and run["promoted"] is False
    assert prediction.model_version()["sha256"] == before and not config.ARCHIVE_DIR.exists()
    assert not list(config.CANDIDATE_DIR.glob("run_*"))


def test_real_evaluation_on_validation_set(service_env, monkeypatch):
    fake_training(monkeypatch)
    run = run_and_wait()
    assert run["status"] == "completed", run["message"]
    assert run["promoted"] == retrain.should_promote(run["candidate_macro_f1"], run["current_macro_f1"])


def test_empty_base_data_fails_with_a_clear_message(service_env, tmp_path, monkeypatch):
    fake_training(monkeypatch)
    empty = tmp_path / "fresh_clone_data"
    monkeypatch.setattr(config, "DATA_DIR", empty)
    report = retrain.base_data_report()
    assert not report["ok"] and report["train"]["n_images"] == 0
    assert f"base training data not found at {empty / 'train'}, 0 images" in report["problems"]
    run = run_and_wait()
    assert run["status"] == "failed" and run["message"].startswith("BaseDataMissing: base training data not found at")
    assert "stack" not in run["message"] and not retrain.JOB_LOCK.locked()


def test_a_class_without_images_is_reported(service_env):
    shutil.rmtree(config.DATA_DIR / "valid" / "rainy")
    report = retrain.base_data_report()
    assert report["ok"] is False and report["problems"] == [
        f"base validation data at {config.DATA_DIR / 'valid'} has no images for rainy"]
    with pytest.raises(retrain.BaseDataMissing, match="no images for rainy"):
        retrain.prepare_retraining_data(size=32)


def test_auto_retrain_does_not_start_without_base_data(service_env, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RETRAIN_AUTO", True)
    monkeypatch.setattr(config, "RETRAIN_UPLOAD_THRESHOLD", 0)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "empty")
    assert retrain.maybe_auto_retrain(lambda run_id: None) is None and database.list_runs(5) == []
