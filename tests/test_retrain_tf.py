import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("tf2onnx")

from src import config, database, placeholder, prediction, retrain, uploads


def test_real_fine_tune_cycle_from_placeholder(service_env, make_jpeg, monkeypatch):
    monkeypatch.setattr(config, "RETRAIN_MAX_EPOCHS", 1)
    placeholder.build_placeholder_keras(config.H5_PATH, config.ONNX_PATH)
    uploads.save_uploads([("a.jpg", make_jpeg(1)), ("b.jpg", make_jpeg(2))], "rainy")
    run = database.get_run(retrain.start_retraining("test", wait=True))
    assert run["status"] == "completed", run["message"]
    assert run["n_uploads"] == 2 and run["details"]["export_method"] in {"from_keras", "saved_model_cli"}
    assert run["promoted"] == retrain.should_promote(run["candidate_macro_f1"], run["current_macro_f1"])
    assert prediction.predict_image_bytes(make_jpeg(5), "x.jpg")["label"] in config.CLASS_NAMES
