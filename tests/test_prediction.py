import subprocess
import sys
import threading

import numpy as np
import onnx
import pytest

from src import placeholder, prediction, retrain
from src.preprocessing import add_batch_dimension, preprocess_image_bytes


def test_placeholder_model_is_marked_and_valid(service_env):
    model = onnx.load(str(service_env["ONNX_PATH"]))
    props = {prop.key: prop.value for prop in model.metadata_props}
    assert props["placeholder"] == "true" and "PLACEHOLDER" in props["warning"]
    version = prediction.model_version()
    assert version["is_placeholder"] and version["model_name"] == "placeholder_model"
    session = prediction.get_session()
    assert session.get_inputs()[0].name == "image" and session.get_outputs()[0].shape[1] == 4


def test_prediction_format(service_env, make_jpeg):
    result = prediction.predict_image_bytes(make_jpeg(), "photo.jpg")
    assert result["label"] in result["probabilities"] and len(result["probabilities"]) == 4
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert result["confidence"] == max(result["probabilities"].values())
    assert result["uncertain"] == (result["confidence"] < 0.6)


def test_prediction_uses_training_preprocessing(service_env, make_jpeg):
    data = make_jpeg(3)
    expected = prediction.run_inference(prediction.get_session(), add_batch_dimension(preprocess_image_bytes(data)))[0]
    result = prediction.predict_image_bytes(data, "x.png")
    assert [result["probabilities"][name] for name in result["probabilities"]] == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("data, filename, status", [
    (b"", "a.jpg", 400),
    (b"plain text, not an image", "notes.txt", 415),
    (b"\xff\xd8\xff\xe0 truncated jpeg", "broken.jpg", 415),
])
def test_validation_errors(data, filename, status):
    with pytest.raises(prediction.InvalidImageError) as error:
        prediction.validate_image_bytes(data, filename)
    assert error.value.status_code == status


def test_oversized_file_is_refused(make_jpeg):
    with pytest.raises(prediction.InvalidImageError) as error:
        prediction.validate_image_bytes(make_jpeg() + b"\0" * 2_000_000, "big.jpg", max_mb=1)
    assert error.value.status_code == 413


def test_logits_become_probabilities():
    probabilities = prediction.ensure_probabilities(np.array([[2.0, 1.0, 0.0, -1.0]]))
    assert probabilities.sum() == pytest.approx(1.0) and probabilities.argmax() == 0


def test_hot_reload_without_failed_requests(service_env, make_jpeg):
    data, errors, stop = make_jpeg(), [], threading.Event()
    before = prediction.model_version()["sha256"]

    def keep_predicting():
        while not stop.is_set():
            try:
                prediction.predict_image_bytes(data, "x.jpg")
            except Exception as error:
                errors.append(error)

    workers = [threading.Thread(target=keep_predicting) for _ in range(4)]
    [worker.start() for worker in workers]
    try:
        for seed in range(1, 6):
            replacement = service_env["ONNX_PATH"].with_name("incoming.onnx")
            onnx.save(placeholder.build_placeholder_onnx(seed=seed), str(replacement))
            retrain.atomic_replace(replacement, service_env["ONNX_PATH"])
    finally:
        stop.set()
        [worker.join() for worker in workers]
    assert errors == [] and prediction.model_version()["sha256"] != before


def test_prediction_and_api_do_not_import_tensorflow():
    code = "import sys, src.prediction, app.api; print('tensorflow' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"
