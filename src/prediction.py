import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort

from src import config, metrics
from src.preprocessing import add_batch_dimension, is_image_file, is_valid_image_bytes, preprocess_image_bytes

LOGGER = logging.getLogger("prediction")
SESSION_LOCK = threading.Lock()
SESSION_CACHE = {}


class InvalidImageError(ValueError):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code


def load_class_names(path=None):
    path = Path(path or config.CLASS_NAMES_PATH)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else list(config.CLASS_NAMES)


def session_options(threads=config.ONNX_THREADS):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return options


def load_onnx_session(path, threads=config.ONNX_THREADS):
    model_bytes = Path(path).read_bytes()
    session = ort.InferenceSession(model_bytes, sess_options=session_options(threads), providers=["CPUExecutionProvider"])
    return session, hashlib.sha256(model_bytes).hexdigest()[:12]


def file_signature(path):
    stat = Path(path).stat()
    return stat.st_mtime_ns, stat.st_size, stat.st_ino


def cached_entry(path):
    entry = SESSION_CACHE.get(str(path))
    return entry if entry and entry["signature"] == file_signature(path) else None


def reload_session(path, threads):
    signature = file_signature(path)
    session, digest = load_onnx_session(path, threads)
    entry = {"signature": signature, "session": session, "sha256": digest,
             "loaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    SESSION_CACHE[str(path)] = entry
    LOGGER.info("Loaded model %s (sha256 %s)", path, digest)
    return entry


def get_session_entry(path=None, threads=config.ONNX_THREADS):
    path = Path(path or config.ONNX_PATH)
    entry = cached_entry(path)
    if entry:
        return entry
    with SESSION_LOCK:
        entry = cached_entry(path)
        if entry:
            return entry
        try:
            return reload_session(path, threads)
        except Exception:
            if str(path) in SESSION_CACHE:
                LOGGER.exception("Reload of %s failed, keeping the previous model", path)
                return SESSION_CACHE[str(path)]
            raise


def get_session(path=None):
    return get_session_entry(path)["session"]


def drop_session(path):
    with SESSION_LOCK:
        SESSION_CACHE.pop(str(Path(path)), None)


def file_digest(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.exists() else None


def run_inference(session, batch):
    feed = {session.get_inputs()[0].name: batch.astype(np.float32)}
    return session.run(None, feed)[0]


def softmax(values):
    shifted = np.exp(values - values.max(axis=1, keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)


def ensure_probabilities(outputs):
    outputs = np.asarray(outputs, dtype=np.float64)
    is_distribution = outputs.min() >= 0 and np.allclose(outputs.sum(axis=1), 1.0, atol=1e-3)
    return outputs if is_distribution else softmax(outputs)


def is_uncertain(confidence, threshold=config.UNCERTAIN_THRESHOLD):
    return bool(confidence < threshold)


def format_prediction(probabilities, class_names):
    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index])
    return {"label": class_names[index], "confidence": round(confidence, 4),
            "probabilities": {name: round(float(p), 4) for name, p in zip(class_names, probabilities)},
            "uncertain": is_uncertain(confidence)}


def check_not_empty(data):
    if not data:
        raise InvalidImageError("The file is empty.")


def check_size(data, max_mb=config.MAX_UPLOAD_MB):
    if len(data) > max_mb * 1024 * 1024:
        raise InvalidImageError(f"The file is larger than {max_mb} MB.", status_code=413)


def check_extension(filename):
    if filename and Path(filename).suffix and not is_image_file(filename):
        raise InvalidImageError(f"Unsupported file type: {Path(filename).suffix or 'none'}. Use JPG, PNG, BMP or WEBP.",
                                status_code=415)


def check_decodable(data):
    if not is_valid_image_bytes(data):
        raise InvalidImageError("The file is not a readable image.", status_code=415)


def validate_image_bytes(data, filename=None, max_mb=config.MAX_UPLOAD_MB):
    check_not_empty(data)
    check_size(data, max_mb)
    check_extension(filename)
    check_decodable(data)
    return data


def predict_array(array, path=None):
    outputs = run_inference(get_session(path), add_batch_dimension(array))
    return format_prediction(ensure_probabilities(outputs)[0], load_class_names())


def predict_image_bytes(data, filename=None, path=None):
    validate_image_bytes(data, filename)
    return predict_array(preprocess_image_bytes(data), path)


def predict_probabilities(images, path=None, batch_size=config.BATCH_SIZE):
    session = get_session(path)
    batches = [run_inference(session, images[start:start + batch_size]) for start in range(0, len(images), batch_size)]
    return ensure_probabilities(np.concatenate(batches))


def evaluate_onnx_model(images, labels, path=None):
    return metrics.classification_metrics(predict_probabilities(images, path), labels)


def model_metadata(session):
    return dict(session.get_modelmeta().custom_metadata_map)


def model_version(path=None):
    path = Path(path or config.ONNX_PATH)
    entry = get_session_entry(path)
    metadata = model_metadata(entry["session"])
    stat = Path(path).stat()
    return {"model_name": Path(path).stem, "file": Path(path).name, "sha256": entry["sha256"],
            "size_kb": round(stat.st_size / 1024, 1), "loaded_at": entry["loaded_at"],
            "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
            "is_placeholder": metadata.get("placeholder") == "true" or Path(path).stem.startswith("placeholder"),
            "metadata": metadata}


def load_model_metrics(path=None):
    path = Path(path or config.METRICS_PATH)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
