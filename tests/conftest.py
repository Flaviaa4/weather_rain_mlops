import io

import numpy as np
import pytest
from PIL import Image

from src import config


def gradient(width=300, height=168):
    row = np.linspace(0, 255, width, dtype=np.float32)
    gray = np.tile(row, (height, 1))
    return Image.fromarray(np.stack([gray, gray * 0.7, gray * 0.3], axis=-1).astype(np.uint8))


def tinted_gradient(width, height, scale, offset):
    array = np.asarray(gradient(width, height), dtype=np.float32) * scale + offset
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def write_tiny_dataset(root, per_class=3):
    for split in ["train", "valid"]:
        for index, name in enumerate(config.CLASS_NAMES):
            folder = root / split / name
            folder.mkdir(parents=True)
            for i in range(per_class):
                tinted_gradient(200 + 37 * i + index, 120 + 11 * i, 0.4 + 0.2 * i, 25 * index).save(folder / f"{name}{i}.jpg", quality=90)
    return root


@pytest.fixture
def tiny_dataset(tmp_path):
    return write_tiny_dataset(tmp_path / "data")


# a fixture, not an import: Kaggle has an unrelated "tests" package that shadows "from tests.conftest import"
@pytest.fixture
def make_tiny_dataset():
    return write_tiny_dataset


@pytest.fixture
def gradient_image():
    return gradient()


def jpeg_bytes(seed=0, size=(96, 64)):
    rng = np.random.default_rng(seed)
    buffer = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


@pytest.fixture
def make_jpeg():
    return jpeg_bytes


def point_config_at(monkeypatch, data_dir, models_dir):
    paths = {"DATA_DIR": data_dir, "UPLOADS_DIR": data_dir / "uploads", "DB_PATH": data_dir / "weather.db",
             "MODELS_DIR": models_dir, "ONNX_PATH": models_dir / "placeholder_model.onnx",
             "H5_PATH": models_dir / "placeholder_model.h5", "METRICS_PATH": models_dir / "placeholder_model_metrics.json",
             "CLASS_NAMES_PATH": models_dir / "class_names.json", "ARCHIVE_DIR": models_dir / "archive",
             "CANDIDATE_DIR": models_dir / "candidate", "INSIGHTS_CACHE": data_dir / "cache" / "features.csv",
             "EDA_FEATURES_CSV": data_dir / "no_eda_features.csv"}
    for name, value in paths.items():
        monkeypatch.setattr(config, name, value)
    monkeypatch.setattr(config, "MIN_FREE_DISK_MB", 0)
    monkeypatch.setattr(config, "RETRAIN_AUTO", False)
    monkeypatch.setenv("BOOTSTRAP_PLACEHOLDER", "0")
    return paths


@pytest.fixture
def service_env(tiny_dataset, tmp_path, monkeypatch):
    from src import database, placeholder, prediction
    paths = point_config_at(monkeypatch, tiny_dataset, tmp_path / "models")
    placeholder.save_placeholder_onnx(paths["ONNX_PATH"])
    database.init_db()
    prediction.SESSION_CACHE.clear()
    yield paths
    prediction.SESSION_CACHE.clear()
