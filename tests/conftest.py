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


@pytest.fixture
def gradient_image():
    return gradient()
