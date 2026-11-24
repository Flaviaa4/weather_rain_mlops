import io
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from src import insights, preprocessing


def image_bytes(image, fmt="JPEG", **kwargs):
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


def test_inference_path_does_not_import_tensorflow():
    code = "import sys, src.preprocessing, src.insights; print('tensorflow' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


def test_is_image_file():
    assert preprocessing.is_image_file("a/B.JPG") and not preprocessing.is_image_file("a/notes.txt")


def test_is_valid_image_bytes(gradient_image):
    assert preprocessing.is_valid_image_bytes(image_bytes(gradient_image))
    assert not preprocessing.is_valid_image_bytes(b"not an image at all")
    assert not preprocessing.is_valid_image_bytes(image_bytes(gradient_image)[:200])


@pytest.mark.parametrize("mode", ["L", "CMYK", "RGBA", "P"])
def test_preprocess_handles_modes(mode, gradient_image):
    fmt = "PNG" if mode in ("RGBA", "P") else "JPEG"
    array = preprocessing.preprocess_image_bytes(image_bytes(gradient_image.convert(mode), fmt), size=64)
    assert array.shape == (64, 64, 3) and array.dtype == np.float32
    assert 0.0 <= array.min() and array.max() <= 1.0


def test_transparent_pixels_become_white():
    image = Image.new("RGBA", (40, 30), (0, 0, 0, 0))
    array = preprocessing.preprocess_image_bytes(image_bytes(image, "PNG"), size=16)
    assert np.allclose(array, 1.0)


def test_exif_orientation_is_applied(gradient_image):
    exif = Image.Exif()
    exif[274] = 6
    data = image_bytes(gradient_image, exif=exif)
    upright = preprocessing.to_rgb(preprocessing.open_image_bytes(data))
    assert upright.size == (168, 300)


def test_normalize_and_batch_dimension():
    array = preprocessing.normalize_array(np.array([[[0, 255, 51]]], dtype=np.uint8))
    assert np.allclose(array, [[[0.0, 1.0, 0.2]]])
    assert preprocessing.add_batch_dimension(array).shape == (1, 1, 1, 3)


def test_list_labeled_files_is_sorted_and_labeled(tiny_dataset):
    root = tiny_dataset
    files = preprocessing.list_labeled_files(root / "train")
    assert [label for _, label in files] == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert files == sorted(files, key=lambda item: (item[1], str(item[0])))


def test_training_loader_equals_inference_path(tiny_dataset):
    root = tiny_dataset
    images, labels, paths = preprocessing.load_split("train", size=64, data_dir=root)
    for image, path in zip(images, paths):
        with open(path, "rb") as handle:
            inference = preprocessing.preprocess_image_bytes(handle.read(), size=64)
        assert np.array_equal(image, inference)
    assert images.shape == (12, 64, 64, 3) and labels.dtype == np.int32


def test_unknown_augmentation_strength_fails():
    with pytest.raises(ValueError):
        preprocessing.augment_params("extreme")


def test_features_on_known_images():
    gray = np.full((32, 32, 3), 0.5, dtype=np.float32)
    blue = np.zeros((32, 32, 3), dtype=np.float32)
    blue[..., 2] = 1.0
    assert insights.brightness(gray) == pytest.approx(0.5) and insights.contrast(gray) == pytest.approx(0.0)
    assert insights.mean_saturation(gray) == pytest.approx(0.0) and insights.mean_saturation(blue) == pytest.approx(1.0)
    assert insights.blue_red_ratio(blue) > 50 and insights.sharpness(gray) == pytest.approx(0.0)


def test_rank_auc_extremes():
    scores = np.array([1, 2, 3, 4])
    assert insights.rank_auc(scores, np.array([False, False, True, True])) == 1.0
    assert insights.rank_auc(scores, np.array([True, True, False, False])) == 0.0


def test_describe_feature_mentions_extremes(tiny_dataset):
    frame = insights.extract_features(tiny_dataset, ["train"], size=32)
    text = insights.describe_feature(frame, "brightness")
    assert "highest for" in text and "AUC" in text
