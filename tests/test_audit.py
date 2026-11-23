import numpy as np
import pandas as pd
from PIL import Image

from src import audit


def write_image(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path, quality=95)
    return path


def gradient_image(width, height, flip=False):
    row = np.linspace(0, 255, width)
    row = row[::-1] if flip else row
    gray = np.tile(row, (height, 1))
    return np.stack([gray, gray * 0.8, gray * 0.5], axis=-1)


def test_original_stem_strips_roboflow_suffix():
    assert audit.original_stem("rain1_jpg.rf.99c3f88f7e8b049da1086f0107e1a779.jpg") == "rain1_jpg"


def test_dhash_identical_for_resized_copy(tmp_path):
    a = write_image(tmp_path / "a.jpg", gradient_image(300, 168))
    b = write_image(tmp_path / "b.jpg", gradient_image(600, 336))
    bits = audit.hashes_to_bit_matrix([audit.compute_dhash(a, 8), audit.compute_dhash(b, 8)], 64)
    assert audit.pairwise_hamming(bits)[0, 1] <= 2


def test_dhash_far_for_flipped_gradient(tmp_path):
    a = write_image(tmp_path / "a.jpg", gradient_image(300, 168))
    b = write_image(tmp_path / "b.jpg", gradient_image(300, 168, flip=True))
    bits = audit.hashes_to_bit_matrix([audit.compute_dhash(a, 8), audit.compute_dhash(b, 8)], 64)
    assert audit.pairwise_hamming(bits)[0, 1] > 50


def test_corrupt_file_is_reported(tmp_path):
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    assert audit.check_image_integrity(broken) != ""


def test_groups_merge_transitive_pairs():
    pairs = pd.DataFrame({"path_a": ["train/x/a", "valid/x/b"], "path_b": ["valid/x/b", "test/x/c"]})
    groups = audit.build_duplicate_groups(pairs)
    assert groups == [["test/x/c", "train/x/a", "valid/x/b"]]


def test_keeper_prefers_train_then_valid():
    members = ["test/x/c", "valid/x/b", "train/x/a"]
    assert min(members, key=audit.keeper_sort_key) == "train/x/a"
    assert min(members[:2], key=audit.keeper_sort_key) == "valid/x/b"


def test_reviewed_exception_matches_either_order():
    exceptions = [{"file_a": "sunrise193_jpg.rf.", "file_b": "sunrise291_jpg.rf."}]
    assert audit.is_reviewed_exception("t/s/sunrise291_jpg.rf.1.jpg", "t/s/sunrise193_jpg.rf.2.jpg", exceptions)
    assert not audit.is_reviewed_exception("t/s/sunrise19_jpg.rf.1.jpg", "t/s/sunrise291_jpg.rf.2.jpg", exceptions)


def test_confirmation_rules():
    pairs = pd.DataFrame({
        "path_a": ["a1", "a2", "a3", "a4"], "path_b": ["b1", "b2", "b3", "b4"],
        "hamming": [2, 3, 12, 12], "thumb_mae": [0.01, 0.30, 0.05, 0.30], "gray_corr": [0.99, 0.60, 0.95, 0.99],
    })
    confirmed = audit.mark_confirmed(pairs)["confirmed"].tolist()
    assert confirmed == [True, False, True, False]
