import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src import config, preprocessing


def collect(dataset):
    images, labels = zip(*[(x.numpy(), y.numpy()) for x, y in dataset])
    return np.concatenate(images), np.concatenate(labels)


def test_eval_dataset_equals_inference_path(tiny_dataset):
    root = tiny_dataset
    images, labels, paths = preprocessing.load_split("train", size=64, data_dir=root)
    batched, batched_labels = collect(preprocessing.build_dataset(images, labels, batch_size=5))
    for image, path in zip(batched, paths):
        with open(path, "rb") as handle:
            assert np.array_equal(image, preprocessing.preprocess_image_bytes(handle.read(), size=64))
    assert np.array_equal(batched_labels, labels)


def test_training_without_augmentation_only_reorders(tiny_dataset):
    images, labels, _ = preprocessing.load_split("train", size=32, data_dir=tiny_dataset)
    shuffled, shuffled_labels = collect(preprocessing.build_dataset(images, labels, 4, training=True, seed=11))
    order = [int(np.argmin(np.abs(images - image).sum(axis=(1, 2, 3)))) for image in shuffled]
    assert sorted(order) == list(range(len(images))) and np.array_equal(labels[order], shuffled_labels)


@pytest.mark.parametrize("strength", ["light", "medium", "strong"])
def test_augmentation_keeps_shape_range_and_is_seeded(strength):
    image = tf.constant(np.random.default_rng(0).random((64, 64, 3), dtype=np.float32))
    seed = tf.constant([11, 1], tf.int64)
    first = preprocessing.augment_image(image, seed, strength).numpy()
    again = preprocessing.augment_image(image, seed, strength).numpy()
    assert first.shape == (64, 64, 3) and first.min() >= 0.0 and first.max() <= 1.0
    assert np.allclose(first, again, atol=1e-5)
    other = preprocessing.augment_image(image, tf.constant([22, 1], tf.int64), strength).numpy()
    assert not np.allclose(first, other, atol=1e-5)


def test_augmented_dataset_changes_each_epoch(tiny_dataset):
    images, labels, _ = preprocessing.load_split("train", size=32, data_dir=tiny_dataset)
    dataset = preprocessing.build_dataset(images, labels, 12, training=True, strength="medium", seed=11)
    epoch_one, _ = collect(dataset)
    epoch_two, _ = collect(dataset)
    assert not np.allclose(np.sort(epoch_one.ravel()), np.sort(epoch_two.ravel()))
    assert config.AUGMENT_STRENGTH in config.AUGMENT_STRENGTHS
