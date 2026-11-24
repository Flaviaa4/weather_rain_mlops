import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from src import config


def is_image_file(path):
    return Path(path).suffix.lower() in config.IMAGE_EXTENSIONS


def is_valid_image_bytes(data):
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        return True
    except Exception:
        return False


def open_image_bytes(data):
    image = Image.open(io.BytesIO(data))
    image.load()
    return image


def has_transparency(image):
    return image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info)


def flatten_on_white(image):
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba)


def to_rgb(image):
    image = ImageOps.exif_transpose(image)
    if has_transparency(image):
        image = flatten_on_white(image)
    return image.convert("RGB")


def resize_image(image, size=config.IMAGE_SIZE):
    return image.resize((size, size), Image.Resampling.BILINEAR)


def to_array(image):
    return np.asarray(image, dtype=np.uint8)


def normalize_array(array):
    return array.astype(np.float32) / 255.0


def preprocess_image(image, size=config.IMAGE_SIZE):
    return normalize_array(to_array(resize_image(to_rgb(image), size)))


def preprocess_image_bytes(data, size=config.IMAGE_SIZE):
    return preprocess_image(open_image_bytes(data), size)


def preprocess_image_file(path, size=config.IMAGE_SIZE):
    return preprocess_image_bytes(Path(path).read_bytes(), size)


def add_batch_dimension(array):
    return np.expand_dims(array, axis=0)


def list_labeled_files(split_dir, class_names=config.CLASS_NAMES):
    files = []
    for label, class_name in enumerate(class_names):
        class_dir = Path(split_dir) / class_name
        paths = sorted(p for p in class_dir.iterdir() if p.is_file() and is_image_file(p)) if class_dir.exists() else []
        files.extend((path, label) for path in paths)
    return files


def load_split(split, size=config.IMAGE_SIZE, data_dir=config.DATA_DIR, class_names=config.CLASS_NAMES):
    files = list_labeled_files(Path(data_dir) / split, class_names)
    images = np.stack([preprocess_image_file(path, size) for path, _ in files])
    labels = np.array([label for _, label in files], dtype=np.int32)
    return images, labels, [str(path) for path, _ in files]


def import_tensorflow():
    import tensorflow as tf
    return tf


def augment_params(strength):
    if strength not in config.AUGMENT_STRENGTHS:
        raise ValueError(f"unknown augmentation strength: {strength}")
    return config.AUGMENT_STRENGTHS[strength]


def random_flip(image, seed, params):
    tf = import_tensorflow()
    return tf.image.stateless_random_flip_left_right(image, seed) if params["flip"] else image


def random_zoom(image, seed, params):
    tf = import_tensorflow()
    if params["zoom"] <= 0:
        return image
    height, width = image.shape[0], image.shape[1]
    scale = tf.random.stateless_uniform([], seed, 1.0 - params["zoom"], 1.0)
    crop = tf.cast(tf.round(tf.cast([height, width], tf.float32) * scale), tf.int32)
    cropped = tf.image.stateless_random_crop(image, tf.concat([crop, [3]], 0), seed)
    return tf.image.resize(cropped, [height, width], method="bilinear")


def random_light(image, seeds, params):
    tf = import_tensorflow()
    image = tf.image.stateless_random_brightness(image, params["brightness"], seeds[0])
    low, high = 1.0 - params["contrast"], 1.0 + params["contrast"]
    return tf.image.stateless_random_contrast(image, low, high, seeds[1])


def random_color(image, seeds, params):
    tf = import_tensorflow()
    low, high = 1.0 - params["saturation"], 1.0 + params["saturation"]
    image = tf.image.stateless_random_saturation(image, low, high, seeds[0])
    return tf.image.stateless_random_hue(image, params["hue"], seeds[1]) if params["hue"] > 0 else image


def augment_image(image, seed, strength=config.AUGMENT_STRENGTH):
    tf = import_tensorflow()
    params = augment_params(strength)
    if params is None:
        return image
    seeds = tf.random.experimental.stateless_split(tf.cast(seed, tf.int64), num=6)
    image = random_zoom(random_flip(image, seeds[0], params), seeds[1], params)
    image = random_color(random_light(image, seeds[2:4], params), seeds[4:6], params)
    return tf.clip_by_value(image, 0.0, 1.0)


def make_augment_pair(strength):
    def augment_pair(pair, op_seed):
        return augment_image(pair[0], op_seed, strength), pair[1]
    return augment_pair


def add_augmentation(dataset, strength, seed):
    tf = import_tensorflow()
    seeds = tf.data.Dataset.random(seed=seed, rerandomize_each_iteration=True).batch(2, drop_remainder=True)
    paired = tf.data.Dataset.zip((dataset, seeds))
    return paired.map(make_augment_pair(strength), num_parallel_calls=tf.data.AUTOTUNE)


def build_dataset(images, labels, batch_size=config.BATCH_SIZE, training=False, strength="none", seed=0):
    tf = import_tensorflow()
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        dataset = dataset.shuffle(len(images), seed=seed, reshuffle_each_iteration=True)
    if training and augment_params(strength) is not None:
        dataset = add_augmentation(dataset, strength, seed)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
