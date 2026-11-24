from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src import config
from src.metrics import rank_auc
from src.preprocessing import list_labeled_files, preprocess_image_file

LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)
COLOR_FEATURES = ["brightness", "contrast", "saturation", "mean_r", "mean_g", "mean_b", "blue_red_ratio"]
FEATURES = COLOR_FEATURES + ["sharpness"]
SIZE_FEATURES = ["width", "height", "aspect_ratio", "megapixels"]
EXIF_ORIENTATION_TAG = 274
FEATURE_LABELS = {
    "brightness": "brightness", "contrast": "contrast", "saturation": "mean saturation",
    "mean_r": "mean red", "mean_g": "mean green", "mean_b": "mean blue",
    "blue_red_ratio": "blue-to-red ratio", "sharpness": "sharpness (Laplacian variance)",
    "width": "original width", "height": "original height", "aspect_ratio": "aspect ratio",
    "megapixels": "original megapixels",
}


def luminance(array):
    return array @ LUMA_WEIGHTS


def brightness(array):
    return float(luminance(array).mean())


def contrast(array):
    return float(luminance(array).std())


def mean_saturation(array):
    high, low = array.max(axis=-1), array.min(axis=-1)
    return float(np.where(high > 0, (high - low) / np.maximum(high, 1e-6), 0.0).mean())


def channel_means(array):
    return array.reshape(-1, 3).mean(axis=0)


def blue_red_ratio(array):
    red, _, blue = channel_means(array)
    return float((blue + 0.01) / (red + 0.01))


def laplacian(gray):
    return gray[1:-1, :-2] + gray[1:-1, 2:] + gray[:-2, 1:-1] + gray[2:, 1:-1] - 4 * gray[1:-1, 1:-1]


def sharpness(array):
    return float(laplacian(luminance(array)).var())


def image_features(array):
    red, green, blue = channel_means(array)
    return {
        "brightness": brightness(array), "contrast": contrast(array), "saturation": mean_saturation(array),
        "mean_r": float(red), "mean_g": float(green), "mean_b": float(blue),
        "blue_red_ratio": blue_red_ratio(array), "sharpness": sharpness(array),
    }


def original_size(path):
    with Image.open(path) as image:
        width, height = image.size
        orientation = image.getexif().get(EXIF_ORIENTATION_TAG, 1)
    return (height, width) if orientation in (5, 6, 7, 8) else (width, height)


def size_features(path):
    width, height = original_size(path)
    return {"width": width, "height": height, "aspect_ratio": width / height, "megapixels": width * height / 1e6}


def feature_row(path, label, split, size):
    row = {"split": split, "class": config.CLASS_NAMES[label], "path": str(path)}
    row.update(size_features(path))
    row.update(image_features(preprocess_image_file(path, size)))
    return row


def extract_features(data_dir=config.DATA_DIR, splits=config.SPLITS, size=config.IMAGE_SIZE):
    rows = [feature_row(path, label, split, size)
            for split in splits for path, label in list_labeled_files(Path(data_dir) / split)]
    return pd.DataFrame(rows)


def summarize_by_class(frame, features=FEATURES):
    return frame.groupby("class")[features].agg(["median", "mean", "std"]).round(4)


def one_vs_rest_auc(frame, feature):
    values = frame[feature].to_numpy()
    return {name: rank_auc(values, (frame["class"] == name).to_numpy()) for name in config.CLASS_NAMES}


def pair_auc(frame, feature, positive, negative):
    pair = frame[frame["class"].isin([positive, negative])]
    return rank_auc(pair[feature].to_numpy(), (pair["class"] == positive).to_numpy())


def auc_table(frame, features=FEATURES):
    rows = []
    for feature in features:
        row = {"feature": feature, **one_vs_rest_auc(frame, feature)}
        row["rainy_vs_cloudy"] = pair_auc(frame, feature, "rainy", "cloudy")
        row["sunrise_vs_shiny"] = pair_auc(frame, feature, "sunrise", "shiny")
        rows.append(row)
    return pd.DataFrame(rows).round(3)


def separation_word(auc):
    distance = abs(auc - 0.5)
    if distance >= 0.3:
        return "strong"
    if distance >= 0.15:
        return "moderate"
    return "weak" if distance >= 0.07 else "no"


def direction_word(auc):
    return "higher" if auc >= 0.5 else "lower"


def describe_extremes(frame, feature):
    medians = frame.groupby("class")[feature].median().sort_values()
    label = FEATURE_LABELS.get(feature, feature)
    return (f"{label.capitalize()} is highest for {medians.index[-1]} (median {medians.iloc[-1]:.3f}) "
            f"and lowest for {medians.index[0]} (median {medians.iloc[0]:.3f}).")


def describe_auc(name, auc):
    return f"{name}: AUC {auc:.2f}, {separation_word(auc)} separation ({direction_word(auc)})"


def describe_feature(frame, feature):
    aucs = one_vs_rest_auc(frame, feature)
    best = max(aucs, key=lambda name: abs(aucs[name] - 0.5))
    pairs = [describe_auc("rainy vs cloudy", pair_auc(frame, feature, "rainy", "cloudy")),
             describe_auc("sunrise vs shiny", pair_auc(frame, feature, "sunrise", "shiny"))]
    return (f"{describe_extremes(frame, feature)} Most separated one-vs-rest: "
            f"{describe_auc(best, aucs[best])}. Pairs: {'; '.join(pairs)}.")


def write_interpretations(frame, features=FEATURES):
    return {feature: describe_feature(frame, feature) for feature in features}


INSIGHT_SPLITS = ["train", "valid"]
INSIGHT_FEATURES = ["saturation", "blue_red_ratio", "sharpness", "brightness"]
INSIGHT_STORIES = {
    "class_balance": (
        "Sunrise is the largest class and rainy the smallest in train, and rainy has only 20 validation images. "
        "The imbalance is moderate (about 1.8 to 1), so training keeps every image and uses class weights. "
        "Rainy is the class this project cares most about and the one with the fewest validation images, so its "
        "scores move the most between runs, and new rainy uploads add the most value."),
    "saturation": (
        "Color separates the sunny classes from the gray ones: sunrise and shiny images are saturated, cloudy and "
        "rainy images are close to gray, and almost half of the rainy images are nearly colorless. This is real "
        "weather signal, but it is also a shortcut risk: a gray-toned or black-and-white sunny photo will likely be "
        "read as cloudy or rainy, and saturation barely tells rainy from cloudy."),
    "blue_red_ratio": (
        "The shiny vs sunrise boundary is almost pure color: shiny is a blue sky (blue-to-red ratio above 1.5), "
        "sunrise a warm horizon (about 0.5). This is why hue augmentation is switched off. A blue-toned sunrise "
        "is a likely mistake for the model."),
    "sharpness": (
        "Sharpness (Laplacian variance) is the only simple feature that separates rainy from cloudy. Rain streaks, "
        "splashes and wet streets carry fine texture, cloud fields are smooth. The separation holds among small "
        "images too, so it is content, not file resolution. Risk: a busy dry street may look rainy, and a sky photo "
        "during light rain may look cloudy."),
}


def load_feature_table(eda_csv=None, cache_csv=None, data_dir=None):
    for path in [Path(eda_csv or config.EDA_FEATURES_CSV), Path(cache_csv or config.INSIGHTS_CACHE)]:
        if path.exists():
            return pd.read_csv(path)
    frame = extract_features(data_dir or config.DATA_DIR, INSIGHT_SPLITS)
    Path(cache_csv or config.INSIGHTS_CACHE).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_csv or config.INSIGHTS_CACHE, index=False)
    return frame


def class_counts(data_dir=None, splits=INSIGHT_SPLITS):
    data_dir = Path(data_dir or config.DATA_DIR)
    counts = {split: {name: 0 for name in config.CLASS_NAMES} for split in splits}
    for split in splits:
        for _, label in list_labeled_files(data_dir / split):
            counts[split][config.CLASS_NAMES[label]] += 1
    return counts


def feature_records(frame, features=INSIGHT_FEATURES):
    return frame[["split", "class", *features]].round(5).to_dict(orient="records")


def insight_payload(upload_counts=None, data_dir=None):
    frame = load_feature_table(data_dir=data_dir)
    return {"splits": INSIGHT_SPLITS, "class_names": config.CLASS_NAMES, "class_counts": class_counts(data_dir),
            "upload_counts": upload_counts or {}, "features": feature_records(frame),
            "interpretations": write_interpretations(frame, INSIGHT_FEATURES), "stories": INSIGHT_STORIES}
