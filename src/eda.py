import argparse
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src import config, insights
from src.preprocessing import list_labeled_files, normalize_array, open_image_bytes, resize_image, to_array, to_rgb

EDA_DIR = config.EXPERIMENTS_DIR / "eda"
ANALYSIS_SPLITS = ["train", "valid"]
CLASS_COLORS = {"cloudy": "#2a78d6", "rainy": "#eb6834", "shiny": "#1baf7a", "sunrise": "#eda100"}
SURFACE, INK, INK_SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BOXPLOT_FEATURES = ["brightness", "contrast", "saturation", "blue_red_ratio", "sharpness", "megapixels"]
LOG_SCALE_FEATURES = {"sharpness", "megapixels"}
PREVIEW_COUNT = 8


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_SECONDARY, "text.color": INK,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    })


def style_axis(axis, grid_axis="y"):
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(length=0)


def save_figure(figure, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def save_table(frame, name, index=False):
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(EDA_DIR / name, index=index)
    return EDA_DIR / name


def count_files(data_dir, splits):
    rows = [{"split": split, "class": config.CLASS_NAMES[label]}
            for split in splits for _, label in list_labeled_files(Path(data_dir) / split)]
    return pd.DataFrame(rows).groupby(["split", "class"]).size().unstack(fill_value=0)


def draw_count_bars(axis, counts, split):
    values = [int(counts.loc[split, name]) for name in config.CLASS_NAMES]
    bars = axis.bar(config.CLASS_NAMES, values, width=0.7, edgecolor=SURFACE, linewidth=2,
                    color=[CLASS_COLORS[name] for name in config.CLASS_NAMES])
    axis.bar_label(bars, padding=3, color=INK_SECONDARY, fontsize=9)
    axis.set_title(f"{split} ({sum(values)} images)")
    style_axis(axis)


def plot_class_distribution(counts):
    figure, axes = plt.subplots(1, len(config.SPLITS), figsize=(12, 3.6), sharey=True)
    for axis, split in zip(axes, config.SPLITS):
        draw_count_bars(axis, counts, split)
    axes[0].set_ylabel("images")
    figure.suptitle("Class distribution per split after duplicate cleanup", x=0.01, y=1.04, ha="left", fontweight="bold")
    return save_figure(figure, "class_distribution.png")


def sample_paths(frame, class_name, count, seed):
    rows = frame[(frame["class"] == class_name) & (frame["split"] == "train")]
    return rows.sample(n=min(count, len(rows)), random_state=seed)["path"].tolist()


def load_display_array(path, size=config.IMAGE_SIZE):
    return to_array(resize_image(to_rgb(open_image_bytes(Path(path).read_bytes())), size))


def plot_sample_grid(frame, count=8, seed=config.SAMPLE_SEED):
    figure, axes = plt.subplots(len(config.CLASS_NAMES), count, figsize=(count * 1.4, len(config.CLASS_NAMES) * 1.5))
    for row, class_name in enumerate(config.CLASS_NAMES):
        for column, path in enumerate(sample_paths(frame, class_name, count, seed)):
            axes[row, column].imshow(load_display_array(path))
        axes[row, 0].set_ylabel(class_name, rotation=0, ha="right", va="center", fontsize=11)
    for axis in axes.flat:
        axis.set_xticks([]), axis.set_yticks([]), [spine.set_visible(False) for spine in axis.spines.values()]
    figure.suptitle(f"Random train samples at {config.IMAGE_SIZE} px (seed {seed})", x=0.01, ha="left", fontweight="bold")
    return save_figure(figure, "sample_grid.png")


def set_plain_log_ticks(axis, ticks=(100, 200, 500, 1000, 2000, 5000)):
    for scale_axis in (axis.xaxis, axis.yaxis):
        scale_axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        scale_axis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d"))
        scale_axis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def draw_size_panel(axis, frame, class_name):
    axis.scatter(frame["width"], frame["height"], s=14, color=GRID, linewidths=0)
    rows = frame[frame["class"] == class_name]
    axis.scatter(rows["width"], rows["height"], s=36, color=CLASS_COLORS[class_name], edgecolor=SURFACE, linewidth=0.8)
    share = (rows["megapixels"] > 0.1).mean()
    axis.set_title(f"{class_name}: median {rows['width'].median():.0f}x{rows['height'].median():.0f}, {share:.0%} above 0.1 MP")
    axis.set_xscale("log"), axis.set_yscale("log")
    set_plain_log_ticks(axis)
    style_axis(axis, "both")


def plot_image_sizes(frame):
    figure, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True, sharey=True)
    for axis, class_name in zip(axes.flat, config.CLASS_NAMES):
        draw_size_panel(axis, frame, class_name)
    for axis in axes[1]:
        axis.set_xlabel("original width (px, log)")
    for axis in axes[:, 0]:
        axis.set_ylabel("original height (px, log)")
    figure.suptitle("Original image size by class, train and valid (gray: all classes)", x=0.01, ha="left", fontweight="bold")
    return save_figure(figure, "image_size_distribution.png")


def draw_feature_box(axis, frame, feature):
    groups = [frame.loc[frame["class"] == name, feature] for name in config.CLASS_NAMES]
    boxes = axis.boxplot(groups, tick_labels=config.CLASS_NAMES, patch_artist=True, widths=0.6, showfliers=True,
                         medianprops={"color": INK, "linewidth": 2}, flierprops={"markersize": 3, "markeredgecolor": MUTED})
    for patch, name in zip(boxes["boxes"], config.CLASS_NAMES):
        patch.set(facecolor=CLASS_COLORS[name], edgecolor=INK_SECONDARY, alpha=0.85)
    if feature in LOG_SCALE_FEATURES:
        axis.set_yscale("log")
    axis.set_title(insights.FEATURE_LABELS[feature])
    style_axis(axis)


def plot_feature_boxplots(frame, features=BOXPLOT_FEATURES):
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for axis, feature in zip(axes.flat, features):
        draw_feature_box(axis, frame, feature)
    figure.suptitle(f"Image features by class, train and valid, computed at {config.IMAGE_SIZE} px",
                    x=0.01, ha="left", fontweight="bold")
    return save_figure(figure, "feature_boxplots.png")


def resized_uint8(path, sizes):
    image = to_rgb(open_image_bytes(Path(path).read_bytes()))
    return {size: to_array(resize_image(image, size)) for size in sizes}


def reconstruction_psnr(small, reference):
    size = reference.shape[0]
    restored = to_array(Image.fromarray(small).resize((size, size), Image.Resampling.BILINEAR))
    mse = np.mean((restored.astype(np.float32) - reference.astype(np.float32)) ** 2)
    return float(10 * np.log10(255.0 ** 2 / max(mse, 1e-10)))


def size_rows(path, class_name, sizes, reference_size):
    arrays = resized_uint8(path, sizes + [reference_size])
    rows = []
    for size in sizes + [reference_size]:
        row = {"path": path, "class": class_name, "size": size, **insights.image_features(normalize_array(arrays[size]))}
        row["psnr_vs_reference"] = reconstruction_psnr(arrays[size], arrays[reference_size])
        rows.append(row)
    return rows


def run_size_experiment(frame, sizes=config.EDA_SIZES, reference_size=config.EDA_REFERENCE_SIZE):
    rows = [row for path, name in zip(frame["path"], frame["class"]) for row in size_rows(path, name, sizes, reference_size)]
    return pd.DataFrame(rows)


def feature_correlations(table, reference_size=config.EDA_REFERENCE_SIZE):
    wide = table.pivot(index="path", columns="size", values=insights.FEATURES)
    rows = [{"size": size, **{f: float(np.corrcoef(wide[(f, size)], wide[(f, reference_size)])[0, 1])
                              for f in insights.FEATURES}} for size in sorted(table["size"].unique())]
    return pd.DataFrame(rows).round(4)


def psnr_by_class(table, reference_size=config.EDA_REFERENCE_SIZE):
    rows = table[table["size"] != reference_size]
    return rows.pivot_table(index="size", columns="class", values="psnr_vs_reference", aggfunc="median").round(2)


def sharpness_auc_by_size(table):
    rows = []
    for size, group in table.groupby("size"):
        is_rainy = (group["class"] == "rainy").to_numpy()
        rows.append({"size": size, "rainy_vs_rest": insights.rank_auc(group["sharpness"].to_numpy(), is_rainy),
                     "rainy_vs_cloudy": insights.pair_auc(group, "sharpness", "rainy", "cloudy"),
                     "median_sharpness_rainy": group.loc[is_rainy, "sharpness"].median(),
                     "median_sharpness_rest": group.loc[~is_rainy, "sharpness"].median()})
    return pd.DataFrame(rows).round(4)


def pick_size_examples(frame, seed=config.SAMPLE_SEED):
    picks = [frame[frame["class"] == name].sample(n=1, random_state=seed)["path"].iloc[0] for name in config.CLASS_NAMES]
    largest_rainy = frame[frame["class"] == "rainy"].sort_values("megapixels")["path"].iloc[-1]
    return picks + [largest_rainy]


def plot_size_comparison(paths, sizes=config.EDA_SIZES, reference_size=config.EDA_REFERENCE_SIZE):
    all_sizes = sizes + [reference_size]
    figure, axes = plt.subplots(len(paths), len(all_sizes), figsize=(len(all_sizes) * 2.4, len(paths) * 1.9))
    for row, path in enumerate(paths):
        arrays = resized_uint8(path, all_sizes)
        for column, size in enumerate(all_sizes):
            axes[row, column].imshow(arrays[size], interpolation="nearest")
            axes[row, column].set_xticks([]), axes[row, column].set_yticks([])
            axes[0, column].set_title(f"{size} px")
        axes[row, 0].set_ylabel(Path(path).parent.name, rotation=0, ha="right", va="center")
    figure.suptitle("Same images at 96, 128, 160 and 224 px (nearest-neighbour display)", x=0.01, ha="left", fontweight="bold")
    return save_figure(figure, "image_size_comparison.png")


def save_interpretations(frame):
    lines = ["# Feature interpretations (generated)", "",
             f"Train and valid, {len(frame)} images, features at {config.IMAGE_SIZE} px.", ""]
    lines += [f"- **{feature}**: {text}" for feature, text in insights.write_interpretations(frame).items()]
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    (EDA_DIR / "feature_interpretations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return EDA_DIR / "feature_interpretations.md"


def save_feature_tables(frame):
    save_table(frame, "features_train_valid.csv")
    save_table(insights.summarize_by_class(frame, insights.FEATURES + insights.SIZE_FEATURES), "features_by_class.csv", index=True)
    save_table(insights.auc_table(frame, insights.FEATURES + insights.SIZE_FEATURES), "feature_auc.csv")
    return save_interpretations(frame)


def save_size_experiment(frame):
    table = run_size_experiment(frame)
    save_table(table, "size_experiment_rows.csv")
    save_table(feature_correlations(table), "size_feature_correlation.csv")
    save_table(psnr_by_class(table), "size_psnr_by_class.csv", index=True)
    save_table(sharpness_auc_by_size(table), "size_sharpness_auc.csv")
    return plot_size_comparison(pick_size_examples(frame))


def run_eda(data_dir=config.DATA_DIR):
    apply_style()
    plot_class_distribution(count_files(data_dir, config.SPLITS))
    frame = insights.extract_features(data_dir, ANALYSIS_SPLITS)
    save_feature_tables(frame)
    plot_sample_grid(frame)
    plot_image_sizes(frame)
    plot_feature_boxplots(frame)
    save_size_experiment(frame)
    print("EDA written to", config.FIGURES_DIR, "and", EDA_DIR)


def pick_preview_images(frame, seed=config.SAMPLE_SEED):
    train = frame[frame["split"] == "train"]
    picks = [(f"{name} (random)", train[train["class"] == name].sample(n=1, random_state=seed)["path"].iloc[0])
             for name in config.CLASS_NAMES]
    sunrise, cloudy = train[train["class"] == "sunrise"], train[train["class"] == "cloudy"]
    picks.append(("sunrise (least saturated)", sunrise.sort_values("saturation")["path"].iloc[0]))
    picks.append(("cloudy (brightest)", cloudy.sort_values("brightness")["path"].iloc[-1]))
    return picks


def augmented_versions(path, strength, count=PREVIEW_COUNT, seed=config.SAMPLE_SEED):
    import tensorflow as tf
    from src.preprocessing import augment_image, preprocess_image_file
    image = tf.constant(preprocess_image_file(path))
    return [augment_image(image, tf.constant([seed, index], tf.int64), strength).numpy() for index in range(count)]


def plot_augmentation_preview(picks, strength):
    figure, axes = plt.subplots(len(picks), PREVIEW_COUNT + 1, figsize=((PREVIEW_COUNT + 1) * 1.4, len(picks) * 1.5))
    for row, (label, path) in enumerate(picks):
        images = [load_display_array(path)] + augmented_versions(path, strength)
        for column, image in enumerate(images):
            axes[row, column].imshow(image)
            axes[row, column].set_xticks([]), axes[row, column].set_yticks([])
        axes[row, 0].set_ylabel(label, rotation=0, ha="right", va="center")
    axes[0, 0].set_title("original")
    figure.suptitle(f"Augmentation preview: {strength} {config.AUGMENT_STRENGTHS[strength]}", x=0.01, ha="left", fontweight="bold")
    return save_figure(figure, f"augmentation_preview_{strength}.png")


def run_augmentation_preview(data_dir=config.DATA_DIR):
    apply_style()
    picks = pick_preview_images(insights.extract_features(data_dir, ["train"]))
    paths = [plot_augmentation_preview(picks, strength) for strength in ["light", "medium", "strong"]]
    print("Augmentation previews written:", *paths, sep="\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment-preview", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_augmentation_preview() if parse_args().augment_preview else run_eda()
