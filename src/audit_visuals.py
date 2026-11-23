from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from src import config
from src.audit import load_rgb

TILE = 160


def make_tile(path, caption):
    tile = ImageOps.pad(load_rgb(path), (TILE, TILE), color=(255, 255, 255))
    canvas = Image.new("RGB", (TILE, TILE + 28), (255, 255, 255))
    canvas.paste(tile, (0, 0))
    ImageDraw.Draw(canvas).text((3, TILE + 2), caption, fill=(0, 0, 0))
    return canvas


def short_caption(index, path, label):
    stem = Path(path).name.split("_jpg")[0]
    return f"{index} {label}\n{stem}"


def grid_sheet(tiles, columns):
    rows = (len(tiles) + columns - 1) // columns
    width, height = tiles[0].size
    sheet = Image.new("RGB", (columns * width, rows * height), (220, 220, 220))
    for position, tile in enumerate(tiles):
        sheet.paste(tile, ((position % columns) * width, (position // columns) * height))
    return sheet


def pair_tiles(pairs):
    tiles = []
    for index, row in enumerate(pairs.itertuples()):
        caption_a = f"P{index} {row.split_a}/{row.label_a}\nh={row.hamming} mae={row.thumb_mae}"
        caption_b = f"P{index} {row.split_b}/{row.label_b}\n{Path(row.path_b).name.split('_jpg')[0]}"
        tiles.extend([make_tile(row.path_a, caption_a), make_tile(row.path_b, caption_b)])
    return tiles


def save_pair_sheet(pairs, file_name):
    out = config.AUDIT_DIR / "sheets" / file_name
    out.parent.mkdir(parents=True, exist_ok=True)
    grid_sheet(pair_tiles(pairs), columns=4).save(out, quality=85)
    return out


def save_image_sheet(frame, file_name, columns=5):
    tiles = [make_tile(r.path, short_caption(i, r.path, r.label)) for i, r in enumerate(frame.itertuples())]
    out = config.AUDIT_DIR / "sheets" / file_name
    out.parent.mkdir(parents=True, exist_ok=True)
    grid_sheet(tiles, columns).save(out, quality=85)
    return out


def sample_pairs(pairs, query, n, seed):
    subset = pairs.query(query)
    return subset.sample(min(n, len(subset)), random_state=seed)


def simple_color_stats(path):
    pixels = np.asarray(load_rgb(path).resize((64, 64)), dtype=np.float32) / 255.0
    red, blue = pixels[..., 0].mean(), pixels[..., 2].mean()
    return {"brightness": float(pixels.mean()), "warmth": float(red - blue)}


def add_color_stats(frame, data_dir):
    stats = pd.DataFrame([simple_color_stats(data_dir / p) for p in frame["path"]], index=frame.index)
    return frame.join(stats)


def pick_boundary(frame, label, column, ascending, n):
    return frame[frame["label"] == label].sort_values(column, ascending=ascending).head(n)


def save_label_review_sheets(table, data_dir, per_class, seed):
    full = table.assign(path=[str(data_dir / p) for p in table["path"]])
    for label in config.CLASS_NAMES:
        sample = full[full["label"] == label].sample(per_class, random_state=seed)
        save_image_sheet(sample, f"label_review_{label}.jpg")
    stats = add_color_stats(full, Path("."))
    half = per_class // 2
    warm_pair = pd.concat([pick_boundary(stats, "shiny", "warmth", False, half), pick_boundary(stats, "sunrise", "warmth", True, half)])
    gray_pair = pd.concat([pick_boundary(stats, "cloudy", "brightness", True, half), pick_boundary(stats, "rainy", "brightness", False, half)])
    save_image_sheet(warm_pair, "label_review_shiny_vs_sunrise.jpg")
    save_image_sheet(gray_pair, "label_review_cloudy_vs_rainy.jpg")
