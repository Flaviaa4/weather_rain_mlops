import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from src import config

REVIEWED_PATH = config.AUDIT_DIR / "reviewed_not_duplicates.json"
SPLIT_PRIORITY = {"train": 0, "valid": 1, "test": 2}


def is_image_path(path):
    return path.suffix.lower() in config.IMAGE_EXTENSIONS


def list_class_files(split_dir, class_name):
    class_dir = split_dir / class_name
    if not class_dir.is_dir():
        return []
    return sorted(p for p in class_dir.iterdir() if p.is_file() and is_image_path(p))


def list_split_files(data_dir, split):
    rows = []
    for class_name in config.CLASS_NAMES:
        for path in list_class_files(data_dir / split, class_name):
            rows.append({"split": split, "label": class_name, "path": path})
    return rows


def list_all_files(data_dir, splits):
    rows = []
    for split in splits:
        rows.extend(list_split_files(data_dir, split))
    return rows


def find_unexpected_folders(data_dir, splits):
    found = []
    for split in splits:
        for child in (data_dir / split).iterdir():
            if child.is_dir() and child.name not in config.CLASS_NAMES:
                found.append(str(child))
    return found


def original_stem(path):
    return re.sub(r"\.rf\.[0-9a-f]+$", "", Path(path).stem)


def check_image_integrity(path):
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
        return ""
    except Exception as error:
        return f"{type(error).__name__}: {error}"


def read_exif_orientation(image):
    try:
        return int(image.getexif().get(274, 1))
    except Exception:
        return 1


def read_image_properties(path):
    with Image.open(path) as image:
        return {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": image.format,
            "exif_orientation": read_exif_orientation(image),
        }


def load_rgb(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def compute_dhash(path, hash_size):
    gray = load_rgb(path).convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def audit_one_file(row, data_dir):
    path = row["path"]
    record = {**row, "path": path.relative_to(data_dir).as_posix(), "original_stem": original_stem(path)}
    record["file_size_kb"] = round(path.stat().st_size / 1024, 2)
    record["error"] = check_image_integrity(path)
    if record["error"]:
        return record
    record.update(read_image_properties(path))
    record["aspect_ratio"] = round(record["width"] / record["height"], 4)
    record["megapixels"] = round(record["width"] * record["height"] / 1e6, 4)
    record["dhash"] = str(compute_dhash(path, config.DHASH_SIZE))
    return record


def build_audit_table(rows, data_dir):
    return pd.DataFrame([audit_one_file(row, data_dir) for row in rows])


def hashes_to_bit_matrix(hashes, n_bits):
    as_text = [format(int(h), f"0{n_bits}b") for h in hashes]
    return np.array([[c == "1" for c in text] for text in as_text], dtype=np.int32)


def pairwise_hamming(bit_matrix):
    ones = bit_matrix.sum(axis=1)
    return ones[:, None] + ones[None, :] - 2 * (bit_matrix @ bit_matrix.T)


def describe_pair(table, i, j, hamming, corr):
    a, b = table.iloc[i], table.iloc[j]
    return {
        "path_a": a["path"], "split_a": a["split"], "label_a": a["label"],
        "path_b": b["path"], "split_b": b["split"], "label_b": b["label"],
        "hamming": int(hamming), "gray_corr": round(float(corr), 4),
        "cross_split": a["split"] != b["split"],
        "label_conflict": a["label"] != b["label"],
    }


def correlation_matrix(table, data_dir):
    matrix = np.stack([gray_thumbnail(data_dir / p, config.THUMB_SIZE) for p in table["path"]])
    return matrix @ matrix.T / matrix.shape[1]


def find_candidates(table, data_dir):
    valid = table[table["error"] == ""].reset_index(drop=True)
    hamming = pairwise_hamming(hashes_to_bit_matrix(valid["dhash"], config.DHASH_SIZE ** 2))
    corr = correlation_matrix(valid, data_dir)
    flagged = (hamming <= config.DUPLICATE_MAX_DISTANCE) | (corr >= config.CORR_STAGE_MIN_CORR)
    rows_i, rows_j = np.where(np.triu(flagged, k=1))
    pairs = [describe_pair(valid, i, j, hamming[i, j], corr[i, j]) for i, j in zip(rows_i, rows_j)]
    return pd.DataFrame(pairs)


def color_thumbnail(path, size):
    image = load_rgb(path).resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def thumbnail_distance(path_a, path_b):
    a = color_thumbnail(path_a, config.THUMB_SIZE)
    b = color_thumbnail(path_b, config.THUMB_SIZE)
    return float(np.abs(a - b).mean())


def gray_thumbnail(path, size):
    image = load_rgb(path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
    values = np.asarray(image, dtype=np.float32).flatten()
    return (values - values.mean()) / (values.std() + 1e-6)


def gray_correlation(path_a, path_b):
    a = gray_thumbnail(path_a, config.THUMB_SIZE)
    b = gray_thumbnail(path_b, config.THUMB_SIZE)
    return float(np.mean(a * b))


def add_color_distance(pairs, data_dir):
    mae = [round(thumbnail_distance(data_dir / a, data_dir / b), 4) for a, b in zip(pairs["path_a"], pairs["path_b"])]
    return pairs.assign(thumb_mae=mae)


def load_reviewed_exceptions():
    if not REVIEWED_PATH.exists():
        return []
    return json.loads(REVIEWED_PATH.read_text())


def is_reviewed_exception(path_a, path_b, exceptions):
    names = {Path(path_a).name, Path(path_b).name}
    for item in exceptions:
        hit_a = any(n.startswith(item["file_a"]) for n in names)
        hit_b = any(n.startswith(item["file_b"]) for n in names)
        if hit_a and hit_b:
            return True
    return False


def hash_stage_match(pairs):
    close_hash = pairs["hamming"] <= config.DUPLICATE_MAX_DISTANCE
    similar = (pairs["thumb_mae"] <= config.DUPLICATE_MAX_MAE) | (pairs["gray_corr"] >= config.DUPLICATE_MIN_CORR)
    return close_hash & similar


def correlation_stage_match(pairs):
    return (pairs["gray_corr"] >= config.CORR_STAGE_MIN_CORR) & (pairs["thumb_mae"] <= config.CORR_STAGE_MAX_MAE)


def mark_confirmed(pairs):
    exceptions = load_reviewed_exceptions()
    rejected = pd.Series([is_reviewed_exception(a, b, exceptions) for a, b in zip(pairs["path_a"], pairs["path_b"])], index=pairs.index)
    matched = hash_stage_match(pairs) | correlation_stage_match(pairs)
    return pairs.assign(reviewed_reject=rejected, confirmed=matched & ~rejected)


def find_root(parents, node):
    while parents[node] != node:
        parents[node] = parents[parents[node]]
        node = parents[node]
    return node


def build_duplicate_groups(confirmed_pairs):
    parents = {}
    for a, b in zip(confirmed_pairs["path_a"], confirmed_pairs["path_b"]):
        parents.setdefault(a, a)
        parents.setdefault(b, b)
        parents[find_root(parents, a)] = find_root(parents, b)
    groups = {}
    for node in parents:
        groups.setdefault(find_root(parents, node), []).append(node)
    return [sorted(members) for members in groups.values()]


def keeper_sort_key(path):
    return (SPLIT_PRIORITY[path.split("/")[0]], path)


def plan_group(group_id, members, table_by_path):
    keeper = min(members, key=keeper_sort_key)
    rows = []
    for path in members:
        info = table_by_path.loc[path]
        action = "keep" if path == keeper else "quarantine"
        rows.append({"group_id": group_id, "path": path, "split": info["split"], "label": info["label"],
                     "action": action, "keeper": keeper, "group_size": len(members)})
    return rows


def plan_quarantine(table, confirmed_pairs):
    table_by_path = table.set_index("path")
    rows = []
    for group_id, members in enumerate(build_duplicate_groups(confirmed_pairs)):
        rows.extend(plan_group(group_id, members, table_by_path))
    return pd.DataFrame(rows)


def move_to_quarantine(data_dir, relative_path):
    source = data_dir / relative_path
    target = config.QUARANTINE_DIR / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))


def apply_quarantine(plan, data_dir):
    to_move = plan[plan["action"] == "quarantine"]
    for relative_path in to_move["path"]:
        if (data_dir / relative_path).exists():
            move_to_quarantine(data_dir, relative_path)
    return len(to_move)


def count_split_class(table):
    counts = pd.crosstab(table["split"], table["label"], margins=True, margins_name="total")
    return counts.reindex(index=config.SPLITS + ["total"], columns=config.CLASS_NAMES + ["total"]).fillna(0).astype(int)


def summarize_properties_by_class(table):
    numeric = ["width", "height", "aspect_ratio", "megapixels", "file_size_kb"]
    return table.groupby("label")[numeric].agg(["median", "mean", "min", "max"]).round(3)


def count_modes_by_class(table):
    return pd.crosstab(table["label"], table["mode"])


def top_resolutions_by_class(table, top_n):
    table = table.assign(resolution=table["width"].astype(str) + "x" + table["height"].astype(str))
    counts = table.groupby(["label", "resolution"]).size().rename("count").reset_index()
    counts["share"] = (counts["count"] / counts.groupby("label")["count"].transform("sum")).round(3)
    return counts.sort_values(["label", "count"], ascending=[True, False]).groupby("label").head(top_n)


def save_table(frame, name, index=False):
    config.AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.AUDIT_DIR / name, index=index)


def scan_dataset(data_dir):
    table = build_audit_table(list_all_files(data_dir, config.SPLITS), data_dir)
    pairs = mark_confirmed(add_color_distance(find_candidates(table, data_dir), data_dir))
    return table, pairs


def save_scan(table, pairs, prefix):
    save_table(table, f"{prefix}audit_table.csv")
    save_table(pairs, f"{prefix}duplicate_candidates.csv")
    save_table(pairs[pairs["confirmed"]], f"{prefix}confirmed_duplicates.csv")
    save_table(count_split_class(table), f"{prefix}split_class_counts.csv", index=True)
    save_table(summarize_properties_by_class(table), f"{prefix}properties_by_class.csv", index=True)
    save_table(count_modes_by_class(table), f"{prefix}modes_by_class.csv", index=True)
    save_table(top_resolutions_by_class(table, 5), f"{prefix}top_resolutions_by_class.csv")


def print_scan_summary(table, pairs, data_dir):
    confirmed = pairs[pairs["confirmed"]]
    print(count_split_class(table))
    print("corrupt files:", int((table["error"] != "").sum()))
    print("unexpected folders:", find_unexpected_folders(data_dir, config.SPLITS))
    print("candidates:", len(pairs), "with label conflict:", int(pairs["label_conflict"].sum()))
    print("confirmed duplicates:", len(confirmed), "cross-split:", int(confirmed["cross_split"].sum()),
          "label conflict:", int(confirmed["label_conflict"].sum()))


def run_audit(data_dir, prefix):
    table, pairs = scan_dataset(data_dir)
    save_scan(table, pairs, prefix)
    print_scan_summary(table, pairs, data_dir)
    return table, pairs


def run_quarantine(data_dir):
    table, pairs = run_audit(data_dir, "before_")
    plan = plan_quarantine(table, pairs[pairs["confirmed"]])
    save_table(plan, "quarantine_manifest.csv")
    print("files moved to quarantine:", apply_quarantine(plan, data_dir))
    run_audit(data_dir, "after_")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--quarantine", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.quarantine:
        run_quarantine(args.data_dir)
    else:
        run_audit(args.data_dir, "")
