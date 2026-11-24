import hashlib
import io
import re
import shutil
import sqlite3
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image

from src import config, database
from src.prediction import InvalidImageError, validate_image_bytes

SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
IGNORED_ZIP_PARTS = {"__MACOSX", ".DS_Store"}


class UploadRefused(Exception):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code


def validate_label(label, class_names=config.CLASS_NAMES):
    cleaned = (label or "").strip().lower()
    if cleaned not in class_names:
        raise UploadRefused(f"Unknown label '{label}'. Use one of: {', '.join(class_names)}.", status_code=422)
    return cleaned


def free_disk_mb(path=None):
    path = Path(path or config.UPLOADS_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / (1024 * 1024)


def check_disk_space(incoming_bytes=0, path=None, minimum_mb=None):
    minimum_mb = config.MIN_FREE_DISK_MB if minimum_mb is None else minimum_mb
    free = free_disk_mb(path)
    if free - incoming_bytes / (1024 * 1024) < minimum_mb:
        raise UploadRefused(f"Server disk is nearly full ({free:.0f} MB free, {minimum_mb} MB must stay free). "
                            "Upload refused, no file was saved.", status_code=507)
    return free


def safe_filename(filename):
    name = SAFE_NAME.sub("_", PurePosixPath(str(filename).replace("\\", "/")).name).strip("._")
    return name[-80:] or "image.jpg"


def image_size(data):
    with Image.open(io.BytesIO(data)) as image:
        return image.size


def stored_path(label, digest, filename, uploads_dir=None):
    return Path(uploads_dir or config.UPLOADS_DIR) / label / f"{digest[:12]}_{safe_filename(filename)}"


def write_file(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(data)
    partial.replace(path)
    return path


def upload_record(filename, label, data, path, digest, uploads_dir):
    width, height = image_size(data)
    relative = Path(path).relative_to(uploads_dir or config.UPLOADS_DIR).as_posix()
    return {"filename": safe_filename(filename), "stored_path": relative, "label": label, "sha256": digest,
            "size_bytes": len(data), "width": width, "height": height}


def save_upload(data, filename, label, uploads_dir=None, db_path=None):
    try:
        validate_image_bytes(data, filename)
    except InvalidImageError as error:
        return {"filename": filename, "status": "rejected", "reason": str(error)}
    digest = hashlib.sha256(data).hexdigest()
    if database.upload_exists(digest, db_path):
        return {"filename": filename, "status": "duplicate", "reason": "This exact image was already uploaded."}
    path = write_file(stored_path(label, digest, filename, uploads_dir), data)
    try:
        upload_id = database.add_upload(upload_record(filename, label, data, path, digest, uploads_dir), db_path)
    except sqlite3.IntegrityError:
        return {"filename": filename, "status": "duplicate", "reason": "This exact image was already uploaded."}
    return {"filename": filename, "status": "saved", "id": upload_id, "label": label}


def upload_file_path(row, uploads_dir=None):
    return Path(uploads_dir or config.UPLOADS_DIR) / row["stored_path"]


def summarize_results(results):
    counts = {status: sum(r["status"] == status for r in results) for status in ["saved", "duplicate", "rejected"]}
    return {**counts, "results": results}


def check_file_count(count, maximum=config.MAX_FILES_PER_UPLOAD):
    if count == 0:
        raise UploadRefused("No files were sent.")
    if count > maximum:
        raise UploadRefused(f"Too many files ({count}). The limit is {maximum} per request.", status_code=413)


def save_uploads(files, label, uploads_dir=None, db_path=None):
    label = validate_label(label)
    check_file_count(len(files))
    check_disk_space(sum(len(data) for _, data in files), uploads_dir)
    return summarize_results([save_upload(data, name, label, uploads_dir, db_path) for name, data in files])


def zip_member_label(name):
    parts = [part for part in PurePosixPath(name).parts if part not in IGNORED_ZIP_PARTS]
    return parts[-2].strip().lower() if len(parts) >= 2 else None


def is_wanted_member(info):
    parts = PurePosixPath(info.filename).parts
    return not info.is_dir() and not any(part in IGNORED_ZIP_PARTS or part.startswith("._") for part in parts)


def open_zip(zip_bytes, max_mb=config.MAX_ZIP_MB):
    try:
        archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile:
        raise UploadRefused("The file is not a valid zip archive.", status_code=415)
    if sum(info.file_size for info in archive.infolist()) > max_mb * 1024 * 1024:
        raise UploadRefused(f"The zip expands to more than {max_mb} MB.", status_code=413)
    return archive


def extract_zip_uploads(zip_bytes, class_names=config.CLASS_NAMES):
    archive = open_zip(zip_bytes)
    members = [info for info in archive.infolist() if is_wanted_member(info)]
    entries, skipped = [], []
    for info in members:
        label = zip_member_label(info.filename)
        if label in class_names:
            entries.append((label, PurePosixPath(info.filename).name, archive.read(info)))
        else:
            skipped.append({"filename": info.filename, "status": "rejected",
                            "reason": "The file is not inside a folder named after a class."})
    return entries, skipped


def save_zip_uploads(zip_bytes, uploads_dir=None, db_path=None):
    check_disk_space(len(zip_bytes), uploads_dir)
    entries, skipped = extract_zip_uploads(zip_bytes)
    check_file_count(len(entries) + len(skipped), maximum=config.MAX_FILES_PER_UPLOAD * 4)
    check_disk_space(sum(len(data) for _, _, data in entries), uploads_dir)
    results = [save_upload(data, name, label, uploads_dir, db_path) for label, name, data in entries]
    return summarize_results(results + skipped)
