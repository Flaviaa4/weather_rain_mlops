import io
import zipfile

import pytest

from src import config, database, uploads


def make_zip(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def test_validate_label():
    assert uploads.validate_label(" Rainy ") == "rainy"
    with pytest.raises(uploads.UploadRefused) as error:
        uploads.validate_label("snowy")
    assert error.value.status_code == 422


def test_safe_filename_blocks_path_tricks():
    assert uploads.safe_filename("../../etc/passwd.jpg") == "passwd.jpg"
    assert uploads.safe_filename("C:\\Users\\me\\my photo (1).JPG") == "my_photo_1_.JPG"


def test_save_uploads_saves_registers_and_skips_duplicates(service_env, make_jpeg):
    files = [("a.jpg", make_jpeg(1)), ("b.jpg", make_jpeg(2)), ("again.jpg", make_jpeg(1)), ("bad.jpg", b"nope")]
    result = uploads.save_uploads(files, "rainy")
    assert (result["saved"], result["duplicate"], result["rejected"]) == (2, 1, 1)
    rows = database.unused_uploads()
    assert [row["label"] for row in rows] == ["rainy", "rainy"]
    assert all(uploads.upload_file_path(row).exists() for row in rows)
    assert uploads.upload_file_path(rows[0]).parent == service_env["UPLOADS_DIR"] / "rainy"


def test_disk_guard_refuses_and_saves_nothing(service_env, make_jpeg, monkeypatch):
    monkeypatch.setattr(config, "MIN_FREE_DISK_MB", 10**9)
    with pytest.raises(uploads.UploadRefused) as error:
        uploads.save_uploads([("a.jpg", make_jpeg())], "cloudy")
    assert error.value.status_code == 507 and database.count_unused_uploads() == 0


def test_too_many_or_no_files(service_env):
    with pytest.raises(uploads.UploadRefused):
        uploads.save_uploads([], "cloudy")


def test_zip_uses_folder_names_as_labels(service_env, make_jpeg):
    archive = make_zip({"batch/rainy/r1.jpg": make_jpeg(1), "batch/shiny/s1.jpg": make_jpeg(2),
                        "__MACOSX/batch/rainy/._r1.jpg": b"junk", "batch/snow/x.jpg": make_jpeg(3),
                        "loose.jpg": make_jpeg(4)})
    result = uploads.save_zip_uploads(archive)
    assert result["saved"] == 2 and result["rejected"] == 2
    assert database.uploads_summary()["by_label"]["shiny"]["new"] == 1


def test_bad_zip_is_refused(service_env):
    with pytest.raises(uploads.UploadRefused) as error:
        uploads.save_zip_uploads(b"not a zip")
    assert error.value.status_code == 415
