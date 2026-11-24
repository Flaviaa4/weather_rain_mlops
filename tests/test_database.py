import sqlite3

import pytest

from src import database


def record(sha="a" * 64, label="rainy"):
    return {"filename": "x.jpg", "stored_path": f"{label}/x.jpg", "label": label, "sha256": sha, "size_bytes": 10,
            "width": 4, "height": 3}


def test_uploads_are_counted_and_unique(tmp_path):
    db = tmp_path / "w.db"
    database.init_db(db)
    database.add_upload(record(), db)
    database.add_upload(record("b" * 64, "cloudy"), db)
    assert database.count_unused_uploads(db) == 2 and database.upload_exists("a" * 64, db)
    with pytest.raises(sqlite3.IntegrityError):
        database.add_upload(record(), db)
    summary = database.uploads_summary(db)
    assert summary["by_label"]["rainy"]["new"] == 1 and summary["unused"] == 2 and summary["total"] == 2


def test_marking_used_removes_from_unused(tmp_path):
    db = tmp_path / "w.db"
    database.init_db(db)
    first = database.add_upload(record(), db)
    database.mark_uploads([first], 7, "used", db)
    assert database.count_unused_uploads(db) == 0 and len(database.all_usable_uploads(db)) == 1


def test_run_lifecycle_and_interrupted_runs(tmp_path):
    db = tmp_path / "w.db"
    database.init_db(db)
    run_id = database.start_run("manual", db)
    database.update_run(run_id, db, status="completed", promoted=1, details={"current": {"macro_f1": 0.5}})
    run = database.get_run(run_id, db)
    assert run["promoted"] is True and run["details"]["current"]["macro_f1"] == 0.5
    database.start_run("auto", db)
    assert database.mark_interrupted_runs(db) == 1
    assert database.list_runs(5, db)[0]["status"] == "interrupted"


def test_update_run_rejects_unknown_fields(tmp_path):
    db = tmp_path / "w.db"
    database.init_db(db)
    with pytest.raises(KeyError):
        database.update_run(database.start_run("manual", db), db, colour="blue")
