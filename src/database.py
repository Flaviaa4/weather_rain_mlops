import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from src import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    label TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    size_bytes INTEGER NOT NULL,
    width INTEGER,
    height INTEGER,
    uploaded_at TEXT NOT NULL,
    used_in_run INTEGER,
    status TEXT NOT NULL DEFAULT 'new'
);
CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    duration_seconds REAL,
    n_uploads INTEGER,
    n_train_images INTEGER,
    current_macro_f1 REAL,
    candidate_macro_f1 REAL,
    promoted INTEGER,
    model_version TEXT,
    message TEXT,
    details TEXT
);
"""
RUN_FIELDS = {"status", "stage", "finished_at", "duration_seconds", "n_uploads", "n_train_images", "current_macro_f1",
              "candidate_macro_f1", "promoted", "model_version", "message", "details"}


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path=None):
    db_path = db_path or config.DB_PATH
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path), timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def execute(sql, params=(), db_path=None):
    with closing(connect(db_path)) as connection, connection:
        return connection.execute(sql, params).lastrowid


def query(sql, params=(), db_path=None):
    with closing(connect(db_path)) as connection:
        return [dict(row) for row in connection.execute(sql, params).fetchall()]


def init_db(db_path=None):
    with closing(connect(db_path)) as connection, connection:
        connection.executescript(SCHEMA)
    return Path(db_path or config.DB_PATH)


def upload_exists(sha256, db_path=None):
    return bool(query("SELECT 1 FROM uploads WHERE sha256 = ?", (sha256,), db_path))


def add_upload(record, db_path=None):
    columns = ["filename", "stored_path", "label", "sha256", "size_bytes", "width", "height"]
    values = [record[name] for name in columns] + [now_iso()]
    sql = f"INSERT INTO uploads ({', '.join(columns)}, uploaded_at) VALUES ({', '.join('?' * (len(columns) + 1))})"
    return execute(sql, values, db_path)


def unused_uploads(db_path=None):
    return query("SELECT * FROM uploads WHERE used_in_run IS NULL AND status = 'new' ORDER BY id", (), db_path)


def count_unused_uploads(db_path=None):
    return len(unused_uploads(db_path))


def all_usable_uploads(db_path=None):
    return query("SELECT * FROM uploads WHERE status IN ('new', 'used') ORDER BY id", (), db_path)


def mark_uploads(ids, run_id, status, db_path=None):
    with closing(connect(db_path)) as connection, connection:
        connection.executemany("UPDATE uploads SET used_in_run = ?, status = ? WHERE id = ?",
                               [(run_id, status, upload_id) for upload_id in ids])
    return len(ids)


def uploads_summary(db_path=None):
    rows = query("SELECT label, status, COUNT(*) AS n FROM uploads GROUP BY label, status", (), db_path)
    by_label = {name: {"new": 0, "used": 0, "rejected": 0} for name in config.CLASS_NAMES}
    for row in rows:
        by_label.setdefault(row["label"], {"new": 0, "used": 0, "rejected": 0})[row["status"]] = row["n"]
    return {"by_label": by_label, "total": sum(sum(v.values()) for v in by_label.values()),
            "unused": sum(v["new"] for v in by_label.values())}


def start_run(trigger, db_path=None):
    sql = "INSERT INTO training_runs (trigger, status, stage, started_at) VALUES (?, 'running', 'queued', ?)"
    return execute(sql, (trigger, now_iso()), db_path)


def update_run(run_id, db_path=None, **fields):
    unknown = set(fields) - RUN_FIELDS
    if unknown:
        raise KeyError(f"unknown run fields: {sorted(unknown)}")
    if "details" in fields and not isinstance(fields["details"], str):
        fields["details"] = json.dumps(fields["details"], default=float)
    assignments = ", ".join(f"{name} = ?" for name in fields)
    execute(f"UPDATE training_runs SET {assignments} WHERE id = ?", [*fields.values(), run_id], db_path)
    return run_id


def decode_run(row):
    row = dict(row)
    row["details"] = json.loads(row["details"]) if row.get("details") else None
    row["promoted"] = None if row["promoted"] is None else bool(row["promoted"])
    return row


def list_runs(limit=20, db_path=None):
    return [decode_run(row) for row in query("SELECT * FROM training_runs ORDER BY id DESC LIMIT ?", (limit,), db_path)]


def get_run(run_id, db_path=None):
    rows = query("SELECT * FROM training_runs WHERE id = ?", (run_id,), db_path)
    return decode_run(rows[0]) if rows else None


def mark_interrupted_runs(db_path=None):
    sql = "UPDATE training_runs SET status = 'interrupted', finished_at = ?, message = ? WHERE status = 'running'"
    with closing(connect(db_path)) as connection, connection:
        return connection.execute(sql, (now_iso(), "The trainer stopped while this run was active.")).rowcount
