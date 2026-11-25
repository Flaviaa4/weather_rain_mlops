import importlib.util
import logging
import os
import shutil
import socket
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src import config, database, insights, prediction, retrain, uploads

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("trainer_api")
STARTED = time.time()
STARTED_AT = datetime.now(timezone.utc).isoformat(timespec="seconds")
HOSTNAME = socket.gethostname()


def tensorflow_available():
    return importlib.util.find_spec("tensorflow") is not None


def bootstrap_placeholder():
    if not config.IS_PLACEHOLDER or os.environ.get("BOOTSTRAP_PLACEHOLDER", "1") != "1":
        return
    try:
        from src.placeholder import ensure_placeholder_files
        LOGGER.info("Placeholder model files: %s", ensure_placeholder_files(with_keras=tensorflow_available()))
    except Exception:
        LOGGER.exception("Could not create the placeholder model")


def log_base_data(report):
    if report["ok"]:
        LOGGER.info("Base data for retraining: train %d images, valid %d images", report["train"]["n_images"],
                    report["valid"]["n_images"])
    else:
        LOGGER.error("Retraining is not possible: %s", "; ".join(report["problems"]))
    return report


@asynccontextmanager
async def lifespan(_app):
    database.init_db()
    interrupted = database.mark_interrupted_runs()
    if interrupted:
        LOGGER.warning("Marked %d unfinished retrain runs as interrupted", interrupted)
    bootstrap_placeholder()
    log_base_data(retrain.base_data_report())
    yield


app = FastAPI(title="Weather rain trainer API", version="1.0", lifespan=lifespan, root_path=os.environ.get("ROOT_PATH", ""),
              description="Uploads of labelled images, retraining, model metrics and data insights.")


def refuse(error):
    raise HTTPException(getattr(error, "status_code", 400), str(error))


def read_limited(upload, max_mb):
    return upload.file.read(max_mb * 1024 * 1024 + 1)


def folder_size_mb(folder):
    folder = Path(folder)
    total = sum(path.stat().st_size for path in folder.rglob("*") if path.is_file()) if folder.exists() else 0
    return round(total / (1024 * 1024), 2)


def file_size_kb(path):
    return round(Path(path).stat().st_size / 1024, 1) if Path(path).exists() else None


def disk_report():
    usage = shutil.disk_usage(config.DATA_DIR)
    return {"total_mb": round(usage.total / 2**20), "used_mb": round(usage.used / 2**20), "free_mb": round(usage.free / 2**20),
            "min_free_mb": config.MIN_FREE_DISK_MB}


def with_auto_retrain(result):
    return {**result, "auto_retrain_run_id": retrain.maybe_auto_retrain(), "retrain": retrain.get_status()}


@app.get("/health")
def health():
    database.query("SELECT 1")
    base_data = retrain.base_data_report()
    return {"status": "ok" if base_data["ok"] else "degraded", "database": "ok", "retrain_running": retrain.JOB_LOCK.locked(),
            "tensorflow": tensorflow_available(), "base_data": base_data, "hostname": HOSTNAME}


@app.get("/uptime")
def uptime():
    return {"started_at": STARTED_AT, "uptime_seconds": round(time.time() - STARTED, 1), "hostname": HOSTNAME}


@app.get("/system")
def system():
    return {"disk": disk_report(), "uploads": database.uploads_summary(), "uploads_folder_mb": folder_size_mb(config.UPLOADS_DIR),
            "model_files_kb": {path.name: file_size_kb(path) for path in [config.ONNX_PATH, config.H5_PATH, config.METRICS_PATH]},
            "model_name": config.MODEL_NAME, "is_placeholder": config.IS_PLACEHOLDER}


@app.post("/upload")
def upload(files: List[UploadFile] = File(...), label: str = Form(...)):
    try:
        result = uploads.save_uploads([(item.filename, read_limited(item, config.MAX_UPLOAD_MB)) for item in files], label)
    except uploads.UploadRefused as error:
        refuse(error)
    return with_auto_retrain(result)


@app.post("/upload_zip")
def upload_zip(file: UploadFile = File(...)):
    data = read_limited(file, config.MAX_ZIP_MB)
    try:
        if len(data) > config.MAX_ZIP_MB * 1024 * 1024:
            raise uploads.UploadRefused(f"The zip is larger than {config.MAX_ZIP_MB} MB.", status_code=413)
        result = uploads.save_zip_uploads(data)
    except uploads.UploadRefused as error:
        refuse(error)
    return with_auto_retrain(result)


@app.get("/uploads/summary")
def uploads_summary():
    return database.uploads_summary()


@app.post("/retrain", status_code=202)
def start_retrain():
    try:
        run_id = retrain.start_retraining("manual")
    except retrain.RetrainRunning as error:
        raise HTTPException(409, str(error))
    return {"run_id": run_id, "status": "started"}


@app.get("/retrain/status")
def retrain_status():
    return retrain.get_status()


@app.get("/metrics")
def metrics():
    try:
        version = prediction.model_version()
    except FileNotFoundError:
        version = None
    return {"model": version, "metrics": prediction.load_model_metrics(), "class_names": prediction.load_class_names()}


@app.get("/insights")
def data_insights():
    return insights.insight_payload(database.uploads_summary()["by_label"])


@app.get("/runs")
def runs(limit: int = 20):
    return database.list_runs(min(max(limit, 1), 200))
