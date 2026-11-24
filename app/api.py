import logging
import socket
import time
from datetime import datetime, timezone

from fastapi import FastAPI, File, HTTPException, UploadFile

from src import config, prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
STARTED = time.time()
STARTED_AT = datetime.now(timezone.utc).isoformat(timespec="seconds")
HOSTNAME = socket.gethostname()

app = FastAPI(title="Weather rain prediction API", version="1.0",
              description="Classifies a sky photo as cloudy, rainy, shiny or sunrise with an ONNX model.")


def uptime_seconds():
    return round(time.time() - STARTED, 1)


def format_duration(seconds):
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h {minutes}m {secs}s"


def model_status():
    try:
        return prediction.model_version()
    except FileNotFoundError:
        return None


def read_upload(upload, max_mb=config.MAX_UPLOAD_MB):
    return upload.file.read(max_mb * 1024 * 1024 + 1)


@app.get("/health")
def health():
    status = model_status()
    if status is None:
        raise HTTPException(503, f"No model file at {config.ONNX_PATH.name}.")
    return {"status": "ok", "model_loaded": True, "model": status["model_name"], "is_placeholder": status["is_placeholder"],
            "hostname": HOSTNAME}


@app.get("/uptime")
def uptime():
    seconds = uptime_seconds()
    return {"started_at": STARTED_AT, "uptime_seconds": seconds, "uptime": format_duration(seconds), "hostname": HOSTNAME}


@app.get("/model-info")
def model_info():
    status = model_status()
    if status is None:
        raise HTTPException(503, f"No model file at {config.ONNX_PATH.name}.")
    return {**status, "class_names": prediction.load_class_names(), "image_size": config.IMAGE_SIZE,
            "uncertain_threshold": config.UNCERTAIN_THRESHOLD, "metrics": prediction.load_model_metrics(),
            "hostname": HOSTNAME}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    started = time.perf_counter()
    try:
        result = prediction.predict_image_bytes(read_upload(file), file.filename)
    except prediction.InvalidImageError as error:
        raise HTTPException(error.status_code, str(error))
    except FileNotFoundError:
        raise HTTPException(503, "The model file is missing.")
    version = model_status() or {}
    return {**result, "filename": file.filename, "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "hostname": HOSTNAME, "model_sha256": version.get("sha256"), "is_placeholder": version.get("is_placeholder")}
