import logging
import socket
import time
from datetime import datetime, timezone

import anyio
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from src import config, prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
STARTED = time.time()
STARTED_AT = datetime.now(timezone.utc).isoformat(timespec="seconds")
HOSTNAME = socket.gethostname()
MULTIPART_OVERHEAD_BYTES = 64 * 1024
# predictions get their own thread limiter, so waiting requests hold no thread and /health keeps the default pool
PREDICTION_LIMITER = anyio.CapacityLimiter(config.MAX_CONCURRENT_PREDICTIONS)

app = FastAPI(title="Weather rain prediction API", version="1.0",
              description="Classifies a sky photo as cloudy, rainy, shiny or sunrise with an ONNX model.")


def body_too_large(request):
    length = request.headers.get("content-length", "")
    return length.isdigit() and int(length) > config.MAX_UPLOAD_MB * 1024 * 1024 + MULTIPART_OVERHEAD_BYTES


@app.middleware("http")
async def reject_oversized_predictions(request: Request, call_next):
    if request.url.path == "/predict" and body_too_large(request):
        return JSONResponse({"detail": f"The file is larger than {config.MAX_UPLOAD_MB} MB."}, status_code=413)
    return await call_next(request)


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


def predict_upload(upload):
    return prediction.predict_image_bytes(read_upload(upload), upload.filename)


async def limited_predict(upload):
    return await anyio.to_thread.run_sync(predict_upload, upload, limiter=PREDICTION_LIMITER)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    started = time.perf_counter()
    try:
        result = await limited_predict(file)
    except prediction.InvalidImageError as error:
        raise HTTPException(error.status_code, str(error))
    except FileNotFoundError:
        raise HTTPException(503, "The model file is missing.")
    version = model_status() or {}
    return {**result, "filename": file.filename, "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "hostname": HOSTNAME, "model_sha256": version.get("sha256"), "is_placeholder": version.get("is_placeholder")}
