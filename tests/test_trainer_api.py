import io
import threading
import zipfile

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from app import trainer_api
from src import retrain


@pytest.fixture
def client(service_env):
    with TestClient(trainer_api.app) as test_client:
        yield test_client


def image_files(make_jpeg, count, start=0):
    return [("files", (f"img{i}.jpg", make_jpeg(i), "image/jpeg")) for i in range(start, start + count)]


def wait_for_idle():
    retrain.JOB_LOCK.acquire(timeout=10)
    retrain.JOB_LOCK.release()


def test_health_system_and_uptime(client):
    assert client.get("/health").json()["database"] == "ok"
    system = client.get("/system").json()
    assert system["disk"]["free_mb"] > 0 and system["is_placeholder"] is True
    assert client.get("/uptime").json()["uptime_seconds"] >= 0


def test_upload_saves_and_summarizes(client, make_jpeg):
    response = client.post("/upload", files=image_files(make_jpeg, 3), data={"label": "rainy"})
    assert response.status_code == 200 and response.json()["saved"] == 3
    assert client.get("/uploads/summary").json()["by_label"]["rainy"]["new"] == 3


def test_upload_with_unknown_label_is_refused(client, make_jpeg):
    response = client.post("/upload", files=image_files(make_jpeg, 1), data={"label": "snow"})
    assert response.status_code == 422


def test_upload_zip(client, make_jpeg):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("cloudy/a.jpg", make_jpeg(7))
        archive.writestr("sunrise/b.jpg", make_jpeg(8))
    response = client.post("/upload_zip", files={"file": ("batch.zip", buffer.getvalue(), "application/zip")})
    assert response.status_code == 200 and response.json()["saved"] == 2


def test_retrain_conflict_status_and_runs(client, monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(retrain, "run_retraining", lambda run_id: release.wait(5))
    first = client.post("/retrain")
    assert first.status_code == 202
    assert client.post("/retrain").status_code == 409
    assert client.get("/retrain/status").json()["running"] is True
    release.set()
    wait_for_idle()
    runs = client.get("/runs").json()
    assert runs[0]["id"] == first.json()["run_id"] and runs[0]["trigger"] == "manual"


def test_metrics_and_insights(client):
    metrics = client.get("/metrics").json()
    assert metrics["model"]["is_placeholder"] is True
    insights = client.get("/insights").json()
    assert insights["class_counts"]["train"]["rainy"] == 3 and "test" not in insights["class_counts"]
    assert {"saturation", "sharpness"} <= set(insights["interpretations"]) and len(insights["stories"]) >= 3
