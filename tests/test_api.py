import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from app import api


@pytest.fixture
def client(service_env):
    return TestClient(api.app)


def test_health_uptime_and_model_info(client):
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["is_placeholder"] is True and health["hostname"]
    assert client.get("/uptime").json()["uptime_seconds"] >= 0
    info = client.get("/model-info").json()
    assert info["class_names"] == ["cloudy", "rainy", "shiny", "sunrise"] and info["image_size"] == 128


def test_predict_returns_label_latency_and_hostname(client, make_jpeg):
    response = client.post("/predict", files={"file": ("sky.jpg", make_jpeg(), "image/jpeg")})
    body = response.json()
    assert response.status_code == 200 and body["label"] in body["probabilities"]
    assert body["latency_ms"] > 0 and body["hostname"] == api.HOSTNAME and body["is_placeholder"] is True


@pytest.mark.parametrize("name, data, status", [("notes.txt", b"hello", 415), ("empty.jpg", b"", 400),
                                                ("broken.jpg", b"\xff\xd8\xff garbage", 415)])
def test_predict_rejects_bad_files(client, name, data, status):
    response = client.post("/predict", files={"file": (name, data, "application/octet-stream")})
    assert response.status_code == status and response.json()["detail"]


def test_missing_model_gives_503(client, service_env):
    service_env["ONNX_PATH"].unlink()
    api.prediction.SESSION_CACHE.clear()
    assert client.get("/health").status_code == 503
