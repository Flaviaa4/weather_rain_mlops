import threading

import anyio
import pytest

httpx = pytest.importorskip("httpx")
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


class BlockingPredictor:
    def __init__(self):
        self.release, self.lock, self.running, self.peak = threading.Event(), threading.Lock(), 0, 0

    def __call__(self, data, filename=None):
        with self.lock:
            self.running += 1
            self.peak = max(self.peak, self.running)
        self.release.wait(timeout=10)
        with self.lock:
            self.running -= 1
        return {"label": "rainy", "confidence": 0.9, "probabilities": {"rainy": 0.9}, "uncertain": False}


async def wait_for_queue(limiter, running, waiting):
    with anyio.fail_after(5):
        while limiter.borrowed_tokens < running or limiter.statistics().tasks_waiting < waiting:
            await anyio.sleep(0.01)


async def flood_with_health_check(predictor, image, total):
    responses = []
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as http:
        async def send():
            responses.append(await http.post("/predict", files={"file": ("sky.jpg", image, "image/jpeg")}))

        async with anyio.create_task_group() as group:
            for _ in range(total):
                group.start_soon(send)
            await wait_for_queue(api.PREDICTION_LIMITER, 2, total - 2)
            with anyio.fail_after(5):
                health = await http.get("/health")
            predictor.release.set()
    return responses, health


def test_predictions_wait_behind_the_cap_and_health_does_not(service_env, make_jpeg, monkeypatch):
    predictor = BlockingPredictor()
    monkeypatch.setattr(api, "PREDICTION_LIMITER", anyio.CapacityLimiter(2))
    monkeypatch.setattr(api.prediction, "predict_image_bytes", predictor)
    responses, health = anyio.run(flood_with_health_check, predictor, make_jpeg(), 6)
    assert health.status_code == 200 and predictor.peak == 2
    assert [response.status_code for response in responses] == [200] * 6


def test_cap_defaults_to_eight():
    assert api.config.MAX_CONCURRENT_PREDICTIONS == 8 and api.PREDICTION_LIMITER.total_tokens == 8


def test_oversized_upload_is_rejected_before_decoding(client, monkeypatch):
    monkeypatch.setattr(api.config, "MAX_UPLOAD_MB", 1)
    monkeypatch.setattr(api, "predict_upload", lambda upload: pytest.fail("the body was decoded"))
    response = client.post("/predict", files={"file": ("big.jpg", b"\xff" * (2 * 1024 * 1024), "image/jpeg")})
    assert response.status_code == 413 and "larger than 1 MB" in response.json()["detail"]
