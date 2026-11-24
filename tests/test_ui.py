from pathlib import Path

import pytest
import requests

pytest.importorskip("streamlit")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from app import api, trainer_api

UI_SCRIPT = str(Path(__file__).resolve().parent.parent / "app" / "ui.py")
PAGES = ["Predict", "Insights", "Upload and retrain", "Monitoring"]


@pytest.fixture
def routed_services(service_env, monkeypatch):
    with TestClient(api.app) as api_client, TestClient(trainer_api.app) as trainer_client:
        clients = {"http://localhost:8000": api_client, "http://localhost:8001": trainer_client}

        def fake_request(method, url, timeout=None, **kwargs):
            base = next(prefix for prefix in clients if url.startswith(prefix))
            return clients[base].request(method, url[len(base):], **kwargs)

        monkeypatch.setattr(requests, "request", fake_request)
        yield clients


def run_page(page):
    app = AppTest.from_file(UI_SCRIPT, default_timeout=60)
    app.run()
    app.sidebar.radio[0].set_value(page).run()
    return app


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_errors(routed_services, page):
    app = run_page(page)
    assert not app.exception, app.exception
    assert not app.error, [error.value for error in app.error]


def test_insights_page_has_interpretations(routed_services):
    app = run_page("Insights")
    texts = " ".join(item.value for item in app.markdown)
    assert texts.count("**Interpretation.**") >= 3 and len(app.subheader) >= 4


def test_monitoring_shows_placeholder_warning(routed_services):
    app = run_page("Monitoring")
    assert any("PLACEHOLDER" in item.value for item in app.warning)
    assert [metric.value for metric in app.metric] == ["up", "up"]


def test_unreachable_trainer_shows_error_not_crash(service_env, monkeypatch):
    def refuse(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "request", refuse)
    app = run_page("Insights")
    assert not app.exception and "Cannot reach" in app.error[0].value
