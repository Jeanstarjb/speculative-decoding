"""
Real tests against the real API -- no mocked models. Uses tiny
max_new_tokens so the small CPU-friendly model pair still runs fast.
"""
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["device"] in ("cpu", "cuda")
    assert data["draft_model"] == "distilgpt2"


def test_generate(client):
    response = client.post("/generate", json={"prompt": "Hello world", "max_new_tokens": 5})
    assert response.status_code == 200
    data = response.json()
    assert data["text"].startswith("Hello world")
    assert data["target_forward_passes"] >= 1
    assert 0.0 <= data["acceptance_rate"] <= 1.0
    assert data["naive_text"] is None  # not requested


def test_generate_rejects_empty_prompt(client):
    response = client.post("/generate", json={"prompt": "   "})
    assert response.status_code == 422


def test_generate_rejects_out_of_range_params(client):
    response = client.post("/generate", json={"prompt": "hi", "max_new_tokens": 10_000})
    assert response.status_code == 422


def test_generate_with_naive_comparison(client):
    response = client.post(
        "/generate",
        json={"prompt": "Once upon a time", "max_new_tokens": 5, "compare_to_naive": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["naive_text"] is not None
    assert data["speedup"] is not None
    assert data["speedup"] > 0
