import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from unittest.mock import patch, MagicMock

client = TestClient(app)

@patch('backend.app.main.cache')
@patch('backend.app.main.monitoring')
@patch('backend.app.main.draft_model')
@patch('backend.app.main.target_model')
def test_health_check(mock_target, mock_draft, mock_monitoring, mock_cache):
    mock_cache.redis_conn.ping.return_value = True
    mock_monitoring.get_status.return_value = 'ok'
    mock_draft.get_status.return_value = 'ok'
    mock_target.get_status.return_value = 'ok'

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert all(v == 'ok' for v in data.values())

@patch('backend.app.main.SpeculativeDecoder')
def test_generation_endpoint(mock_decoder):
    mock_instance = mock_decoder.return_value
    mock_instance.generate.return_value = [100, 200, 300]
    
    response = client.post(
        "/generate",
        json={"prompt": "Test input", "max_length": 50, "temperature": 0.7}
    )
    assert response.status_code == 200
    assert 'generated_text' in response.json()
    response = client.post("/generate", json={"prompt": ""})
    assert response.status_code == 422
