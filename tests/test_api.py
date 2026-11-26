"""End-to-End API integration tests for FastAPI backend."""
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_tools_list_endpoint():
    response = client.get("/tools")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] >= 4
    tool_names = [t["name"] for t in data["tools"]]
    assert "calculator" in tool_names
    assert "unit_converter" in tool_names


def test_predict_and_execute_endpoint():
    payload = {
        "query": "What is 25 multiplied by 4?",
        "execute_immediately": True
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid_json"] is True
    assert data["is_schema_valid"] is True
    assert data["execution_result"] is not None
    assert data["execution_result"]["success"] is True
    assert "output" in data["execution_result"]


def test_dataset_and_training_status_endpoints():
    r_data = client.get("/dataset/info")
    assert r_data.status_code == 200
    assert "sample_counts" in r_data.json()

    r_train = client.get("/training/status")
    assert r_train.status_code == 200
