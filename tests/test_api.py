import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["total_documents"] >= 3
    assert data["total_chunks"] >= 3


def test_list_documents():
    response = client.get("/api/documents")
    assert response.status_code == 200
    docs = response.json()
    assert isinstance(docs, list)
    assert len(docs) >= 3


def test_upload_document():
    file_content = b"# Test Document\nTesting FastAPI file upload endpoint."
    response = client.post(
        "/api/documents/upload?strategy=semantic",
        files={"file": ("test_doc.md", file_content, "text/markdown")}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "test_doc.md"
    assert data["was_added_new"] is True


def test_hybrid_search():
    payload = {
        "query": "connection pool limit",
        "mode": "hybrid",
        "top_k": 3
    }
    response = client.post("/api/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "hybrid"
    assert len(data["results"]) > 0


def test_run_investigation():
    payload = {
        "query": "Which system changes caused payment failure in March?",
        "search_mode": "hybrid",
        "top_k": 5,
        "strict_groundedness": True,
        "detect_contradictions": True
    }
    response = client.post("/api/investigate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "answer" in data
    assert "trajectory" in data
    assert len(data["trajectory"]) > 0
