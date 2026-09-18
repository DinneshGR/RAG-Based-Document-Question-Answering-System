"""Integration-style tests for the FastAPI endpoints, with the RAG pipeline
and vector store mocked out so tests run without Ollama or real embeddings."""
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "message" in resp.json()


def test_health_endpoint_reports_status():
    with patch("app.api.routes_query.check_ollama_health", return_value=False):
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["ollama_reachable"] is False


def test_query_endpoint_returns_answer_when_pipeline_succeeds():
    fake_result = {
        "answer": "Revenue grew by 18%. (source: report.pdf, page 12)",
        "sources": [{"document_name": "report.pdf", "page_number": 12, "similarity": 0.9}],
        "retrieved_passages": [
            {"text": "Revenue grew by 18%.", "document_name": "report.pdf", "page_number": 12, "similarity": 0.9}
        ],
    }
    with patch("app.api.routes_query.answer_question", return_value=fake_result):
        resp = client.post("/query", json={"question": "What was the revenue growth?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == fake_result["answer"]
    assert body["sources"][0]["document_name"] == "report.pdf"


def test_query_endpoint_rejects_empty_question():
    resp = client.post("/query", json={"question": ""})
    assert resp.status_code == 422  # pydantic min_length validation


def test_documents_list_endpoint():
    with patch("app.api.routes_documents.list_documents", return_value=[]):
        resp = client.get("/documents")
    assert resp.status_code == 200
    assert resp.json() == {"documents": []}


def test_delete_document_not_found():
    with patch("app.api.routes_documents.delete_document", return_value=0):
        resp = client.delete("/documents/nonexistent-id")
    assert resp.status_code == 404
