import io
import pytest
from fastapi.testclient import TestClient

from app.core.models.registry import ModelRegistry
from app.main import app
from tests.test_models import DummyMockModel

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "cuda_available" in data
    assert "gpu_device" in data


def test_list_models_endpoint():
    response = client.get("/api/v1/models/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    keys = [item["key"] for item in data]
    assert "bge-m3" in keys
    assert "qwen3-8b" in keys
    assert "multilingual-e5-large" in keys
    assert "legal-bertimbau-base" in keys


def test_get_single_model_endpoint():
    response = client.get("/api/v1/models/bge-m3")
    assert response.status_code == 200
    data = response.json()
    assert data["key"] == "bge-m3"
    assert data["model_id"] == "BAAI/bge-m3"


def test_get_nonexistent_model_endpoint():
    response = client.get("/api/v1/models/modelo-inexistente")
    assert response.status_code == 404


def test_benchmark_endpoint_with_mock_model(monkeypatch):
    ModelRegistry.register("mock_model")(DummyMockModel)

    import app.core.config as config
    monkeypatch.setitem(
        config.AVAILABLE_MODELS,
        "mock-test",
        {
            "model_id": "test/mock",
            "display_name": "Mock Test",
            "class_key": "mock_model",
            "max_seq_length": 128,
            "default_batch_size": 2,
            "device": "cpu",
            "torch_dtype": "float32",
            "normalize_embeddings": True,
            "enabled": True,
        },
    )

    file_content = b"Primeiro paragrafo de teste.\n\nSegundo paragrafo para avaliacao."
    files = {"file": ("documento_teste.txt", io.BytesIO(file_content), "text/plain")}
    data = {
        "models": "mock-test",
        "chunk_size": 50,
        "chunk_strategy": "paragraph",
        "save_report": False,
    }

    response = client.post("/api/v1/benchmark/run", files=files, data=data)
    assert response.status_code == 200
    payload = response.json()

    assert payload["document_info"]["filename"] == "documento_teste.txt"
    assert "mock-test" in payload["results"]
    res = payload["results"]["mock-test"]
    assert res["error"] is None
    assert res["embedding_dimension"] == 64
    assert len(res["chunk_metrics"]) >= 2
    assert payload["rankings"]["fastest_total_time"] == "mock-test"


def test_vectorize_endpoint_with_mock_model(monkeypatch):
    ModelRegistry.register("mock_model")(DummyMockModel)
    import app.core.config as config
    monkeypatch.setitem(
        config.AVAILABLE_MODELS,
        "mock-test",
        {
            "model_id": "test/mock",
            "display_name": "Mock Test",
            "class_key": "mock_model",
            "max_seq_length": 128,
            "default_batch_size": 2,
            "device": "cpu",
            "torch_dtype": "float32",
            "normalize_embeddings": True,
            "enabled": True,
        },
    )

    file_content = b"Texto para vetorizacao rapida."
    files = {"file": ("teste.txt", io.BytesIO(file_content), "text/plain")}
    data = {
        "model_name": "mock-test",
        "include_vectors": True,
    }

    response = client.post("/api/v1/benchmark/vectorize", files=files, data=data)
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == "mock-test"
    assert payload["embedding_dimension"] == 64
    assert payload["embeddings"] is not None
    assert len(payload["embeddings"]) >= 1


def test_dashboard_home_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Benchmark de Embeddings Local" in response.text
    assert 'name="models"' in response.text
    assert 'type="checkbox"' in response.text


def test_benchmark_endpoint_with_multiple_checkboxes(monkeypatch):
    ModelRegistry.register("mock_model")(DummyMockModel)
    import app.core.config as config
    monkeypatch.setitem(
        config.AVAILABLE_MODELS,
        "mock-a",
        {
            "model_id": "test/mock-a",
            "display_name": "Mock A",
            "class_key": "mock_model",
            "max_seq_length": 128,
            "default_batch_size": 2,
            "device": "cpu",
            "torch_dtype": "float32",
            "normalize_embeddings": True,
            "enabled": True,
        },
    )
    monkeypatch.setitem(
        config.AVAILABLE_MODELS,
        "mock-b",
        {
            "model_id": "test/mock-b",
            "display_name": "Mock B",
            "class_key": "mock_model",
            "max_seq_length": 128,
            "default_batch_size": 2,
            "device": "cpu",
            "torch_dtype": "float32",
            "normalize_embeddings": True,
            "enabled": True,
        },
    )

    file_content = b"Texto para validar selecao por multiplos checkboxes."
    # Simula envio de múltiplos checkboxes via multipart form
    files = [
        ("file", ("doc.txt", file_content, "text/plain")),
        ("models", (None, "mock-a")),
        ("models", (None, "mock-b")),
        ("chunk_size", (None, "100")),
        ("save_report", (None, "false")),
    ]

    response = client.post("/api/v1/benchmark/run", files=files)
    assert response.status_code == 200
    payload = response.json()
    assert "mock-a" in payload["results"]
    assert "mock-b" in payload["results"]


