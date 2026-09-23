import numpy as np
import pytest

from app.core.config import AVAILABLE_MODELS, get_model_config
from app.core.models.base import BaseEmbeddingModel
from app.core.models.registry import ModelRegistry


class DummyMockModel(BaseEmbeddingModel):
    """Modelo mock para testes unitários rápidos sem necessidade de download do HuggingFace."""

    def _load_model(self):
        return "mock_loaded"

    def _encode_batch(self, texts, batch_size):
        vectors = np.ones((len(texts), 64), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms


def test_model_registry_registration():
    ModelRegistry.register("mock_model")(DummyMockModel)
    assert "mock_model" in ModelRegistry.list_registered_classes()

    cls = ModelRegistry.get_model_class("mock_model")
    assert cls == DummyMockModel


def test_model_encoding_and_metrics():
    cfg = {
        "model_id": "test/mock-model",
        "display_name": "Test Mock Model",
        "device": "cpu",
        "default_batch_size": 2,
        "max_seq_length": 128,
        "torch_dtype": "float32",
        "normalize_embeddings": True,
    }
    model = DummyMockModel(cfg)
    load_time = model.load()
    assert load_time >= 0.0
    assert model.is_loaded
    assert model.dimension == 64

    texts = ["Texto um para teste.", "Texto dois para teste.", "Texto três para teste."]
    embeddings, chunk_times, meta = model.encode(texts, batch_size=2)

    assert embeddings.shape == (3, 64)
    assert len(chunk_times) == 3
    assert meta["has_nan"] is False
    assert meta["has_inf"] is False
    assert 0.99 <= meta["avg_l2_norm"] <= 1.01
    assert meta["embedding_dimension"] == 64
    assert meta["total_inference_time_seconds"] >= 0.0

    model.unload()
    assert not model.is_loaded
    assert model.model is None


def test_config_lookup():
    bge_cfg = get_model_config("bge-m3")
    assert bge_cfg["model_id"] == "BAAI/bge-m3"

    e5_cfg = get_model_config("multilingual-e5-large")
    assert e5_cfg["prefix"] == "passage: "

    with pytest.raises(ValueError):
        get_model_config("modelo_inexistente_123")

