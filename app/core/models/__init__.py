from app.core.models.base import BaseEmbeddingModel
from app.core.models.bge_m3 import BGEM3Model
from app.core.models.e5_large import MultilingualE5LargeModel
from app.core.models.generic_st import GenericSentenceTransformerModel
from app.core.models.legal_bertimbau import LegalBERTimbauModel
from app.core.models.qwen3 import Qwen3EmbeddingModel
from app.core.models.registry import ModelRegistry

__all__ = [
    "BaseEmbeddingModel",
    "ModelRegistry",
    "BGEM3Model",
    "MultilingualE5LargeModel",
    "LegalBERTimbauModel",
    "Qwen3EmbeddingModel",
    "GenericSentenceTransformerModel",
]

