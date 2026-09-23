import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.models.base import BaseEmbeddingModel
from app.core.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("generic_st")
class GenericSentenceTransformerModel(BaseEmbeddingModel):
    """Wrapper genérico para qualquer modelo compatível com SentenceTransformers."""

    def _load_model(self) -> SentenceTransformer:
        logger.info("Carregando modelo genérico SentenceTransformer: %s", self.model_id)
        model = SentenceTransformer(self.model_id, device=self.device)
        if self.max_seq_length:
            model.max_seq_length = self.max_seq_length
        return model

    def _encode_batch(self, texts: List[str], batch_size: int) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return embeddings

