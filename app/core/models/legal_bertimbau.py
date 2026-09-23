import logging
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, models

from app.core.models.base import BaseEmbeddingModel
from app.core.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("legal_bertimbau")
class LegalBERTimbauModel(BaseEmbeddingModel):
    """Implementação para o modelo rufimelo/Legal-BERTimbau-base com pooling configurável."""

    def __init__(self, config):
        super().__init__(config)
        self.pooling_mode: str = config.get("pooling", "mean")

    def _load_model(self) -> SentenceTransformer:
        logger.info(
            "Carregando rufimelo/Legal-BERTimbau-base (pooling=%s) no device %s...",
            self.pooling_mode,
            self.device,
        )
        try:
            model = SentenceTransformer(self.model_id, device=self.device)
        except Exception:
            # Modelos Masked LM sem módulos ST requerem pooling explícito
            logger.info("Construindo módulo Transformer + Pooling para %s", self.model_id)
            word_embedding_model = models.Transformer(
                self.model_id,
                max_seq_length=self.max_seq_length,
            )
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(),
                pooling_mode=self.pooling_mode,
            )
            model = SentenceTransformer(
                modules=[word_embedding_model, pooling_model],
                device=self.device,
            )

        if self.max_seq_length:
            model.max_seq_length = self.max_seq_length
        return model

    def _encode_batch(self, texts: List[str], batch_size: int) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )

