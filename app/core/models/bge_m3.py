import logging
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.core.models.base import BaseEmbeddingModel
from app.core.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("bge_m3")
class BGEM3Model(BaseEmbeddingModel):
    """Implementação dedicada para o modelo BAAI/bge-m3."""

    def _load_model(self) -> SentenceTransformer:
        logger.info("Carregando BAAI/bge-m3 no device %s...", self.device)
        model_kwargs = {}
        if self.torch_dtype == "float16" and "cuda" in self.device:
            model_kwargs["torch_dtype"] = torch.float16
        elif self.torch_dtype == "bfloat16" and "cuda" in self.device:
            model_kwargs["torch_dtype"] = torch.bfloat16

        model = SentenceTransformer(
            self.model_id,
            device=self.device,
            model_kwargs=model_kwargs if model_kwargs else None,
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

