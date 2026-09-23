import logging
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.core.models.base import BaseEmbeddingModel
from app.core.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("e5_large")
class MultilingualE5LargeModel(BaseEmbeddingModel):
    """Implementação dedicada para o modelo intfloat/multilingual-e5-large."""

    def __init__(self, config):
        super().__init__(config)
        self.prefix: str = config.get("prefix", "passage: ")

    def _load_model(self) -> SentenceTransformer:
        logger.info("Carregando intfloat/multilingual-e5-large no device %s...", self.device)
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
        # Prefixo contextual requerido pelo E5 (ex: 'passage: ')
        formatted_texts = [f"{self.prefix}{t}" if self.prefix and not t.startswith(self.prefix) else t for t in texts]
        return self.model.encode(
            formatted_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )

