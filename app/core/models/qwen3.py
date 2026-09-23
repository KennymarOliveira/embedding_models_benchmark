import logging
from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.core.models.base import BaseEmbeddingModel
from app.core.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("qwen3")
class Qwen3EmbeddingModel(BaseEmbeddingModel):
    """Implementação dedicada para o modelo Qwen/Qwen3-Embedding-8B."""

    def __init__(self, config):
        super().__init__(config)
        self.prompt_name: Optional[str] = config.get("prompt_name")
        self.output_dimension: Optional[int] = config.get("output_dimension")

    def _load_model(self) -> SentenceTransformer:
        logger.info(
            "Carregando Qwen/Qwen3-Embedding-8B no device %s (dtype=%s)...",
            self.device,
            self.torch_dtype,
        )

        model_kwargs = {}
        if self.torch_dtype == "bfloat16" and "cuda" in self.device:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif self.torch_dtype == "float16" and "cuda" in self.device:
            model_kwargs["torch_dtype"] = torch.float16

        if self.config.get("device_map"):
            model_kwargs["device_map"] = self.config["device_map"]

        try:
            model = SentenceTransformer(
                self.model_id,
                device=self.device if not model_kwargs.get("device_map") else None,
                model_kwargs=model_kwargs if model_kwargs else None,
                tokenizer_kwargs={"padding_side": "left"},
            )
        except torch.cuda.OutOfMemoryError as oom_err:
            logger.error("VRAM insuficiente para Qwen 8B na GPU. Tentando fallback para CPU...")
            self.device = "cpu"
            model = SentenceTransformer(
                self.model_id,
                device="cpu",
                tokenizer_kwargs={"padding_side": "left"},
            )

        if self.max_seq_length:
            model.max_seq_length = self.max_seq_length

        if self.output_dimension:
            model.truncate_dim = self.output_dimension

        return model

    def _encode_batch(self, texts: List[str], batch_size: int) -> np.ndarray:
        encode_kwargs = {
            "batch_size": batch_size,
            "show_progress_bar": False,
            "normalize_embeddings": self.normalize_embeddings,
            "convert_to_numpy": True,
        }
        if self.prompt_name and hasattr(self.model, "prompts") and self.prompt_name in self.model.prompts:
            encode_kwargs["prompt_name"] = self.prompt_name

        return self.model.encode(texts, **encode_kwargs)

