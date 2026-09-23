import os
from pathlib import Path
from typing import Any, Dict

import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent
BENCHMARK_DATA_DIR = BASE_DIR / "benchmark" / "data"
RESULTS_DIR = BENCHMARK_DATA_DIR / "results"
SAMPLES_DIR = BENCHMARK_DATA_DIR / "sample_documents"

HAS_CUDA = torch.cuda.is_available()
DEFAULT_DEVICE = "cuda" if HAS_CUDA else "cpu"
GPU_DEVICE_NAME = torch.cuda.get_device_name(0) if HAS_CUDA else "CPU Only"
GPU_VRAM_TOTAL_MB = (
    round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 2)
    if HAS_CUDA
    else 0.0
)

DEFAULT_BATCH_SIZE = 8
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_CHUNK_STRATEGY = "paragraph"
DEFAULT_NORMALIZE_EMBEDDINGS = True
WARMUP_RUNS = 1

AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
    "bge-m3": {
        "model_id": "BAAI/bge-m3",
        "display_name": "BAAI BGE-M3",
        "class_key": "bge_m3",
        "max_seq_length": 8192,
        "default_batch_size": 8,
        "device": DEFAULT_DEVICE,
        "torch_dtype": "float16" if HAS_CUDA else "float32",
        "normalize_embeddings": True,
        "enabled": True,
        "description": "Modelo multilíngue de alta capacidade para textos longos (até 8192 tokens).",
    },
    "qwen3-8b": {
        "model_id": "Qwen/Qwen3-Embedding-8B",
        "display_name": "Qwen3 Embedding 8B",
        "class_key": "qwen3",
        "max_seq_length": 8192,
        "default_batch_size": 2,
        "device": DEFAULT_DEVICE,
        "torch_dtype": "bfloat16" if HAS_CUDA else "float32",
        "normalize_embeddings": True,
        "enabled": True,
        "description": "Modelo de 8B parâmetros com suporte a mais de 100 idiomas e dimensão até 4096.",
    },
    "multilingual-e5-large": {
        "model_id": "intfloat/multilingual-e5-large",
        "display_name": "Multilingual E5 Large",
        "class_key": "e5_large",
        "prefix": "passage: ",
        "max_seq_length": 512,
        "default_batch_size": 8,
        "device": DEFAULT_DEVICE,
        "torch_dtype": "float16" if HAS_CUDA else "float32",
        "normalize_embeddings": True,
        "enabled": True,
        "description": "Modelo denso multilíngue de 24 camadas com 1024 dimensões e prefixo de contexto.",
    },
    "legal-bertimbau-base": {
        "model_id": "rufimelo/Legal-BERTimbau-base",
        "display_name": "Legal BERTimbau Base",
        "class_key": "legal_bertimbau",
        "pooling": "mean",
        "max_seq_length": 512,
        "default_batch_size": 8,
        "device": DEFAULT_DEVICE,
        "torch_dtype": "float32",
        "normalize_embeddings": True,
        "enabled": True,
        "description": "BERTimbau especializado no domínio jurídico brasileiro com mean pooling.",
    },
}


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Retorna a configuração do modelo normalizando o nome."""
    normalized_name = model_name.strip().lower()
    if normalized_name in AVAILABLE_MODELS:
        return AVAILABLE_MODELS[normalized_name]

    for key, cfg in AVAILABLE_MODELS.items():
        if normalized_name == cfg["model_id"].lower() or normalized_name == cfg["display_name"].lower():
            return cfg

    raise ValueError(
        f"Modelo '{model_name}' não encontrado no config.py. "
        f"Modelos disponíveis: {list(AVAILABLE_MODELS.keys())}"
    )


def list_enabled_models() -> list[str]:
    """Retorna os identificadores dos modelos ativos."""
    return [name for name, cfg in AVAILABLE_MODELS.items() if cfg.get("enabled", True)]

