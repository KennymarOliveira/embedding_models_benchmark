"""Configurações e constantes visuais do módulo de benchmark."""

from pathlib import Path
from typing import Any, Dict

from app.core.config import AVAILABLE_MODELS, BENCHMARK_DATA_DIR, RESULTS_DIR, SAMPLES_DIR

SUPPORTED_MODELS = list(AVAILABLE_MODELS.keys())

VISUAL_CONFIG: Dict[str, Any] = {
    "palette": ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"],
    "dpi": 300,
    "style": "whitegrid",
    "charts": {
        "latency": {
            "figsize": (11, 5),
            "title": "Comparativo de Tempo e Latência de Inferência",
        },
        "throughput": {
            "figsize": (11, 5),
            "title": "Throughput de Vetorização (Tokens/s e Chunks/s)",
        },
        "memory": {
            "figsize": (11, 5),
            "title": "Consumo de Memória (VRAM GPU e RAM)",
        },
        "dashboard": {
            "figsize": (14, 10),
            "title": "Dashboard Consolidado de Performance dos Modelos de Embedding",
        },
    },
}

