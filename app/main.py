import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

import app.core.models  # Registra subclasses no ModelRegistry
from app.api.endpoints.v1 import benchmark, models
from app.core.config import GPU_DEVICE_NAME, GPU_VRAM_TOTAL_MB, HAS_CUDA
from app.ui.dashboard import get_dashboard_html

app = FastAPI(
    title="Local Embedding Models Benchmark API",
    description="API para benchmark e vetorização de documentos (PDF, DOCX, DOC, ODT, TXT) com modelos de embeddings locais.",
    version="0.1.0",
)

app.include_router(benchmark.router, prefix="/api/v1/benchmark", tags=["Benchmark & Vectorization"])
app.include_router(models.router, prefix="/api/v1/models", tags=["Models"])


@app.get("/", response_class=HTMLResponse)
def dashboard_view():
    """Painel interativo Web com caixas de seleção dos modelos e upload de documentos."""
    return HTMLResponse(content=get_dashboard_html())


@app.get("/health")
def health_check():
    """Endpoint de checagem de saúde e diagnóstico de hardware."""
    return {
        "status": "ok",
        "cuda_available": HAS_CUDA,
        "gpu_device": GPU_DEVICE_NAME,
        "vram_total_mb": GPU_VRAM_TOTAL_MB,
        "cpu_cores": os.cpu_count(),
    }

