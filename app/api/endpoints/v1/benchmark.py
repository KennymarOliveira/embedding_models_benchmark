import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.core.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_STRATEGY
from app.schemas.benchmark import BenchmarkResponse, VectorizeResponse
from app.services.benchmark_service import run_document_benchmark, vectorize_single_document

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/run", response_model=BenchmarkResponse)
async def run_benchmark_endpoint(
    request: Request,
    file: UploadFile = File(...),
    models: Optional[str] = Form(None, description="Modelos a avaliar. Pode enviar checkboxes com múltiplos valores 'models' ou string separada por vírgula."),
    batch_size: Optional[int] = Form(None, description="Tamanho do lote de inferência"),
    chunk_size: int = Form(DEFAULT_CHUNK_SIZE, description="Tamanho de cada chunk em caracteres"),
    chunk_overlap: int = Form(DEFAULT_CHUNK_OVERLAP, description="Sobreposição entre chunks"),
    chunk_strategy: str = Form(DEFAULT_CHUNK_STRATEGY, description="Estratégia: paragraph, fixed, sentence"),
    save_report: bool = Form(True, description="Se True, salva os relatórios de sumário no disco"),
):
    """
    Executa o benchmark completo comparando o desempenho de modelos de embeddings
    locais sobre o documento enviado (PDF, DOCX, DOC, ODT, TXT).
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="O arquivo enviado não possui nome.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="O arquivo enviado está vazio.")

    form = await request.form()
    raw_models = form.getlist("models")
    model_list = []

    for item in raw_models:
        if isinstance(item, str):
            for part in item.split(","):
                part = part.strip()
                if part and part not in model_list:
                    model_list.append(part)

    if not model_list and models:
        for part in models.split(","):
            part = part.strip()
            if part and part not in model_list:
                model_list.append(part)

    selected_models = model_list if model_list else None

    try:
        response = run_document_benchmark(
            filename=file.filename,
            content=content,
            models=selected_models,
            batch_size=batch_size,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_strategy=chunk_strategy,
            save_report=save_report,
        )
        return response
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Falha inesperada durante benchmark do arquivo %s", file.filename)
        raise HTTPException(status_code=500, detail=f"Erro interno no processamento do benchmark: {exc}") from exc


@router.post("/vectorize", response_model=VectorizeResponse)
async def vectorize_endpoint(
    file: UploadFile = File(...),
    model_name: str = Form("bge-m3", description="Nome do modelo configurado (ex: bge-m3)"),
    batch_size: Optional[int] = Form(None, description="Tamanho do lote"),
    chunk_size: int = Form(DEFAULT_CHUNK_SIZE, description="Tamanho do chunk"),
    chunk_overlap: int = Form(DEFAULT_CHUNK_OVERLAP, description="Sobreposição"),
    include_vectors: bool = Form(False, description="Se True, retorna os vetores numéricos no payload JSON"),
):
    """
    Gera embeddings de um documento com um único modelo específico,
    retornando tempos por chunk e opcionalmente a matriz de vetores.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="O arquivo enviado não possui nome.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="O arquivo enviado está vazio.")

    try:
        result = vectorize_single_document(
            filename=file.filename,
            content=content,
            model_name=model_name,
            batch_size=batch_size,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            include_vectors=include_vectors,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Falha ao vetorizar documento %s com modelo %s", file.filename, model_name)
        raise HTTPException(status_code=500, detail=f"Erro interno ao vetorizar: {exc}") from exc

