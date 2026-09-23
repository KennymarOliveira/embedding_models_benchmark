import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

import torch

from app.core.chunkers.text_chunker import chunk_text
from app.core.config import (
    AVAILABLE_MODELS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_STRATEGY,
    GPU_DEVICE_NAME,
    GPU_VRAM_TOTAL_MB,
    HAS_CUDA,
    RESULTS_DIR,
    list_enabled_models,
)
from app.core.extractors.file_extractor import extract_text
from app.core.models.registry import ModelRegistry
from app.schemas.benchmark import (
    BenchmarkRankings,
    BenchmarkResponse,
    ChunkMetric,
    DocumentInfo,
    HardwareInfo,
    ModelBenchmarkMetrics,
    VectorizeResponse,
)

logger = logging.getLogger(__name__)


def format_summary_text(response: BenchmarkResponse) -> str:
    """Gera relatório descritivo em texto idêntico ao padrão de benchmark."""
    lines = [
        "=" * 75,
        "          RELATÓRIO CONSOLIDADO DO BENCHMARK DE EMBEDDINGS LOCAL          ",
        "=" * 75,
        f"ID da Execução:   {response.benchmark_id}",
        f"Data/Hora:        {response.timestamp}",
        f"Documento:        {response.document_info.filename} ({response.document_info.file_type.upper()})",
        f"Estatísticas Doc: {response.document_info.total_characters} caracteres | "
        f"{response.document_info.total_words} palavras | "
        f"~{response.document_info.total_estimated_tokens} tokens | "
        f"{response.document_info.total_chunks} chunks",
        f"Hardware:         {response.hardware_info.device_name} (CUDA={response.hardware_info.has_cuda}, VRAM={response.hardware_info.total_vram_mb} MB)",
        "-" * 75,
        "",
    ]

    for name, res in response.results.items():
        lines.append(f"Modelo: {res.display_name} ({res.model_id})")
        lines.append("-" * 50)
        if res.error:
            lines.append(f"  [ERRO NA EXECUÇÃO]: {res.error}")
            lines.append("")
            continue

        lines.append(f"  Dispositivo (Device):       {res.device}")
        lines.append(f"  Dimensão do Embedding:      {res.embedding_dimension}")
        lines.append(f"  Tempo Cold-Start (Load):    {res.load_time_seconds:.3f} s")
        lines.append(f"  Tempo de Warm-up:           {res.warmup_time_ms:.2f} ms")
        lines.append(f"  Tempo Total de Inferência:  {res.total_inference_time_seconds:.4f} s ({res.total_inference_time_ms:.2f} ms)")
        lines.append(f"  Latência Média por Chunk:   {res.avg_chunk_latency_ms:.2f} ms")
        lines.append(f"  Latência Mín / Máx Chunk:   {res.min_chunk_latency_ms:.2f} ms / {res.max_chunk_latency_ms:.2f} ms")
        lines.append(f"  Throughput (Tokens/s):      {res.throughput_tokens_per_sec:.2f} tokens/s")
        lines.append(f"  Throughput (Chars/s):       {res.throughput_chars_per_sec:.2f} chars/s")
        lines.append(f"  Throughput (Chunks/s):      {res.throughput_chunks_per_sec:.2f} chunks/s")
        lines.append(f"  Consumo Delta RAM:          {res.ram_delta_mb:.2f} MB (Final: {res.ram_after_mb:.2f} MB)")
        lines.append(f"  Pico VRAM GPU:              {res.gpu_peak_mb:.2f} MB (Alocada: {res.gpu_allocated_mb:.2f} MB)")
        lines.append(f"  Norma L2 Média:             {res.avg_l2_norm:.4f}")
        lines.append(f"  Sanidade (NaN / Inf):       NaN={res.has_nan}, Inf={res.has_inf}")
        lines.append("")

    lines.append("=" * 75)
    lines.append("RANKING GERAL DE DESEMPENHO")
    lines.append("-" * 50)
    lines.append(f"  Mais Rápido (Tempo Total):  {response.rankings.fastest_total_time or 'N/A'}")
    lines.append(f"  Maior Throughput (Tokens):  {response.rankings.highest_token_throughput or 'N/A'}")
    lines.append(f"  Menor Latência por Chunk:   {response.rankings.lowest_latency_per_chunk or 'N/A'}")
    lines.append(f"  Menor Uso de VRAM:          {response.rankings.lowest_vram_usage or 'N/A'}")
    lines.append("=" * 75)

    return "\n".join(lines)


def run_document_benchmark(
    filename: str,
    content: bytes,
    models: Optional[List[str]] = None,
    batch_size: Optional[int] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    chunk_strategy: str = DEFAULT_CHUNK_STRATEGY,
    save_report: bool = True,
) -> BenchmarkResponse:
    """Executa o benchmark completo sobre o documento para os modelos selecionados."""
    benchmark_id = str(uuid.uuid4())[:8]
    now_str = datetime.now().strftime("%d_%m_%Y_TIME_%H_%M_%S")

    raw_text = extract_text(filename, content)
    if not raw_text.strip():
        raise ValueError(f"O documento '{filename}' não contém texto legível para vetorização.")

    chunks = chunk_text(
        raw_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=chunk_strategy,
    )
    if not chunks:
        chunks = [
            {
                "chunk_id": 0,
                "text": raw_text,
                "char_count": len(raw_text),
                "word_count": len(raw_text.split()),
                "est_token_count": max(1, len(raw_text) // 4),
            }
        ]

    chunk_texts = [c["text"] for c in chunks]
    total_chars = sum(c["char_count"] for c in chunks)
    total_words = sum(c["word_count"] for c in chunks)
    total_tokens = sum(c["est_token_count"] for c in chunks)
    total_chunks = len(chunks)

    ext = filename.split(".")[-1].lower()
    doc_info = DocumentInfo(
        filename=filename,
        file_type=ext,
        file_size_bytes=len(content),
        total_characters=total_chars,
        total_words=total_words,
        total_estimated_tokens=total_tokens,
        total_chunks=total_chunks,
    )

    hw_info = HardwareInfo(
        has_cuda=HAS_CUDA,
        device_name=GPU_DEVICE_NAME,
        total_vram_mb=GPU_VRAM_TOTAL_MB,
        cpu_count=os.cpu_count() or 1,
    )

    selected_models = models or list_enabled_models()
    results: Dict[str, ModelBenchmarkMetrics] = {}

    for model_name in selected_models:
        logger.info("==> Iniciando benchmark para o modelo: %s", model_name)
        try:
            model_instance = ModelRegistry.create_model(model_name)
            load_time = model_instance.load()
            warmup_ms = model_instance.warmup()

            embeddings, chunk_times_ms, meta = model_instance.encode(
                chunk_texts,
                batch_size=batch_size,
            )

            total_inf_sec = max(0.0001, meta["total_inference_time_seconds"])
            tokens_per_sec = round(total_tokens / total_inf_sec, 2)
            chars_per_sec = round(total_chars / total_inf_sec, 2)
            words_per_sec = round(total_words / total_inf_sec, 2)
            chunks_per_sec = round(total_chunks / total_inf_sec, 2)

            chunk_metrics_list = []
            for idx, c in enumerate(chunks):
                t_ms = chunk_times_ms[idx] if idx < len(chunk_times_ms) else 0.0
                chunk_metrics_list.append(
                    ChunkMetric(
                        chunk_id=c["chunk_id"],
                        char_count=c["char_count"],
                        word_count=c["word_count"],
                        est_token_count=c["est_token_count"],
                        inference_time_ms=t_ms,
                    )
                )

            valid_times = [cm.inference_time_ms for cm in chunk_metrics_list if cm.inference_time_ms > 0]
            avg_latency = round(sum(valid_times) / len(valid_times), 2) if valid_times else 0.0
            min_latency = round(min(valid_times), 2) if valid_times else 0.0
            max_latency = round(max(valid_times), 2) if valid_times else 0.0

            metric_entry = ModelBenchmarkMetrics(
                model_name=model_name,
                model_id=model_instance.model_id,
                display_name=model_instance.display_name,
                device=model_instance.device,
                embedding_dimension=meta.get("embedding_dimension", model_instance.get_dimensions()),
                load_time_seconds=load_time,
                warmup_time_ms=warmup_ms,
                total_inference_time_seconds=meta["total_inference_time_seconds"],
                total_inference_time_ms=meta["total_inference_time_ms"],
                avg_chunk_latency_ms=avg_latency,
                min_chunk_latency_ms=min_latency,
                max_chunk_latency_ms=max_latency,
                throughput_tokens_per_sec=tokens_per_sec,
                throughput_chars_per_sec=chars_per_sec,
                throughput_words_per_sec=words_per_sec,
                throughput_chunks_per_sec=chunks_per_sec,
                ram_before_mb=meta["ram_before_mb"],
                ram_after_mb=meta["ram_after_mb"],
                ram_delta_mb=meta["ram_delta_mb"],
                gpu_allocated_mb=meta["gpu_allocated_mb"],
                gpu_peak_mb=meta["gpu_peak_mb"],
                gpu_reserved_mb=meta["gpu_reserved_mb"],
                has_nan=meta["has_nan"],
                has_inf=meta["has_inf"],
                avg_l2_norm=meta["avg_l2_norm"],
                chunk_metrics=chunk_metrics_list,
                error=None,
            )
            results[model_name] = metric_entry

        except Exception as exc:
            logger.exception("Erro ao avaliar modelo %s: %s", model_name, exc)
            results[model_name] = ModelBenchmarkMetrics(
                model_name=model_name,
                model_id=AVAILABLE_MODELS.get(model_name, {}).get("model_id", model_name),
                display_name=AVAILABLE_MODELS.get(model_name, {}).get("display_name", model_name),
                device="unknown",
                embedding_dimension=0,
                load_time_seconds=0.0,
                warmup_time_ms=0.0,
                total_inference_time_seconds=0.0,
                total_inference_time_ms=0.0,
                avg_chunk_latency_ms=0.0,
                min_chunk_latency_ms=0.0,
                max_chunk_latency_ms=0.0,
                throughput_tokens_per_sec=0.0,
                throughput_chars_per_sec=0.0,
                throughput_words_per_sec=0.0,
                throughput_chunks_per_sec=0.0,
                ram_before_mb=0.0,
                ram_after_mb=0.0,
                ram_delta_mb=0.0,
                gpu_allocated_mb=0.0,
                gpu_peak_mb=0.0,
                gpu_reserved_mb=0.0,
                has_nan=False,
                has_inf=False,
                avg_l2_norm=0.0,
                chunk_metrics=[],
                error=str(exc),
            )
        finally:
            # Libera VRAM/RAM explicitamente entre modelos
            if "model_instance" in locals():
                model_instance.unload()

    successful_results = {k: v for k, v in results.items() if v.error is None}
    rankings = BenchmarkRankings()
    if successful_results:
        rankings.fastest_total_time = min(
            successful_results.keys(),
            key=lambda k: successful_results[k].total_inference_time_seconds,
        )
        rankings.highest_token_throughput = max(
            successful_results.keys(),
            key=lambda k: successful_results[k].throughput_tokens_per_sec,
        )
        rankings.lowest_latency_per_chunk = min(
            successful_results.keys(),
            key=lambda k: successful_results[k].avg_chunk_latency_ms,
        )
        rankings.lowest_vram_usage = min(
            successful_results.keys(),
            key=lambda k: successful_results[k].gpu_peak_mb,
        )

    response = BenchmarkResponse(
        benchmark_id=benchmark_id,
        timestamp=now_str,
        document_info=doc_info,
        hardware_info=hw_info,
        parameters={
            "batch_size": batch_size or DEFAULT_BATCH_SIZE,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunk_strategy": chunk_strategy,
        },
        models_evaluated=selected_models,
        results=results,
        rankings=rankings,
    )

    if save_report:
        try:
            report_dir = RESULTS_DIR / f"{now_str}_ID_{benchmark_id}"
            report_dir.mkdir(parents=True, exist_ok=True)

            summary_json_path = report_dir / "summary.json"
            summary_json_path.write_text(response.model_dump_json(indent=2), encoding="utf-8")

            summary_txt_path = report_dir / "summary.txt"
            summary_txt_path.write_text(format_summary_text(response), encoding="utf-8")

            response.saved_report_path = str(report_dir)
            logger.info("Relatório de benchmark salvo em: %s", report_dir)
        except Exception as exc:
            logger.error("Falha ao salvar relatório no disco: %s", exc)

    return response


def vectorize_single_document(
    filename: str,
    content: bytes,
    model_name: str,
    batch_size: Optional[int] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    chunk_strategy: str = DEFAULT_CHUNK_STRATEGY,
    include_vectors: bool = False,
) -> VectorizeResponse:
    """Executa a vetorização com um único modelo e retorna métricas e vetores."""
    raw_text = extract_text(filename, content)
    chunks = chunk_text(
        raw_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=chunk_strategy,
    )
    if not chunks:
        chunks = [{"chunk_id": 0, "text": raw_text, "char_count": len(raw_text), "word_count": len(raw_text.split()), "est_token_count": 1}]

    chunk_texts = [c["text"] for c in chunks]

    model_instance = ModelRegistry.create_model(model_name)
    try:
        model_instance.load()
        embeddings, chunk_times_ms, meta = model_instance.encode(
            chunk_texts,
            batch_size=batch_size,
        )

        chunk_metrics = [
            ChunkMetric(
                chunk_id=c["chunk_id"],
                char_count=c["char_count"],
                word_count=c["word_count"],
                est_token_count=c["est_token_count"],
                inference_time_ms=chunk_times_ms[idx] if idx < len(chunk_times_ms) else 0.0,
            )
            for idx, c in enumerate(chunks)
        ]

        avg_latency = (
            round(sum(cm.inference_time_ms for cm in chunk_metrics) / len(chunk_metrics), 2)
            if chunk_metrics
            else 0.0
        )

        return VectorizeResponse(
            model_name=model_name,
            embedding_dimension=meta.get("embedding_dimension", model_instance.get_dimensions()),
            total_chunks=len(chunks),
            total_inference_time_ms=meta["total_inference_time_ms"],
            avg_chunk_latency_ms=avg_latency,
            embeddings=embeddings.tolist() if include_vectors else None,
            chunk_metrics=chunk_metrics,
        )
    finally:
        model_instance.unload()

