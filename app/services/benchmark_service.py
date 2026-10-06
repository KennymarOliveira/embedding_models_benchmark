import json
import logging
import multiprocessing
import os
import threading
from queue import Empty
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
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
    get_model_config,
    list_enabled_models,
)
from app.core.extractors.file_extractor import extract_text
from app.core.models.base import BaseEmbeddingModel
from app.core.models.registry import ModelRegistry
from app.core.resource_guard import ResourceLimitExceeded, ResourceSnapshot, SystemResourceGuard
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


class BenchmarkExecutionLogger:
    """Persiste eventos e amostras compactas sem registrar textos ou vetores."""

    def __init__(self, report_dir: Path, benchmark_id: str) -> None:
        self.execution_path = report_dir / "execution.log"
        self.samples_path = report_dir / "resource_samples.jsonl"
        self.benchmark_id = benchmark_id
        self._lock = threading.Lock()

    def _append(self, path: Path, content: str) -> None:
        try:
            with self._lock, path.open("a", encoding="utf-8") as log_file:
                log_file.write(content)
        except OSError:
            logger.exception("Não foi possível gravar o log do benchmark em %s", path)

    def event(self, event: str, model_name: Optional[str] = None, **details: Any) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        fields = {"event": event, **details}
        if model_name:
            fields["model"] = model_name
        serialized = json.dumps(fields, ensure_ascii=False, default=str)
        self._append(self.execution_path, f"{timestamp} | {serialized}\n")

    def resource_sample(self, model_name: str, snapshot: ResourceSnapshot) -> None:
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "benchmark_id": self.benchmark_id,
            "model": model_name,
            "system_ram_percent": snapshot.system_ram_percent,
            "system_ram_used_mb": snapshot.system_ram_used_mb,
            "swap_percent": snapshot.swap_percent,
            "disk_space_used_percent": snapshot.disk_space_used_percent,
            "disk_activity_percent": snapshot.disk_activity_percent,
            "commit_percent": snapshot.commit_percent,
            "paging_pages_per_second": snapshot.paging_pages_per_second,
            "process_ram_mb": snapshot.process_ram_mb,
            "process_ram_percent": snapshot.process_ram_percent,
        }
        self._append(self.samples_path, json.dumps(payload, ensure_ascii=False) + "\n")


def _emit_worker_event(event_queue, phase: str, status: str, **details: Any) -> None:
    """Envia eventos pequenos ao processo pai; nunca bloqueia a inferência."""
    try:
        event_queue.put_nowait(
            {
                "phase": phase,
                "status": status,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                **details,
            }
        )
    except Exception:
        pass


def _model_worker(
    model_class: Type[BaseEmbeddingModel],
    model_config: Dict[str, Any],
    chunk_texts: List[str],
    batch_size: Optional[int],
    result_queue,
    event_queue,
) -> None:
    """Executa um único modelo fora do processo que monitora o computador."""
    model_instance = None
    try:
        model_instance = model_class(model_config)
        _emit_worker_event(event_queue, "load", "started")
        load_time = model_instance.load()
        _emit_worker_event(event_queue, "load", "completed", duration_seconds=load_time)
        _emit_worker_event(event_queue, "warmup", "started")
        warmup_ms = model_instance.warmup()
        _emit_worker_event(event_queue, "warmup", "completed", duration_ms=warmup_ms)
        _emit_worker_event(event_queue, "encode", "started", chunks=len(chunk_texts), batch_size=batch_size)
        _embeddings, chunk_times_ms, meta = model_instance.encode(chunk_texts, batch_size=batch_size)
        _emit_worker_event(event_queue, "encode", "completed", duration_ms=meta["total_inference_time_ms"])
        result_queue.put(
            {
                "ok": True,
                "load_time": load_time,
                "warmup_ms": warmup_ms,
                "chunk_times_ms": chunk_times_ms,
                "meta": meta,
                "model_id": model_instance.model_id,
                "display_name": model_instance.display_name,
                "device": model_instance.device,
                "embedding_dimension": model_instance.get_dimensions(),
            }
        )
    except Exception as exc:
        _emit_worker_event(event_queue, "worker", "failed", error_type=type(exc).__name__, error=str(exc))
        result_queue.put({"ok": False, "error": str(exc)})
    finally:
        if model_instance is not None:
            _emit_worker_event(event_queue, "unload", "started")
            model_instance.unload()
            _emit_worker_event(event_queue, "unload", "completed")


def _run_model_in_isolated_process(
    model_name: str,
    chunk_texts: List[str],
    batch_size: Optional[int],
    resource_guard: SystemResourceGuard,
    execution_logger: Optional[BenchmarkExecutionLogger] = None,
) -> Dict[str, Any]:
    """Executa e, se necessário, termina o processo do modelo isoladamente."""
    model_config = get_model_config(model_name).copy()
    model_class = ModelRegistry.get_model_class(model_config.get("class_key", "generic_st"))
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    event_queue = context.Queue()
    process = context.Process(
        target=_model_worker,
        args=(model_class, model_config, chunk_texts, batch_size, result_queue, event_queue),
        name=f"benchmark-{model_name}",
    )
    process.start()
    resource_guard.monitor_process(process.pid)
    resource_guard.start()
    if execution_logger is not None:
        execution_logger.event("process_started", model_name, pid=process.pid)

    def drain_worker_events() -> None:
        while True:
            try:
                event = event_queue.get_nowait()
            except Empty:
                return
            if execution_logger is not None:
                execution_logger.event("worker_phase", model_name, **event)

    try:
        while process.is_alive():
            process.join(timeout=0.2)
            drain_worker_events()
            resource_guard.checkpoint()

        drain_worker_events()

        try:
            result = result_queue.get(timeout=2)
            if execution_logger is not None:
                execution_logger.event("process_finished", model_name, exit_code=process.exitcode)
            return result
        except Empty as exc:
            if execution_logger is not None:
                execution_logger.event(
                    "process_crashed",
                    model_name,
                    exit_code=process.exitcode,
                    exit_code_hex=f"0x{process.exitcode & 0xFFFFFFFF:08X}" if process.exitcode is not None else None,
                    last_known_phase="Veja o último evento worker_phase acima.",
                )
            raise RuntimeError(
                f"O processo do modelo terminou sem retornar resultado "
                f"(código {process.exitcode}, 0x{process.exitcode & 0xFFFFFFFF:08X})."
            ) from exc
    except ResourceLimitExceeded:
        if execution_logger is not None:
            execution_logger.event("process_interrupted_by_resource_guard", model_name, reason=resource_guard.reason)
        if process.is_alive():
            process.terminate()
            process.join(timeout=3)
        raise
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=3)
        result_queue.close()
        result_queue.join_thread()
        event_queue.close()
        event_queue.join_thread()


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
        lines.append(f"  Aviso de RAM alta:          {res.ram_warning_emitted}")
        lines.append(f"  Pico memória comprometida: {res.commit_peak_percent if res.commit_peak_percent is not None else 'indisponível'}%")
        paging_label = {True: "sim", False: "não", None: "indisponível"}[res.paging_activity_detected]
        lines.append(f"  Paginação do sistema:      {paging_label} (pico: {res.paging_peak_pages_per_second} páginas/s)")
        lines.append(
            "  Pico RAM do processo:      "
            f"{res.process_ram_peak_mb if res.process_ram_peak_mb is not None else 'indisponível'} MB "
            f"({res.process_ram_peak_percent if res.process_ram_peak_percent is not None else 'indisponível'}%)"
        )
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
    report_dir: Optional[Path] = None
    execution_logger: Optional[BenchmarkExecutionLogger] = None

    if save_report:
        try:
            report_dir = RESULTS_DIR / f"{now_str}_ID_{benchmark_id}"
            report_dir.mkdir(parents=True, exist_ok=True)
            execution_logger = BenchmarkExecutionLogger(report_dir, benchmark_id)
            execution_logger.event(
                "benchmark_started",
                filename=filename,
                models=selected_models,
                batch_size=batch_size or DEFAULT_BATCH_SIZE,
                chunks=total_chunks,
                has_cuda=HAS_CUDA,
                device=GPU_DEVICE_NAME,
            )
        except OSError:
            logger.exception("Falha ao preparar a pasta de relatórios do benchmark.")

    for model_name in selected_models:
        logger.info("==> Iniciando benchmark para o modelo: %s", model_name)
        # Cada modelo recebe seu próprio monitor. Assim, os picos e uma
        # eventual interrupção ficam associados ao resultado correto.
        sample_callback = (
            lambda snapshot, name=model_name: execution_logger.resource_sample(name, snapshot)
            if execution_logger is not None
            else None
        )
        resource_guard = SystemResourceGuard(sample_callback=sample_callback)
        try:
            if execution_logger is not None:
                execution_logger.event("model_started", model_name)
            worker_result = _run_model_in_isolated_process(
                model_name,
                chunk_texts,
                batch_size,
                resource_guard,
                execution_logger,
            )
            if not worker_result["ok"]:
                raise RuntimeError(worker_result["error"])

            load_time = worker_result["load_time"]
            warmup_ms = worker_result["warmup_ms"]
            chunk_times_ms = worker_result["chunk_times_ms"]
            meta = worker_result["meta"]

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
                model_id=worker_result["model_id"],
                display_name=worker_result["display_name"],
                device=worker_result["device"],
                embedding_dimension=meta.get("embedding_dimension", worker_result["embedding_dimension"]),
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
                **resource_guard.metrics(),
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
            if execution_logger is not None:
                execution_logger.event("model_completed", model_name)

        except Exception as exc:
            logger.exception("Erro ao avaliar modelo %s: %s", model_name, exc)
            if execution_logger is not None:
                execution_logger.event(
                    "model_failed", model_name, error_type=type(exc).__name__, error=str(exc)
                )
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
                **resource_guard.metrics(),
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
            resource_guard.stop()

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
            if report_dir is None:
                report_dir = RESULTS_DIR / f"{now_str}_ID_{benchmark_id}"
                report_dir.mkdir(parents=True, exist_ok=True)

            summary_json_path = report_dir / "summary.json"
            summary_txt_path = report_dir / "summary.txt"
            response.saved_report_path = str(report_dir)
            summary_json_path.write_text(response.model_dump_json(indent=2), encoding="utf-8")
            summary_txt_path.write_text(format_summary_text(response), encoding="utf-8")

            if execution_logger is not None:
                execution_logger.event("benchmark_completed", report_path=str(report_dir))
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

