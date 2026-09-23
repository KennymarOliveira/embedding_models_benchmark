import gc
import logging
import os
import resource
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch

logger = logging.getLogger(__name__)


def get_process_memory_mb() -> float:
    """Retorna o uso atual real de memória RAM do processo em Megabytes."""
    try:
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024**2), 2)
    except Exception:
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(rss_kb / 1024.0, 2)


def get_gpu_memory_mb() -> Dict[str, float]:
    """Retorna o uso atual e pico de memória GPU (VRAM) em Megabytes."""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "peak_mb": 0.0, "reserved_mb": 0.0}

    allocated = torch.cuda.memory_allocated() / (1024**2)
    peak = torch.cuda.max_memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)

    return {
        "allocated_mb": round(allocated, 2),
        "peak_mb": round(peak, 2),
        "reserved_mb": round(reserved, 2),
    }


class BaseEmbeddingModel(ABC):
    """Classe base abstrata para todos os modelos de embedding locais."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_id: str = config.get("model_id", "")
        self.display_name: str = config.get("display_name", self.model_id)
        self.device: str = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.default_batch_size: int = config.get("default_batch_size", 8)
        self.max_seq_length: int = config.get("max_seq_length", 512)
        self.torch_dtype: str = config.get("torch_dtype", "float32")
        self.normalize_embeddings: bool = config.get("normalize_embeddings", True)

        self.model: Any = None
        self.is_loaded: bool = False
        self.dimension: Optional[int] = None
        self.load_time_seconds: float = 0.0

    @abstractmethod
    def _load_model(self) -> Any:
        """Método interno para instanciar o modelo específico."""
        pass

    def load(self) -> float:
        """Carrega o modelo na memória e calcula o tempo de cold-start."""
        if self.is_loaded and self.model is not None:
            return 0.0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        self.ram_before_load_mb = get_process_memory_mb()
        start_time = time.perf_counter()
        self.model = self._load_model()
        self.load_time_seconds = round(time.perf_counter() - start_time, 4)
        self.is_loaded = True

        try:
            sample_emb = self._encode_batch(["teste de dimensão"], batch_size=1)
            self.dimension = int(sample_emb.shape[-1])
        except Exception as exc:
            logger.warning("Não foi possível inferir dimensão automaticamente: %s", exc)

        return self.load_time_seconds

    def unload(self):
        """Descarrega o modelo da memória e libera VRAM/RAM."""
        if self.model is not None:
            del self.model
            self.model = None

        self.is_loaded = False
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def warmup(self, sample_text: str = "Warm-up de teste de inferência.") -> float:
        """Executa um ciclo rápido de aquecimento de GPU/CPU."""
        if not self.is_loaded:
            self.load()

        start_time = time.perf_counter()
        self._encode_batch([sample_text], batch_size=1)
        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.synchronize()
        warmup_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return warmup_ms

    @abstractmethod
    def _encode_batch(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Executa a codificação dos textos pelo modelo."""
        pass

    def encode(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        measure_per_chunk: bool = True,
    ) -> Tuple[np.ndarray, List[float], Dict[str, Any]]:
        """
        Gera embeddings para a lista de textos e calcula métricas completas de tempo e memória.

        Retorna:
        - embeddings: Matriz numpy de embeddings
        - chunk_times_ms: Lista com o tempo de inferência de cada chunk em milissegundos
        - metrics_meta: Dicionário contendo tempos totais, vazão e consumo de memória
        """
        if not self.is_loaded:
            self.load()

        effective_batch_size = batch_size or self.default_batch_size
        num_texts = len(texts)
        if num_texts == 0:
            return np.array([]), [], {}

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        ram_before_mb = get_process_memory_mb()
        gpu_before = get_gpu_memory_mb()

        chunk_times_ms: List[float] = [0.0] * num_texts
        all_embeddings: List[np.ndarray] = []

        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.synchronize()

        overall_start = time.perf_counter()

        if measure_per_chunk and effective_batch_size == 1:
            for idx, text in enumerate(texts):
                t0 = time.perf_counter()
                emb = self._encode_batch([text], batch_size=1)
                if torch.cuda.is_available() and "cuda" in self.device:
                    torch.cuda.synchronize()
                t_diff = (time.perf_counter() - t0) * 1000
                chunk_times_ms[idx] = round(t_diff, 2)
                all_embeddings.append(emb[0])
            final_embeddings = np.array(all_embeddings)
        else:
            for i in range(0, num_texts, effective_batch_size):
                batch_texts = texts[i : i + effective_batch_size]
                batch_count = len(batch_texts)

                t0 = time.perf_counter()
                batch_embs = self._encode_batch(batch_texts, batch_size=effective_batch_size)
                if torch.cuda.is_available() and "cuda" in self.device:
                    torch.cuda.synchronize()
                batch_elapsed_ms = (time.perf_counter() - t0) * 1000

                amortized_ms = round(batch_elapsed_ms / max(1, batch_count), 2)
                for j in range(batch_count):
                    chunk_times_ms[i + j] = amortized_ms

                for emb in batch_embs:
                    all_embeddings.append(emb)

            final_embeddings = np.array(all_embeddings)

        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.synchronize()

        overall_elapsed_seconds = round(time.perf_counter() - overall_start, 4)

        ram_after_mb = get_process_memory_mb()
        gpu_after = get_gpu_memory_mb()

        has_nan = bool(np.isnan(final_embeddings).any())
        has_inf = bool(np.isinf(final_embeddings).any())
        avg_l2_norm = float(np.mean(np.linalg.norm(final_embeddings, axis=1))) if len(final_embeddings) > 0 else 0.0

        base_ram = getattr(self, "ram_before_load_mb", ram_before_mb)
        ram_delta = round(max(0.0, ram_after_mb - base_ram), 2)

        metrics_meta = {
            "total_inference_time_seconds": overall_elapsed_seconds,
            "total_inference_time_ms": round(overall_elapsed_seconds * 1000, 2),
            "ram_before_mb": base_ram,
            "ram_after_mb": ram_after_mb,
            "ram_delta_mb": ram_delta,
            "gpu_allocated_mb": gpu_after["allocated_mb"],
            "gpu_peak_mb": gpu_after["peak_mb"],
            "gpu_reserved_mb": gpu_after["reserved_mb"],
            "has_nan": has_nan,
            "has_inf": has_inf,
            "avg_l2_norm": round(avg_l2_norm, 4),
            "embedding_dimension": int(final_embeddings.shape[1]) if len(final_embeddings) > 0 else 0,
        }

        return final_embeddings, chunk_times_ms, metrics_meta

    def get_dimensions(self) -> int:
        """Retorna a dimensão dos vetores de embedding."""
        if self.dimension is None:
            self.load()
        return self.dimension or 0

    def get_model_info(self) -> Dict[str, Any]:
        """Retorna metadados do modelo."""
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "device": self.device,
            "max_seq_length": self.max_seq_length,
            "dimension": self.dimension,
            "torch_dtype": self.torch_dtype,
            "is_loaded": self.is_loaded,
            "load_time_seconds": self.load_time_seconds,
        }

