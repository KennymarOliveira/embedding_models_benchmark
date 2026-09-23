from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChunkMetric(BaseModel):
    chunk_id: int
    char_count: int
    word_count: int
    est_token_count: int
    inference_time_ms: float


class ModelBenchmarkMetrics(BaseModel):
    model_name: str
    model_id: str
    display_name: str
    device: str
    embedding_dimension: int
    load_time_seconds: float
    warmup_time_ms: float
    total_inference_time_seconds: float
    total_inference_time_ms: float
    avg_chunk_latency_ms: float
    min_chunk_latency_ms: float
    max_chunk_latency_ms: float
    throughput_tokens_per_sec: float
    throughput_chars_per_sec: float
    throughput_words_per_sec: float
    throughput_chunks_per_sec: float
    ram_before_mb: float
    ram_after_mb: float
    ram_delta_mb: float
    gpu_allocated_mb: float
    gpu_peak_mb: float
    gpu_reserved_mb: float
    has_nan: bool
    has_inf: bool
    avg_l2_norm: float
    chunk_metrics: List[ChunkMetric] = Field(default_factory=list)
    error: Optional[str] = None


class DocumentInfo(BaseModel):
    filename: str
    file_type: str
    file_size_bytes: int
    total_characters: int
    total_words: int
    total_estimated_tokens: int
    total_chunks: int


class HardwareInfo(BaseModel):
    has_cuda: bool
    device_name: str
    total_vram_mb: float
    cpu_count: int


class BenchmarkRankings(BaseModel):
    fastest_total_time: Optional[str] = None
    highest_token_throughput: Optional[str] = None
    lowest_latency_per_chunk: Optional[str] = None
    lowest_vram_usage: Optional[str] = None


class BenchmarkResponse(BaseModel):
    benchmark_id: str
    timestamp: str
    document_info: DocumentInfo
    hardware_info: HardwareInfo
    parameters: Dict[str, Any]
    models_evaluated: List[str]
    results: Dict[str, ModelBenchmarkMetrics]
    rankings: BenchmarkRankings
    saved_report_path: Optional[str] = None


class VectorizeResponse(BaseModel):
    model_name: str
    embedding_dimension: int
    total_chunks: int
    total_inference_time_ms: float
    avg_chunk_latency_ms: float
    embeddings: Optional[List[List[float]]] = None
    chunk_metrics: List[ChunkMetric] = Field(default_factory=list)


class ModelConfigResponse(BaseModel):
    key: str
    model_id: str
    display_name: str
    class_key: str
    enabled: bool
    device: str
    torch_dtype: str
    max_seq_length: int
    default_batch_size: int
    description: str

