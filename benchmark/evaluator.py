from pathlib import Path
from typing import List, Optional

from app.schemas.benchmark import BenchmarkResponse
from app.services.benchmark_service import run_document_benchmark


def evaluate_document_file(
    file_path: Path,
    models: Optional[List[str]] = None,
    batch_size: Optional[int] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    chunk_strategy: str = "paragraph",
    save_report: bool = True,
) -> BenchmarkResponse:
    """Executa o benchmark sobre um arquivo de documento físico no disco."""
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    content = file_path.read_bytes()
    return run_document_benchmark(
        filename=file_path.name,
        content=content,
        models=models,
        batch_size=batch_size,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_strategy=chunk_strategy,
        save_report=save_report,
    )

