"""Script CLI principal para execução do benchmark e geração de métricas visuais."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

# Adiciona a raiz do projeto ao sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import AVAILABLE_MODELS, DEFAULT_BATCH_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_STRATEGY, SAMPLES_DIR, RESULTS_DIR
from app.services.benchmark_service import format_summary_text, run_document_benchmark
from benchmark.visualizer import (
    HAS_MATPLOTLIB,
    plot_dashboard,
    plot_latency_comparison,
    plot_memory_comparison,
    plot_throughput_comparison,
)


def print_banner():
    print("=" * 75)
    print("     LOCAL EMBEDDING MODELS BENCHMARK & EVALUATION ENGINE     ")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(
        description="Executa benchmark comparativo de performance de modelos de embeddings locais."
    )
    parser.add_argument(
        "--document",
        type=str,
        default=None,
        help="Caminho de um documento específico (PDF, DOCX, DOC, ODT, TXT). Se não fornecido, busca em --docs-dir",
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default=str(SAMPLES_DIR),
        help=f"Diretório contendo documentos para benchmark (padrão: {SAMPLES_DIR})",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(AVAILABLE_MODELS.keys()),
        help=f"Modelos a avaliar separados por vírgula (padrão: {','.join(AVAILABLE_MODELS.keys())})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIR),
        help=f"Diretório para salvar os gráficos e relatórios (padrão: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Tamanho do batch de inferência (padrão: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Tamanho máximo de cada chunk em caracteres (padrão: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Sobreposição de caracteres entre chunks (padrão: {DEFAULT_CHUNK_OVERLAP})",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=DEFAULT_CHUNK_STRATEGY,
        choices=["paragraph", "fixed", "sentence"],
        help=f"Estratégia de chunking (padrão: {DEFAULT_CHUNK_STRATEGY})",
    )

    args = parser.parse_args()
    print_banner()

    doc_path: Path
    if args.document:
        doc_path = Path(args.document)
        if not doc_path.exists():
            print(f"[ERRO] Documento não encontrado: {doc_path}")
            sys.exit(1)
    else:
        docs_dir = Path(args.docs_dir)
        supported_exts = [".pdf", ".docx", ".doc", ".odt", ".txt"]
        found_docs = [p for p in docs_dir.glob("*") if p.suffix.lower() in supported_exts]
        if not found_docs:
            print(f"[AVISO] Nenhum documento encontrado em: {docs_dir}")
            print(f"Coloque arquivos {supported_exts} na pasta de samples ou use --document <caminho>")
            sys.exit(0)
        doc_path = found_docs[0]
        print(f"Documento selecionado automaticamente: {doc_path.name}")

    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
    timestamp_folder = datetime.now().strftime("%d_%m_%Y_TIME_%H_%M_%S")
    output_dir = Path(args.output_dir) / timestamp_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Documento de teste:  {doc_path.resolve()}")
    print(f"Diretório de saída:  {output_dir.resolve()}")
    print(f"Modelos escolhidos:  {', '.join(selected_models)}")
    print(f"Batch Size:          {args.batch_size}")
    print(f"Chunking:            Tamanho={args.chunk_size} | Overlap={args.chunk_overlap} | Estratégia={args.strategy}")
    print("-" * 75)

    print("Iniciando extração e vetorização...")
    content = doc_path.read_bytes()

    response = run_document_benchmark(
        filename=doc_path.name,
        content=content,
        models=selected_models,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunk_strategy=args.strategy,
        save_report=False,
    )

    print("\nResultados obtidos por modelo:")
    print("-" * 75)
    for model_name, res in response.results.items():
        if res.error:
            print(f"  [{model_name.upper()}]: FALHA ({res.error})")
        else:
            print(
                f"  [{model_name.upper()}]: "
                f"Tempo: {res.total_inference_time_seconds:.3f}s | "
                f"Latência/Chunk: {res.avg_chunk_latency_ms:.1f}ms | "
                f"Vazão: {res.throughput_tokens_per_sec:.1f} tok/s | "
                f"Dim: {res.embedding_dimension} | "
                f"Pico VRAM: {res.gpu_peak_mb:.1f}MB"
            )

    if HAS_MATPLOTLIB:
        print("\nGerando gráficos comparativos de desempenho...")
        lat_path = output_dir / "latency_comparison.png"
        plot_latency_comparison(response.results, lat_path)
        print(f"  - Gráfico de Latência:    {lat_path.name}")

        th_path = output_dir / "throughput_comparison.png"
        plot_throughput_comparison(response.results, th_path)
        print(f"  - Gráfico de Throughput:  {th_path.name}")

        mem_path = output_dir / "memory_comparison.png"
        plot_memory_comparison(response.results, mem_path)
        print(f"  - Gráfico de Memória:     {mem_path.name}")

        dash_path = output_dir / "metrics_dashboard.png"
        plot_dashboard(response.results, dash_path)
        print(f"  - Painel Dashboard:       {dash_path.name}")
    else:
        print("\n[INFO] Matplotlib não disponível. Gráficos em PNG não foram gerados.")

    summary_txt_path = output_dir / "summary.txt"
    summary_txt_path.write_text(format_summary_text(response), encoding="utf-8")
    print(f"\n==> Resumo textual salvo em: {summary_txt_path.name}")

    summary_json_path = output_dir / "summary.json"
    summary_json_path.write_text(response.model_dump_json(indent=2), encoding="utf-8")
    print(f"==> Resumo JSON salvo em:    {summary_json_path.name}")

    print("\n" + "=" * 75)
    print("Benchmark concluído com sucesso!")
    print("=" * 75)


if __name__ == "__main__":
    main()
