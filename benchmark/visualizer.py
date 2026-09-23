import logging
from pathlib import Path
from typing import Dict

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from benchmark.config import VISUAL_CONFIG
from app.schemas.benchmark import ModelBenchmarkMetrics

logger = logging.getLogger(__name__)


def _set_chart_style():
    if not HAS_MATPLOTLIB:
        return
    sns.set_theme(style=VISUAL_CONFIG.get("style", "whitegrid"))
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"


def plot_latency_comparison(results: Dict[str, ModelBenchmarkMetrics], output_path: Path):
    """Gera gráfico comparando tempo total de inferência e latência média por chunk."""
    if not HAS_MATPLOTLIB:
        return

    _set_chart_style()
    valid_results = {k: v for k, v in results.items() if v.error is None}
    if not valid_results:
        return

    names = [v.display_name for v in valid_results.values()]
    total_times = [v.total_inference_time_seconds for v in valid_results.values()]
    avg_latencies = [v.avg_chunk_latency_ms for v in valid_results.values()]

    fig, axes = plt.subplots(1, 2, figsize=VISUAL_CONFIG["charts"]["latency"]["figsize"])
    palette = VISUAL_CONFIG["palette"]

    bars1 = axes[0].bar(names, total_times, color=palette[: len(names)], edgecolor="black", alpha=0.85)
    axes[0].set_title("Tempo Total de Inferência (s)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Segundos (s)")
    axes[0].tick_params(axis="x", rotation=20)
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, yval + (yval * 0.02), f"{yval:.3f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")

    bars2 = axes[1].bar(names, avg_latencies, color=palette[: len(names)], edgecolor="black", alpha=0.85)
    axes[1].set_title("Latência Média por Chunk (ms)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Milissegundos (ms)")
    axes[1].tick_params(axis="x", rotation=20)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, yval + (yval * 0.02), f"{yval:.1f}ms", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.suptitle("Comparativo de Desempenho de Latência", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=VISUAL_CONFIG["dpi"], bbox_inches="tight")
    plt.close(fig)


def plot_throughput_comparison(results: Dict[str, ModelBenchmarkMetrics], output_path: Path):
    """Gera gráfico comparando Throughput em Tokens/s e Chunks/s."""
    if not HAS_MATPLOTLIB:
        return

    _set_chart_style()
    valid_results = {k: v for k, v in results.items() if v.error is None}
    if not valid_results:
        return

    names = [v.display_name for v in valid_results.values()]
    tokens_sec = [v.throughput_tokens_per_sec for v in valid_results.values()]
    chunks_sec = [v.throughput_chunks_per_sec for v in valid_results.values()]

    fig, axes = plt.subplots(1, 2, figsize=VISUAL_CONFIG["charts"]["throughput"]["figsize"])
    palette = VISUAL_CONFIG["palette"]

    bars1 = axes[0].bar(names, tokens_sec, color=palette[: len(names)], edgecolor="black", alpha=0.85)
    axes[0].set_title("Vazão: Tokens por Segundo (tokens/s)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Tokens / seg")
    axes[0].tick_params(axis="x", rotation=20)
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, yval + (yval * 0.02), f"{yval:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    bars2 = axes[1].bar(names, chunks_sec, color=palette[: len(names)], edgecolor="black", alpha=0.85)
    axes[1].set_title("Vazão: Chunks por Segundo (chunks/s)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Chunks / seg")
    axes[1].tick_params(axis="x", rotation=20)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, yval + (yval * 0.02), f"{yval:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.suptitle("Comparativo de Throughput de Vetorização", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=VISUAL_CONFIG["dpi"], bbox_inches="tight")
    plt.close(fig)


def plot_memory_comparison(results: Dict[str, ModelBenchmarkMetrics], output_path: Path):
    """Gera gráfico comparando o consumo de RAM e pico de VRAM."""
    if not HAS_MATPLOTLIB:
        return

    _set_chart_style()
    valid_results = {k: v for k, v in results.items() if v.error is None}
    if not valid_results:
        return

    names = [v.display_name for v in valid_results.values()]
    vram_peak = [v.gpu_peak_mb for v in valid_results.values()]
    ram_delta = [v.ram_delta_mb for v in valid_results.values()]

    fig, axes = plt.subplots(1, 2, figsize=VISUAL_CONFIG["charts"]["memory"]["figsize"])
    palette = VISUAL_CONFIG["palette"]

    bars1 = axes[0].bar(names, vram_peak, color=palette[: len(names)], edgecolor="black", alpha=0.85)
    axes[0].set_title("Pico de VRAM da GPU (MB)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Megabytes (MB)")
    axes[0].tick_params(axis="x", rotation=20)
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, yval + (yval * 0.02), f"{yval:.1f} MB", ha="center", va="bottom", fontsize=10, fontweight="bold")

    bars2 = axes[1].bar(names, ram_delta, color=palette[: len(names)], edgecolor="black", alpha=0.85)
    axes[1].set_title("Variação de RAM do Processo (Δ RAM MB)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Megabytes (MB)")
    axes[1].tick_params(axis="x", rotation=20)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, yval + (yval * 0.02), f"{yval:.1f} MB", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.suptitle("Consumo de Recursos de Memória", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=VISUAL_CONFIG["dpi"], bbox_inches="tight")
    plt.close(fig)


def plot_dashboard(results: Dict[str, ModelBenchmarkMetrics], output_path: Path):
    """Gera um painel dashboard 2x2 com todas as métricas consolidadas."""
    if not HAS_MATPLOTLIB:
        return

    _set_chart_style()
    valid_results = {k: v for k, v in results.items() if v.error is None}
    if not valid_results:
        return

    names = [v.display_name for v in valid_results.values()]
    palette = VISUAL_CONFIG["palette"]

    fig, axes = plt.subplots(2, 2, figsize=VISUAL_CONFIG["charts"]["dashboard"]["figsize"])

    times = [v.total_inference_time_seconds for v in valid_results.values()]
    b0 = axes[0, 0].bar(names, times, color=palette[: len(names)], edgecolor="black", alpha=0.85)
    axes[0, 0].set_title("Tempo Total de Inferência (s)", fontweight="bold")
    axes[0, 0].set_ylabel("Segundos")
    axes[0, 0].tick_params(axis="x", rotation=20)
    for bar in b0:
        y = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, y, f"{y:.2f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")

    tokens = [v.throughput_tokens_per_sec for v in valid_results.values()]
    b1 = axes[0, 1].bar(names, tokens, color=palette[: len(names)], edgecolor="black", alpha=0.85)
    axes[0, 1].set_title("Throughput (Tokens/s)", fontweight="bold")
    axes[0, 1].set_ylabel("Tokens / s")
    axes[0, 1].tick_params(axis="x", rotation=20)
    for bar in b1:
        y = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, y, f"{y:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    lat = [v.avg_chunk_latency_ms for v in valid_results.values()]
    b2 = axes[1, 0].bar(names, lat, color=palette[: len(names)], edgecolor="black", alpha=0.85)
    axes[1, 0].set_title("Latência Média por Chunk (ms)", fontweight="bold")
    axes[1, 0].set_ylabel("Milissegundos")
    axes[1, 0].tick_params(axis="x", rotation=20)
    for bar in b2:
        y = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, y, f"{y:.1f}ms", ha="center", va="bottom", fontsize=9, fontweight="bold")

    vram = [v.gpu_peak_mb for v in valid_results.values()]
    b3 = axes[1, 1].bar(names, vram, color=palette[: len(names)], edgecolor="black", alpha=0.85)
    axes[1, 1].set_title("Pico de Memória GPU (VRAM MB)", fontweight="bold")
    axes[1, 1].set_ylabel("Megabytes")
    axes[1, 1].tick_params(axis="x", rotation=20)
    for bar in b3:
        y = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, y, f"{y:.1f} MB", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.suptitle("Dashboard Geral de Performance de Embeddings", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=VISUAL_CONFIG["dpi"], bbox_inches="tight")
    plt.close(fig)

