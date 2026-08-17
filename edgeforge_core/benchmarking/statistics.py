from __future__ import annotations

import statistics

from edgeforge_core.benchmarking.result import BenchmarkResult


def build_benchmark_result(
    model_id: str,
    device: str,
    batch_size: int,
    latencies_ms: list[float],
) -> BenchmarkResult:
    """Turn raw latency measurements into a benchmark summary."""

    if not latencies_ms:
        raise ValueError("At least one latency measurement is required.")

    mean_ms = statistics.mean(latencies_ms)
    median_ms = statistics.median(latencies_ms)
    min_ms = min(latencies_ms)
    max_ms = max(latencies_ms)

    throughput_items_per_second = (batch_size * 1000) / mean_ms

    return BenchmarkResult(
        model_id=model_id,
        device=device,
        batch_size=batch_size,
        mean_ms=mean_ms,
        median_ms=median_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        throughput_items_per_second=throughput_items_per_second,
    )
