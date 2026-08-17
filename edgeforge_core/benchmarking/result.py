from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkResult:
    """Summary of a completed benchmark."""

    model_id: str
    device: str
    batch_size: int

    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float

    throughput_items_per_second: float
