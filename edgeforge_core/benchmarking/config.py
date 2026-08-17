from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkConfig:
    """Settings used when running a benchmark."""

    warmup_runs: int = 5
    measured_runs: int = 20
    batch_size: int = 1
