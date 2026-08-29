from __future__ import annotations

from typing import Any

from edgeforge_core.benchmarking.config import BenchmarkConfig
from edgeforge_core.benchmarking.timer import measure_inference_ms
from edgeforge_core.runtimes.base import RuntimeAdapter


def collect_latency_measurements(
    runtime: RuntimeAdapter,
    model: Any,
    inputs: Any,
    config: BenchmarkConfig,
) -> list[float]:
    """Warm up the runtime and collect inference latency measurements."""

    for _ in range(config.warmup_runs):
        measure_inference_ms(
            runtime=runtime,
            model=model,
            inputs=inputs,
        )

    latencies_ms: list[float] = []

    for _ in range(config.measured_runs):
        latency_ms = measure_inference_ms(
            runtime=runtime,
            model=model,
            inputs=inputs,
        )

        latencies_ms.append(latency_ms)

    return latencies_ms
