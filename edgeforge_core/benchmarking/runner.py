from __future__ import annotations

import torch
from torch import nn

from edgeforge_core.benchmarking.config import BenchmarkConfig
from edgeforge_core.benchmarking.timer import measure_inference_ms


def collect_latency_measurements(
    model: nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
    config: BenchmarkConfig,
) -> list[float]:
    """Warm up the model and collect inference latency measurements."""

    for _ in range(config.warmup_runs):
        measure_inference_ms(
            model=model,
            inputs=inputs,
            device=device,
        )

    latencies_ms: list[float] = []

    for _ in range(config.measured_runs):
        latency_ms = measure_inference_ms(
            model=model,
            inputs=inputs,
            device=device,
        )

        latencies_ms.append(latency_ms)

    return latencies_ms
