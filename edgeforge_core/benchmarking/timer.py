from __future__ import annotations

import time

import torch
from torch import nn


def measure_inference_ms(
    model: nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
) -> float:
    """Measure one model inference and return latency in milliseconds."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start_time = time.perf_counter()

    with torch.no_grad():
        model(inputs)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    end_time = time.perf_counter()

    elapsed_seconds = end_time - start_time
    elapsed_ms = elapsed_seconds * 1000

    return elapsed_ms
