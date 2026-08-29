from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from edgeforge_core.benchmarking.config import BenchmarkConfig
from edgeforge_core.benchmarking.result import BenchmarkResult
from edgeforge_core.benchmarking.runner import collect_latency_measurements
from edgeforge_core.benchmarking.statistics import build_benchmark_result
from edgeforge_core.models.tiny_classifier_adapter import TinyClassifierAdapter
from edgeforge_core.runtimes.onnx_runtime import ONNXRuntime
from edgeforge_core.runtimes.pytorch_runtime import PyTorchRuntime


def benchmark_pytorch(
    adapter: TinyClassifierAdapter,
    device: torch.device,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    runtime = PyTorchRuntime(device)

    model = adapter.load_model(device)

    inputs = adapter.create_sample_input(
        batch_size=config.batch_size,
        device=device,
    )

    latencies = collect_latency_measurements(
        runtime=runtime,
        model=model,
        inputs=inputs,
        config=config,
    )

    return build_benchmark_result(
        model_id=adapter.model_id,
        device=runtime.runtime_id,
        batch_size=config.batch_size,
        latencies_ms=latencies,
    )


def benchmark_onnx(
    provider: str,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    runtime = ONNXRuntime(provider)

    model = runtime.load_model(Path("artifacts/models/tiny_classifier.onnx"))

    inputs = np.random.rand(
        config.batch_size,
        3,
        224,
        224,
    ).astype(np.float32)

    latencies = collect_latency_measurements(
        runtime=runtime,
        model=model,
        inputs=inputs,
        config=config,
    )

    return build_benchmark_result(
        model_id="tiny_classifier_v1",
        device=runtime.runtime_id,
        batch_size=config.batch_size,
        latencies_ms=latencies,
    )


def print_results(results: list[BenchmarkResult]) -> None:
    print()
    print(
        f"{'Runtime':<18}"
        f"{'Mean ms':>12}"
        f"{'Median ms':>14}"
        f"{'Min ms':>12}"
        f"{'Max ms':>12}"
        f"{'Items/sec':>14}"
    )

    print("-" * 82)

    for result in results:
        print(
            f"{result.device:<18}"
            f"{result.mean_ms:>12.3f}"
            f"{result.median_ms:>14.3f}"
            f"{result.min_ms:>12.3f}"
            f"{result.max_ms:>12.3f}"
            f"{result.throughput_items_per_second:>14.2f}"
        )


def main() -> None:
    config = BenchmarkConfig(
        warmup_runs=5,
        measured_runs=20,
        batch_size=1,
    )

    adapter = TinyClassifierAdapter()

    results = [
        benchmark_pytorch(
            adapter=adapter,
            device=torch.device("cpu"),
            config=config,
        ),
        benchmark_pytorch(
            adapter=adapter,
            device=torch.device("cuda"),
            config=config,
        ),
        benchmark_onnx(
            provider="CPUExecutionProvider",
            config=config,
        ),
        benchmark_onnx(
            provider="CUDAExecutionProvider",
            config=config,
        ),
    ]

    print_results(results)


if __name__ == "__main__":
    main()
