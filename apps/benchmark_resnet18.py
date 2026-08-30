from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from edgeforge_core.benchmarking.config import BenchmarkConfig
from edgeforge_core.benchmarking.result import BenchmarkResult
from edgeforge_core.benchmarking.runner import collect_latency_measurements
from edgeforge_core.benchmarking.statistics import build_benchmark_result
from edgeforge_core.models.resnet18_adapter import ResNet18Adapter
from edgeforge_core.runtimes.onnx_runtime import ONNXRuntime
from edgeforge_core.runtimes.pytorch_runtime import PyTorchRuntime

ONNX_PATH = Path("artifacts/models/resnet18.onnx")
RESULT_PATH = Path("artifacts/results/resnet18_benchmark.json")


def benchmark_pytorch(
    adapter: ResNet18Adapter,
    device: torch.device,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    runtime = PyTorchRuntime(device)

    model = adapter.load_model(device)

    inputs = adapter.create_sample_input(
        batch_size=config.batch_size,
        device=device,
    )

    inputs = adapter.preprocess(inputs)

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
    adapter: ResNet18Adapter,
    provider: str,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    runtime = ONNXRuntime(provider)
    model = runtime.load_model(ONNX_PATH)

    cpu_device = torch.device("cpu")

    inputs = adapter.create_sample_input(
        batch_size=config.batch_size,
        device=cpu_device,
    )

    inputs = adapter.preprocess(inputs)

    numpy_inputs = inputs.numpy().astype(np.float32)

    latencies = collect_latency_measurements(
        runtime=runtime,
        model=model,
        inputs=numpy_inputs,
        config=config,
    )

    return build_benchmark_result(
        model_id=adapter.model_id,
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


def save_results(results: list[BenchmarkResult]) -> None:
    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = [asdict(result) for result in results]

    RESULT_PATH.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )

    print()
    print("Saved:", RESULT_PATH)


def main() -> None:
    adapter = ResNet18Adapter()

    config = BenchmarkConfig(
        warmup_runs=5,
        measured_runs=20,
        batch_size=1,
    )

    results = [
        benchmark_pytorch(
            adapter,
            torch.device("cpu"),
            config,
        ),
        benchmark_pytorch(
            adapter,
            torch.device("cuda"),
            config,
        ),
        benchmark_onnx(
            adapter,
            "CPUExecutionProvider",
            config,
        ),
        benchmark_onnx(
            adapter,
            "CUDAExecutionProvider",
            config,
        ),
    ]

    print_results(results)
    save_results(results)


if __name__ == "__main__":
    main()
