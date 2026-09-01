from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
import openvino as ov  # type: ignore[import-untyped]
import torch

from edgeforge_core.benchmarking.config import BenchmarkConfig
from edgeforge_core.benchmarking.result import BenchmarkResult
from edgeforge_core.benchmarking.runner import collect_latency_measurements
from edgeforge_core.benchmarking.statistics import build_benchmark_result
from edgeforge_core.models.resnet18_adapter import ResNet18Adapter
from edgeforge_core.runtimes.onnx_runtime import ONNXRuntime
from edgeforge_core.runtimes.openvino_runtime import OpenVINORuntime
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


def create_numpy_inputs(
    adapter: ResNet18Adapter,
    config: BenchmarkConfig,
) -> np.ndarray:
    device = torch.device("cpu")

    inputs = adapter.create_sample_input(
        batch_size=config.batch_size,
        device=device,
    )

    inputs = adapter.preprocess(inputs)

    return inputs.numpy().astype(np.float32)


def benchmark_onnx(
    adapter: ResNet18Adapter,
    provider: str,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    runtime = ONNXRuntime(provider)
    model = runtime.load_model(ONNX_PATH)

    inputs = create_numpy_inputs(adapter, config)

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


def benchmark_openvino(
    adapter: ResNet18Adapter,
    device: str,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    runtime = OpenVINORuntime(device)
    model = runtime.load_model(ONNX_PATH)

    inputs = create_numpy_inputs(adapter, config)

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


def print_results(results: list[BenchmarkResult]) -> None:
    print()
    print(
        f"{'Runtime':<20}"
        f"{'Mean ms':>12}"
        f"{'Median ms':>14}"
        f"{'Min ms':>12}"
        f"{'Max ms':>12}"
        f"{'Items/sec':>14}"
    )

    print("-" * 84)

    for result in results:
        print(
            f"{result.device:<20}"
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

    RESULT_PATH.write_text(
        json.dumps(
            [asdict(result) for result in results],
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Saved:", RESULT_PATH)


def main() -> None:
    if not ONNX_PATH.exists():
        raise FileNotFoundError(
            "ResNet-18 ONNX model not found. "
            "Run 'python -m apps.export_resnet18_onnx' first."
        )

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
        benchmark_onnx(
            adapter,
            "CPUExecutionProvider",
            config,
        ),
        benchmark_openvino(
            adapter,
            "CPU",
            config,
        ),
    ]

    if torch.cuda.is_available():
        results.append(
            benchmark_pytorch(
                adapter,
                torch.device("cuda"),
                config,
            )
        )

        if "CUDAExecutionProvider" in ort.get_available_providers():
            results.append(
                benchmark_onnx(
                    adapter,
                    "CUDAExecutionProvider",
                    config,
                )
            )

    openvino_devices = ov.Core().available_devices

    gpu_devices = [device for device in openvino_devices if device.startswith("GPU")]

    if gpu_devices:
        results.append(
            benchmark_openvino(
                adapter,
                gpu_devices[0],
                config,
            )
        )

    print_results(results)
    save_results(results)


if __name__ == "__main__":
    main()
