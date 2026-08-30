from __future__ import annotations

from pathlib import Path

import numpy as np
import openvino as ov  # type: ignore[import-untyped]

from edgeforge_core.benchmarking.config import BenchmarkConfig
from edgeforge_core.benchmarking.runner import collect_latency_measurements
from edgeforge_core.benchmarking.statistics import build_benchmark_result
from edgeforge_core.runtimes.openvino_runtime import OpenVINORuntime

ONNX_PATH = Path("artifacts/models/resnet18.onnx")

FP32_XML_PATH = Path("artifacts/models/resnet18_fp32.xml")
INT8_XML_PATH = Path("artifacts/models/resnet18_int8.xml")


def model_size_mb(xml_path: Path) -> float:
    bin_path = xml_path.with_suffix(".bin")

    total_bytes = xml_path.stat().st_size + bin_path.stat().st_size

    return total_bytes / (1024 * 1024)


def benchmark_model(
    model_path: Path,
    runtime_name: str,
    inputs: np.ndarray,
):
    runtime = OpenVINORuntime("CPU")
    model = runtime.load_model(model_path)

    config = BenchmarkConfig(
        warmup_runs=5,
        measured_runs=20,
        batch_size=1,
    )

    latencies = collect_latency_measurements(
        runtime=runtime,
        model=model,
        inputs=inputs,
        config=config,
    )

    return build_benchmark_result(
        model_id="resnet18_imagenet",
        device=runtime_name,
        batch_size=config.batch_size,
        latencies_ms=latencies,
    )


def main() -> None:
    if not INT8_XML_PATH.exists():
        raise FileNotFoundError(
            "INT8 model not found. Run quantize_resnet18_int8 first."
        )

    core = ov.Core()

    if not FP32_XML_PATH.exists():
        fp32_model = core.read_model(ONNX_PATH)

        ov.save_model(
            fp32_model,
            FP32_XML_PATH,
            compress_to_fp16=False,
        )

    inputs = np.random.rand(
        1,
        3,
        224,
        224,
    ).astype(np.float32)

    fp32_result = benchmark_model(
        FP32_XML_PATH,
        "openvino_fp32_cpu",
        inputs,
    )

    int8_result = benchmark_model(
        INT8_XML_PATH,
        "openvino_int8_cpu",
        inputs,
    )

    fp32_size = model_size_mb(FP32_XML_PATH)
    int8_size = model_size_mb(INT8_XML_PATH)

    speedup = fp32_result.mean_ms / int8_result.mean_ms
    size_reduction = (1 - int8_size / fp32_size) * 100

    print()
    print("OpenVINO ResNet-18")
    print("-" * 55)

    print(f"{'Runtime':<22}{'Mean ms':>12}{'Items/sec':>14}")

    print(
        f"{fp32_result.device:<22}"
        f"{fp32_result.mean_ms:>12.3f}"
        f"{fp32_result.throughput_items_per_second:>14.2f}"
    )

    print(
        f"{int8_result.device:<22}"
        f"{int8_result.mean_ms:>12.3f}"
        f"{int8_result.throughput_items_per_second:>14.2f}"
    )

    print()
    print(f"FP32 size: {fp32_size:.2f} MB")
    print(f"INT8 size: {int8_size:.2f} MB")
    print(f"INT8 speedup: {speedup:.2f}x")
    print(f"Size reduction: {size_reduction:.1f}%")


if __name__ == "__main__":
    main()
