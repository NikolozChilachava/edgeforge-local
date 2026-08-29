import pytest
import torch

from edgeforge_core.benchmarking.config import BenchmarkConfig
from edgeforge_core.benchmarking.runner import collect_latency_measurements
from edgeforge_core.benchmarking.statistics import build_benchmark_result
from edgeforge_core.models.tiny_classifier_adapter import TinyClassifierAdapter
from edgeforge_core.runtimes.pytorch_runtime import PyTorchRuntime


def test_build_benchmark_result_calculates_statistics() -> None:
    latencies = [2.0, 4.0, 6.0]

    result = build_benchmark_result(
        model_id="tiny_classifier_v1",
        device="cpu",
        batch_size=1,
        latencies_ms=latencies,
    )

    assert result.model_id == "tiny_classifier_v1"
    assert result.device == "cpu"
    assert result.batch_size == 1

    assert result.mean_ms == pytest.approx(4.0)
    assert result.median_ms == pytest.approx(4.0)
    assert result.min_ms == pytest.approx(2.0)
    assert result.max_ms == pytest.approx(6.0)
    assert result.throughput_items_per_second == pytest.approx(250.0)


def test_build_benchmark_result_rejects_empty_measurements() -> None:
    with pytest.raises(ValueError):
        build_benchmark_result(
            model_id="tiny_classifier_v1",
            device="cpu",
            batch_size=1,
            latencies_ms=[],
        )


def test_collect_latency_measurements_returns_expected_count() -> None:
    adapter = TinyClassifierAdapter()

    device = torch.device("cpu")
    runtime = PyTorchRuntime(device)

    model = adapter.load_model(device)

    config = BenchmarkConfig(
        warmup_runs=2,
        measured_runs=5,
        batch_size=1,
    )

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

    assert len(latencies) == 5
    assert all(latency >= 0.0 for latency in latencies)
