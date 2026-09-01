import pytest

from edgeforge_core.validation.performance_regression import compare_performance


def baseline() -> dict[str, object]:
    return {
        "model_id": "resnet18_imagenet",
        "batch_size": 1,
        "threshold_percent": 15.0,
        "runtimes": {
            "pytorch_cpu": 40.0,
            "onnx_cpu": 10.0,
        },
    }


def result(runtime: str, mean_ms: float) -> dict[str, object]:
    return {
        "model_id": "resnet18_imagenet",
        "device": runtime,
        "batch_size": 1,
        "mean_ms": mean_ms,
    }


def test_accepts_results_within_threshold() -> None:
    comparisons, missing = compare_performance(
        baseline(),
        [result("pytorch_cpu", 44.0), result("onnx_cpu", 10.5)],
    )

    assert missing == []
    assert all(not comparison.is_regression for comparison in comparisons)


def test_detects_latency_regression() -> None:
    comparisons, _ = compare_performance(
        baseline(),
        [result("pytorch_cpu", 48.0), result("onnx_cpu", 10.0)],
    )

    regression = next(
        comparison
        for comparison in comparisons
        if comparison.runtime_id == "pytorch_cpu"
    )

    assert regression.change_percent == pytest.approx(20.0)
    assert regression.is_regression


def test_reports_missing_runtime() -> None:
    _, missing = compare_performance(
        baseline(),
        [result("pytorch_cpu", 40.0)],
    )

    assert missing == ["onnx_cpu"]


def test_rejects_negative_threshold() -> None:
    invalid_baseline = baseline()
    invalid_baseline["threshold_percent"] = -1

    with pytest.raises(ValueError, match="non-negative"):
        compare_performance(invalid_baseline, [])
