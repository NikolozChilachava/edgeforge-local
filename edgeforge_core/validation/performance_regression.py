from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PerformanceComparison:
    runtime_id: str
    baseline_mean_ms: float
    current_mean_ms: float
    change_percent: float
    threshold_percent: float

    @property
    def is_regression(self) -> bool:
        return self.change_percent > self.threshold_percent


def compare_performance(
    baseline: Mapping[str, Any],
    current_results: Sequence[Mapping[str, Any]],
) -> tuple[list[PerformanceComparison], list[str]]:
    model_id = str(baseline["model_id"])
    batch_size = int(baseline["batch_size"])
    threshold_percent = float(baseline["threshold_percent"])
    baseline_runtimes = baseline["runtimes"]

    if threshold_percent < 0:
        raise ValueError("threshold_percent must be non-negative")

    if not isinstance(baseline_runtimes, Mapping):
        raise TypeError("runtimes must be an object")

    current_by_runtime = {
        str(result["device"]): result
        for result in current_results
        if str(result.get("model_id")) == model_id
        and int(result.get("batch_size", -1)) == batch_size
    }

    comparisons: list[PerformanceComparison] = []
    missing_runtimes: list[str] = []

    for runtime_id, baseline_value in baseline_runtimes.items():
        runtime_name = str(runtime_id)
        result = current_by_runtime.get(runtime_name)

        if result is None:
            missing_runtimes.append(runtime_name)
            continue

        baseline_mean_ms = float(baseline_value)
        current_mean_ms = float(result["mean_ms"])

        if baseline_mean_ms <= 0 or current_mean_ms <= 0:
            raise ValueError("mean latency values must be positive")

        change_percent = (current_mean_ms - baseline_mean_ms) / baseline_mean_ms * 100

        comparisons.append(
            PerformanceComparison(
                runtime_id=runtime_name,
                baseline_mean_ms=baseline_mean_ms,
                current_mean_ms=current_mean_ms,
                change_percent=change_percent,
                threshold_percent=threshold_percent,
            )
        )

    return comparisons, sorted(missing_runtimes)
