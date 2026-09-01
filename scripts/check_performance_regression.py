from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from edgeforge_core.validation.performance_regression import compare_performance

DEFAULT_BASELINE = Path("configs/performance_baseline.json")
DEFAULT_RESULTS = Path("artifacts/results/resnet18_benchmark.json")


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare benchmark results with the approved baseline."
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline = load_json(args.baseline)
    results = load_json(args.results)

    if not isinstance(baseline, dict):
        raise TypeError("The baseline must be a JSON object")

    if not isinstance(results, list):
        raise TypeError("Benchmark results must be a JSON array")

    comparisons, missing_runtimes = compare_performance(baseline, results)
    regressions = [comparison for comparison in comparisons if comparison.is_regression]

    for comparison in comparisons:
        status = "REGRESSION" if comparison.is_regression else "PASS"
        print(
            f"{status:10} {comparison.runtime_id:20} "
            f"{comparison.current_mean_ms:8.3f} ms "
            f"({comparison.change_percent:+6.2f}%)"
        )

    if missing_runtimes:
        print("Missing runtimes:", ", ".join(missing_runtimes))

    if regressions or missing_runtimes:
        return 1

    print("Performance regression gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
