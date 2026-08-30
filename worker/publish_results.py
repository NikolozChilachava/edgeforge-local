from __future__ import annotations

import json
import os
from pathlib import Path
from urllib import error, request

RESULT_PATH = Path("artifacts/results/resnet18_benchmark.json")

API_URL = os.getenv(
    "EDGEFORGE_API_URL",
    "http://127.0.0.1:8000",
)


def publish_result(result: dict[str, object]) -> None:
    payload = {
        "model_id": result["model_id"],
        "runtime_id": result["device"],
        "batch_size": result["batch_size"],
        "mean_ms": result["mean_ms"],
        "median_ms": result["median_ms"],
        "min_ms": result["min_ms"],
        "max_ms": result["max_ms"],
        "throughput_items_per_second": (result["throughput_items_per_second"]),
    }

    data = json.dumps(payload).encode("utf-8")

    api_request = request.Request(
        f"{API_URL}/benchmarks",
        data=data,
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with request.urlopen(
        api_request,
        timeout=10,
    ) as response:
        body = json.loads(response.read().decode("utf-8"))

    print(f"Published {body['runtime_id']} (database id={body['id']})")


def main() -> None:
    if not RESULT_PATH.exists():
        raise FileNotFoundError(
            "Benchmark results not found. Run the ResNet benchmark first."
        )

    results = json.loads(
        RESULT_PATH.read_text(
            encoding="utf-8",
        )
    )

    print(f"Publishing {len(results)} benchmark results...")

    try:
        for result in results:
            publish_result(result)

    except error.URLError as exc:
        raise RuntimeError(
            "Could not connect to the EdgeForge API. Make sure Uvicorn is running."
        ) from exc

    print("All benchmark results published.")


if __name__ == "__main__":
    main()
