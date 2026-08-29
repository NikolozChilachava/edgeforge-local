from __future__ import annotations

import time
from typing import Any

from edgeforge_core.runtimes.base import RuntimeAdapter


def measure_inference_ms(
    runtime: RuntimeAdapter,
    model: Any,
    inputs: Any,
) -> float:
    """Measure one runtime inference in milliseconds."""

    runtime.synchronize()

    start_time = time.perf_counter()

    runtime.run(
        model=model,
        inputs=inputs,
    )

    runtime.synchronize()

    end_time = time.perf_counter()

    elapsed_seconds = end_time - start_time
    elapsed_ms = elapsed_seconds * 1000

    return elapsed_ms
