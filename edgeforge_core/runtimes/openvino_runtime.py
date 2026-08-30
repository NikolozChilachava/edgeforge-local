from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import openvino as ov  # type: ignore[import-untyped]

from edgeforge_core.runtimes.base import RuntimeAdapter


class OpenVINORuntime(RuntimeAdapter):
    """Run models using OpenVINO."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.core = ov.Core()

    @property
    def runtime_id(self) -> str:
        clean_device = self.device.lower().replace(".", "_")
        return f"openvino_{clean_device}"

    def load_model(self, model_path: Path) -> ov.CompiledModel:
        """Load and compile a model for the selected OpenVINO device."""

        if self.device not in self.core.available_devices:
            raise RuntimeError(
                f"OpenVINO device '{self.device}' is unavailable. "
                f"Available devices: {self.core.available_devices}"
            )

        model = self.core.read_model(model_path)

        return self.core.compile_model(
            model,
            self.device,
        )

    def run(
        self,
        model: ov.CompiledModel,
        inputs: Any,
    ) -> np.ndarray:
        """Run one synchronous OpenVINO inference."""

        results = model(inputs)

        output = next(iter(results.values()))

        return cast(np.ndarray, output)
