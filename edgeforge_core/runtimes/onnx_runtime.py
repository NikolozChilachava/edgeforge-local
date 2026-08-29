from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]

from edgeforge_core.runtimes.base import RuntimeAdapter


class ONNXRuntime(RuntimeAdapter):
    """Run ONNX models using ONNX Runtime."""

    def __init__(self, provider: str) -> None:
        self.provider = provider

    @property
    def runtime_id(self) -> str:
        if self.provider == "CPUExecutionProvider":
            return "onnx_cpu"

        if self.provider == "CUDAExecutionProvider":
            return "onnx_cuda"

        return f"onnx_{self.provider.lower()}"

    def load_model(self, model_path: Path) -> ort.InferenceSession:

        available_providers = ort.get_available_providers()

        if self.provider not in available_providers:
            raise RuntimeError(
                f"ONNX Runtime provider '{self.provider}' is not available. "
                f"Available providers: {available_providers}"
            )

        return ort.InferenceSession(
            str(model_path),
            providers=[self.provider],
        )

    def run(
        self,
        model: ort.InferenceSession,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """Run one ONNX inference."""

        input_name = model.get_inputs()[0].name

        output = cast(
            np.ndarray,
            model.run(
                None,
                {
                    input_name: inputs,
                },
            )[0],
        )

        return output
