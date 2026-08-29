from __future__ import annotations

from pathlib import Path

import onnx


def validate_onnx_model(model_path: Path) -> None:
    """Load an ONNX model and verify that its graph is valid."""

    model = onnx.load(model_path)

    onnx.checker.check_model(model)
