from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
import torch
from torch import nn


def compare_pytorch_and_onnx_outputs(
    model: nn.Module,
    sample_input: torch.Tensor,
    onnx_path: Path,
) -> float:
    """Compare PyTorch and ONNX outputs for the same input."""

    model.eval()

    with torch.no_grad():
        pytorch_output = model(sample_input)

    pytorch_output_numpy = pytorch_output.detach().cpu().numpy()

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    onnx_output = cast(
        np.ndarray,
        session.run(
            None,
            {
                input_name: sample_input.detach().cpu().numpy(),
            },
        )[0],
    )

    np.testing.assert_allclose(
        pytorch_output_numpy,
        onnx_output,
        rtol=1e-4,
        atol=1e-5,
    )

    max_absolute_difference = float(np.max(np.abs(pytorch_output_numpy - onnx_output)))

    return max_absolute_difference
