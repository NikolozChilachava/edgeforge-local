from pathlib import Path

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest
import torch

from edgeforge_core.models.onnx_exporter import export_model_to_onnx
from edgeforge_core.models.tiny_classifier_adapter import TinyClassifierAdapter
from edgeforge_core.runtimes.onnx_runtime import ONNXRuntime


def create_test_onnx_model(tmp_path: Path) -> Path:
    """Create a temporary ONNX model for runtime tests."""

    adapter = TinyClassifierAdapter()
    device = torch.device("cpu")

    model = adapter.load_model(device)

    sample_input = adapter.create_sample_input(
        batch_size=2,
        device=device,
    )

    model_path = tmp_path / "tiny_classifier.onnx"

    return export_model_to_onnx(
        model=model,
        sample_input=sample_input,
        output_path=model_path,
    )


def test_onnx_cpu_runtime_id() -> None:
    runtime = ONNXRuntime("CPUExecutionProvider")

    assert runtime.runtime_id == "onnx_cpu"


def test_onnx_cpu_runtime_runs_model(tmp_path: Path) -> None:
    model_path = create_test_onnx_model(tmp_path)

    runtime = ONNXRuntime("CPUExecutionProvider")
    model = runtime.load_model(model_path)

    inputs = np.random.rand(
        4,
        3,
        224,
        224,
    ).astype(np.float32)

    outputs = runtime.run(
        model=model,
        inputs=inputs,
    )

    assert outputs.shape == (4, 10)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or "CUDAExecutionProvider" not in ort.get_available_providers(),
    reason="A usable CUDA GPU is not available for ONNX Runtime.",
)
def test_onnx_cuda_runtime_runs_model(tmp_path: Path) -> None:
    model_path = create_test_onnx_model(tmp_path)

    runtime = ONNXRuntime("CUDAExecutionProvider")
    model = runtime.load_model(model_path)

    inputs = np.random.rand(
        4,
        3,
        224,
        224,
    ).astype(np.float32)

    outputs = runtime.run(
        model=model,
        inputs=inputs,
    )

    assert runtime.runtime_id == "onnx_cuda"
    assert "CUDAExecutionProvider" in model.get_providers()
    assert outputs.shape == (4, 10)
