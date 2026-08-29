from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
import torch

from edgeforge_core.models.onnx_comparator import compare_pytorch_and_onnx_outputs
from edgeforge_core.models.onnx_exporter import export_model_to_onnx
from edgeforge_core.models.onnx_validator import validate_onnx_model
from edgeforge_core.models.tiny_classifier_adapter import TinyClassifierAdapter


def test_onnx_export_creates_model_file(tmp_path: Path) -> None:
    adapter = TinyClassifierAdapter()
    device = torch.device("cpu")

    model = adapter.load_model(device)

    sample_input = adapter.create_sample_input(
        batch_size=2,
        device=device,
    )

    output_path = tmp_path / "tiny_classifier.onnx"

    exported_path = export_model_to_onnx(
        model=model,
        sample_input=sample_input,
        output_path=output_path,
    )

    assert exported_path.exists()
    assert exported_path.stat().st_size > 0


def test_exported_onnx_model_is_valid(tmp_path: Path) -> None:
    adapter = TinyClassifierAdapter()
    device = torch.device("cpu")

    model = adapter.load_model(device)

    sample_input = adapter.create_sample_input(
        batch_size=2,
        device=device,
    )

    output_path = tmp_path / "tiny_classifier.onnx"

    export_model_to_onnx(
        model=model,
        sample_input=sample_input,
        output_path=output_path,
    )

    validate_onnx_model(output_path)


def test_onnx_output_matches_pytorch(tmp_path: Path) -> None:
    adapter = TinyClassifierAdapter()
    device = torch.device("cpu")

    model = adapter.load_model(device)

    sample_input = adapter.create_sample_input(
        batch_size=2,
        device=device,
    )

    output_path = tmp_path / "tiny_classifier.onnx"

    export_model_to_onnx(
        model=model,
        sample_input=sample_input,
        output_path=output_path,
    )

    difference = compare_pytorch_and_onnx_outputs(
        model=model,
        sample_input=sample_input,
        onnx_path=output_path,
    )

    assert difference < 1e-4


def test_onnx_supports_dynamic_batch_sizes(tmp_path: Path) -> None:
    adapter = TinyClassifierAdapter()
    device = torch.device("cpu")

    model = adapter.load_model(device)

    sample_input = adapter.create_sample_input(
        batch_size=2,
        device=device,
    )

    output_path = tmp_path / "tiny_classifier.onnx"

    export_model_to_onnx(
        model=model,
        sample_input=sample_input,
        output_path=output_path,
    )

    session = ort.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    for batch_size in (1, 4, 8):
        inputs = np.random.rand(
            batch_size,
            3,
            224,
            224,
        ).astype(np.float32)

        outputs = cast(
            np.ndarray,
            session.run(
                None,
                {
                    input_name: inputs,
                },
            )[0],
        )

        assert outputs.shape == (batch_size, 10)
