from __future__ import annotations

import warnings
from pathlib import Path

import torch
from torch import nn


def export_model_to_onnx(
    model: nn.Module,
    sample_input: torch.Tensor,
    output_path: Path,
) -> Path:
    """Export a PyTorch model to ONNX with a dynamic batch dimension."""

    model.eval()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    batch_dimension = torch.export.Dim("batch")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"`isinstance\(treespec, LeafSpec\)` is deprecated, use "
                r"`isinstance\(treespec, TreeSpec\) and treespec\.is_leaf\(\)` instead\."
            ),
            category=FutureWarning,
        )
        torch.onnx.export(
            model,
            (sample_input,),
            output_path,
            input_names=["input"],
            output_names=["output"],
            dynamo=True,
            dynamic_shapes=({0: batch_dimension},),
        )

    return output_path
