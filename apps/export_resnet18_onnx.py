from __future__ import annotations

from pathlib import Path

import torch

from edgeforge_core.models.onnx_exporter import export_model_to_onnx
from edgeforge_core.models.resnet18_adapter import ResNet18Adapter

OUTPUT_PATH = Path("artifacts/models/resnet18.onnx")


def main() -> None:
    device = torch.device("cpu")
    adapter = ResNet18Adapter()
    model = adapter.load_model(device)
    sample_input = adapter.create_sample_input(batch_size=1, device=device)
    sample_input = adapter.preprocess(sample_input)

    exported_path = export_model_to_onnx(
        model=model,
        sample_input=sample_input,
        output_path=OUTPUT_PATH,
    )

    print(f"Exported ResNet-18 to {exported_path}")


if __name__ == "__main__":
    main()
