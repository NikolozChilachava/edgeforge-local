from __future__ import annotations

from pathlib import Path

import nncf
import openvino as ov  # type: ignore[import-untyped]
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import ResNet18_Weights

ONNX_PATH = Path("artifacts/models/resnet18.onnx")
INT8_PATH = Path("artifacts/models/resnet18_int8.xml")
DATA_PATH = Path("artifacts/calibration_data")

CALIBRATION_SAMPLES = 100


def main() -> None:
    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()

    dataset = CIFAR10(
        root=DATA_PATH,
        train=True,
        download=True,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    def transform_fn(data_item):
        images, _ = data_item
        return images.numpy()

    calibration_dataset = nncf.Dataset(
        loader,
        transform_fn,
    )

    core = ov.Core()
    model = core.read_model(ONNX_PATH)

    print("Quantizing ResNet-18 to INT8...")

    quantized_model = nncf.quantize(
        model=model,
        calibration_dataset=calibration_dataset,
        subset_size=CALIBRATION_SAMPLES,
    )

    INT8_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ov.save_model(
        quantized_model,
        INT8_PATH,
        compress_to_fp16=False,
    )

    print("INT8 model saved:", INT8_PATH)
    print("Exists:", INT8_PATH.exists())


if __name__ == "__main__":
    main()
