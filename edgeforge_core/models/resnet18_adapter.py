from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (  # type: ignore[import-untyped]
    ResNet18_Weights,
    resnet18,
)

from edgeforge_core.models.base import ModelAdapter


class ResNet18Adapter(ModelAdapter):
    """Adapter for the pretrained TorchVision ResNet-18 model."""

    @property
    def model_id(self) -> str:
        return "resnet18_imagenet"

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (3, 224, 224)

    @property
    def num_classes(self) -> int:
        return 1000

    def load_model(self, device: torch.device) -> nn.Module:
        weights = ResNet18_Weights.DEFAULT

        model = resnet18(weights=weights)
        model = model.to(device)
        model.eval()

        return model

    def create_sample_input(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than zero.")

        return torch.rand(
            batch_size,
            *self.input_shape,
            device=device,
            dtype=torch.float32,
        )

    def preprocess(self, raw_input: torch.Tensor) -> torch.Tensor:
        if raw_input.ndim == 3:
            raw_input = raw_input.unsqueeze(0)

        if raw_input.ndim != 4:
            raise ValueError("Input must have shape [batch, channels, height, width].")

        if tuple(raw_input.shape[1:]) != self.input_shape:
            raise ValueError(
                f"Expected input shape {self.input_shape}, "
                f"but received {tuple(raw_input.shape[1:])}."
            )

        raw_input = raw_input.to(dtype=torch.float32)

        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            device=raw_input.device,
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225],
            device=raw_input.device,
        ).view(1, 3, 1, 1)

        return (raw_input - mean) / std
