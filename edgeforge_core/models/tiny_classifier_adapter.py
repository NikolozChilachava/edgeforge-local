from __future__ import annotations

import torch
from torch import nn

from edgeforge_core.models.base import ModelAdapter
from edgeforge_core.models.tiny_classifier import TinyClassifier


class TinyClassifierAdapter(ModelAdapter):
    @property
    def model_id(self) -> str:
        return "tiny_classifier_v1"

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (3, 224, 224)

    @property
    def num_classes(self) -> int:
        return 10

    def load_model(self, device: torch.device) -> nn.Module:
        """Create the model and place it on the requested device."""
        torch.manual_seed(42)

        model = TinyClassifier(num_classes=self.num_classes)
        model = model.to(device)
        model.eval()

        return model

    def create_sample_input(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create a batch of fake images for benchmarking."""
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than zero.")

        return torch.rand(
            batch_size,
            *self.input_shape,
            device=device,
            dtype=torch.float32,
        )

    def preprocess(self, raw_input: torch.Tensor) -> torch.Tensor:
        """Prepare an input tensor for the model."""
        if raw_input.ndim == 3:
            raw_input = raw_input.unsqueeze(0)

        expected_shape = self.input_shape

        if raw_input.ndim != 4:
            raise ValueError("Input must have shape [batch, channels, height, width].")

        if tuple(raw_input.shape[1:]) != expected_shape:
            raise ValueError(
                f"Expected input shape {expected_shape}, "
                f"but received {tuple(raw_input.shape[1:])}."
            )

        return raw_input.to(dtype=torch.float32)
