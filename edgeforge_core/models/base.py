from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class ModelAdapter(ABC):
    """Common interface that every EdgeForge model adapter must follow."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return a unique name for the model."""

    @property
    @abstractmethod
    def input_shape(self) -> tuple[int, int, int]:
        """Return the expected image shape: channels, height and width."""

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Return the number of output classes."""

    @abstractmethod
    def load_model(self, device: torch.device) -> nn.Module:
        """Create the model and move it onto the requested device."""

    @abstractmethod
    def create_sample_input(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create example input data for testing and benchmarking."""

    @abstractmethod
    def preprocess(self, raw_input: Any) -> torch.Tensor:
        """Convert raw input into the tensor format expected by the model."""
