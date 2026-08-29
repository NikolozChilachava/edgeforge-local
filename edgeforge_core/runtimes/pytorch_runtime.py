from __future__ import annotations

import torch
from torch import nn

from edgeforge_core.runtimes.base import RuntimeAdapter


class PyTorchRuntime(RuntimeAdapter):
    """Run PyTorch models on a specific device."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    @property
    def runtime_id(self) -> str:
        return f"pytorch_{self.device.type}"

    def run(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Run one PyTorch inference."""

        with torch.no_grad():
            output = model(inputs)

        return output

    def synchronize(self) -> None:
        """Wait for CUDA work to finish when using a GPU."""

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
