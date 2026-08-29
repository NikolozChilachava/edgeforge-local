from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class RuntimeAdapter(ABC):
    """Common interface for model execution runtimes."""

    @property
    @abstractmethod
    def runtime_id(self) -> str:
        """Return the unique name of this runtime."""
        raise NotImplementedError

    @abstractmethod
    def run(self, model: Any, inputs: Any) -> Any:
        """Run one inference and return the output."""
        raise NotImplementedError

    def synchronize(self) -> None:
        """Wait for pending runtime work to finish."""
