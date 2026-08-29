import pytest
import torch

from edgeforge_core.models.tiny_classifier_adapter import TinyClassifierAdapter
from edgeforge_core.runtimes.pytorch_runtime import PyTorchRuntime


def test_pytorch_cpu_runtime_id() -> None:
    runtime = PyTorchRuntime(torch.device("cpu"))

    assert runtime.runtime_id == "pytorch_cpu"


def test_pytorch_cpu_runtime_runs_model() -> None:
    adapter = TinyClassifierAdapter()
    device = torch.device("cpu")

    runtime = PyTorchRuntime(device)
    model = adapter.load_model(device)
    inputs = adapter.create_sample_input(
        batch_size=4,
        device=device,
    )

    outputs = runtime.run(
        model=model,
        inputs=inputs,
    )

    assert outputs.shape == (4, 10)
    assert outputs.device.type == "cpu"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available on this machine.",
)
def test_pytorch_cuda_runtime_runs_model() -> None:
    adapter = TinyClassifierAdapter()
    device = torch.device("cuda")

    runtime = PyTorchRuntime(device)
    model = adapter.load_model(device)

    inputs = adapter.create_sample_input(
        batch_size=4,
        device=device,
    )

    outputs = runtime.run(
        model=model,
        inputs=inputs,
    )

    assert runtime.runtime_id == "pytorch_cuda"
    assert outputs.shape == (4, 10)
    assert outputs.device.type == "cuda"
