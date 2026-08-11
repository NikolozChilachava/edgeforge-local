import pytest
import torch

from edgeforge_core.models.tiny_classifier import TinyClassifier
from edgeforge_core.models.tiny_classifier_adapter import TinyClassifierAdapter


# test that the model outputs a tensor with the right number of classes
def test_tiny_classifier_output_shape() -> None:
    model = TinyClassifier(num_classes=10)

    inputs = torch.rand(4, 3, 224, 224)

    outputs = model(inputs)

    assert outputs.shape == (4, 10)


# make sure the model works with various batch sizes
def test_tiny_classifier_different_batch_sizes() -> None:
    model = TinyClassifier(num_classes=10)

    for batch_size in (1, 2, 8):
        inputs = torch.rand(batch_size, 3, 224, 224)

        outputs = model(inputs)

        assert outputs.shape == (batch_size, 10)


# check that the adapter produces a sample input with the expected shape and dtype
def test_adapter_creates_correct_sample_input() -> None:
    adapter = TinyClassifierAdapter()
    device = torch.device("cpu")

    inputs = adapter.create_sample_input(
        batch_size=4,
        device=device,
    )

    assert inputs.shape == (4, 3, 224, 224)
    assert inputs.dtype == torch.float32
    assert inputs.device.type == "cpu"


# verify that preprocessing a single image adds a batch dimension
def test_preprocess_single_image_adds_batch_dimension() -> None:
    adapter = TinyClassifierAdapter()

    image = torch.rand(3, 224, 224)

    processed = adapter.preprocess(image)

    assert processed.shape == (1, 3, 224, 224)
    assert processed.dtype == torch.float32


# ensure the preprocess method rejects images with wrong spatial dimensions
def test_preprocess_rejects_wrong_image_shape() -> None:
    adapter = TinyClassifierAdapter()

    wrong_image = torch.rand(3, 100, 100)

    with pytest.raises(ValueError):
        adapter.preprocess(wrong_image)


# make sure the adapter throws an error when given a batch size of zero
def test_adapter_rejects_invalid_batch_size() -> None:
    adapter = TinyClassifierAdapter()
    device = torch.device("cpu")

    with pytest.raises(ValueError):
        adapter.create_sample_input(
            batch_size=0,
            device=device,
        )
