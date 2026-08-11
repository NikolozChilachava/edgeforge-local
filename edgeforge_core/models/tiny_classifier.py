from __future__ import annotations

import torch
from torch import nn


class TinyClassifier(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.first_convolution = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            padding=1,
        )
        self.first_activation = nn.ReLU()

        self.second_convolution = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, padding=1
        )
        self.second_activation = nn.ReLU()

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(
            in_features=16,
            out_features=num_classes,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        first_features = self.first_convolution(inputs)
        first_activated = self.first_activation(first_features)

        second_features = self.second_convolution(first_activated)
        second_activated = self.second_activation(second_features)

        pooled_features = self.pooling(second_activated)

        flattened_features = torch.flatten(
            pooled_features,
            start_dim=1,
        )
        class_scores = self.classifier(flattened_features)

        return class_scores
