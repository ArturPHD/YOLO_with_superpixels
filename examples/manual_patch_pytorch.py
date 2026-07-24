"""Patch a generic PyTorch model without using Ultralytics."""

from __future__ import annotations

import torch
import torch.nn as nn

from yolo_superpixels import AdjacencyChannelBuilder, SLICSuperpixelAlgorithm, patch_first_conv2d


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


model = TinyBackbone()
builder = AdjacencyChannelBuilder(SLICSuperpixelAlgorithm(n_segments=128, compactness=8.0))
result = patch_first_conv2d(model, builder)

print(result)
print(model(torch.rand(2, 3, 128, 128)).shape)
