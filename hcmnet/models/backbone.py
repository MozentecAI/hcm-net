"""ImageNet-pretrained frame encoders that preserve spatial features."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    resnet18,
    resnet34,
)


class ResNetFrameEncoder(nn.Module):
    """Apply a 2D ResNet to frames and expose the last two feature levels.

    Returns the layer3 map (stride 16, higher spatial resolution for the
    regional attention branches) and the layer4 map (stride 32, richer
    semantics for the overall branches).
    """

    def __init__(
        self,
        name: str = "resnet18",
        pretrained: bool = True,
        input_channels: int = 1,
    ) -> None:
        super().__init__()
        if name == "resnet18":
            network = resnet18(
                weights=ResNet18_Weights.DEFAULT if pretrained else None
            )
        elif name == "resnet34":
            network = resnet34(
                weights=ResNet34_Weights.DEFAULT if pretrained else None
            )
        else:
            raise ValueError(
                f"Unsupported backbone '{name}'. Choose resnet18 or resnet34."
            )
        if input_channels != 1:
            raise ValueError("The grayscale encoder requires input_channels=1.")
        original_conv = network.conv1
        grayscale_conv = nn.Conv2d(
            input_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )
        if pretrained:
            with torch.no_grad():
                grayscale_conv.weight.copy_(
                    original_conv.weight.sum(dim=1, keepdim=True)
                )
        network.conv1 = grayscale_conv

        stages = list(network.children())[:-2]
        self.stem = nn.Sequential(*stages[:-1])
        self.layer4 = stages[-1]
        self.out_channels = int(network.fc.in_features)
        self.mid_channels = int(self.out_channels // 2)

    def _encode(self, video: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, frames, channels, height, width = video.shape
        flattened = video.reshape(
            batch_size * frames, channels, height, width
        )
        mid = self.stem(flattened)
        high = self.layer4(mid)
        return mid, high

    def forward(self, video: Tensor) -> tuple[Tensor, Tensor]:
        """Encode [B, T, C, H, W] into ([B,T,Cm,H/16,W/16], [B,T,Ch,H/32,W/32])."""
        batch_size, frames = video.shape[0], video.shape[1]
        mid, high = self._encode(video)
        mid = mid.reshape(batch_size, frames, *mid.shape[1:])
        high = high.reshape(batch_size, frames, *high.shape[1:])
        return mid, high
