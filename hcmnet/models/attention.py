"""Temporal and anatomy-guided spatial attention modules."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class TemporalAttentionPool(nn.Module):
    """Aggregate frame features independently at every spatial location."""

    def __init__(self, channels: int, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled
        self.score = nn.Conv2d(channels, 1, kernel_size=1) if enabled else None

    def forward(self, features: Tensor) -> Tensor:
        if not self.enabled or self.score is None:
            return features.mean(dim=1)
        batch_size, frames, channels, height, width = features.shape
        logits = self.score(
            features.reshape(batch_size * frames, channels, height, width)
        ).reshape(batch_size, frames, 1, height, width)
        weights = logits.softmax(dim=1)
        return (features * weights).sum(dim=1)


class TemporalAttentionPool1D(nn.Module):
    """Aggregate a sequence of per-frame embedding vectors."""

    def __init__(self, channels: int, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled
        self.score = nn.Linear(channels, 1) if enabled else None

    def forward(self, sequence: Tensor) -> Tensor:
        """Pool [B, T, C] frame embeddings into one [B, C] vector."""
        if not self.enabled or self.score is None:
            return sequence.mean(dim=1)
        weights = self.score(sequence).softmax(dim=1)
        return (sequence * weights).sum(dim=1)


class AnatomyGuidedSpatialPool(nn.Module):
    """Pool a region using content attention anchored by a vertical prior."""

    def __init__(
        self,
        channels: int,
        initial_center: float,
        sigma: float,
    ) -> None:
        super().__init__()
        if not 0.0 < initial_center < 1.0:
            raise ValueError("initial_center must be strictly between 0 and 1.")
        self.content_score = nn.Conv2d(channels, 1, kernel_size=1)
        center_logit = math.log(initial_center / (1.0 - initial_center))
        self.center_logit = nn.Parameter(torch.tensor(center_logit))
        self.prior_strength_raw = nn.Parameter(torch.tensor(0.5413))
        self.sigma = sigma

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        _, _, height, width = features.shape
        rows = torch.linspace(
            0.0,
            1.0,
            height,
            device=features.device,
            dtype=features.dtype,
        ).view(1, 1, height, 1)
        center = self.center_logit.sigmoid()
        prior = -0.5 * ((rows - center) / self.sigma).square()
        strength = torch.nn.functional.softplus(self.prior_strength_raw)
        logits = self.content_score(features) + strength * prior
        attention = logits.flatten(2).softmax(dim=-1).view(
            -1, 1, height, width
        )
        pooled = (features * attention).sum(dim=(-2, -1))
        return pooled, attention


def hard_region_pool(features: Tensor, region: str) -> tuple[Tensor, Tensor]:
    """Pool a fixed top, middle, or bottom third of the feature map."""
    _, _, height, width = features.shape
    boundaries = torch.linspace(0, height, 4).round().long().tolist()
    region_index = {"apex": 0, "middle": 1, "outflow": 2}[region]
    start, stop = boundaries[region_index], boundaries[region_index + 1]
    mask = features.new_zeros((1, 1, height, width))
    mask[:, :, start:stop, :] = 1.0 / max((stop - start) * width, 1)
    pooled = (features * mask).sum(dim=(-2, -1))
    return pooled, mask.expand(features.shape[0], -1, -1, -1)
