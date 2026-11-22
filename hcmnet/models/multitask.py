"""Shared-encoder hierarchical multitask video classifier."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from hcmnet.models.attention import (
    AnatomyGuidedSpatialPool,
    TemporalAttentionPool,
    TemporalAttentionPool1D,
    hard_region_pool,
)
from hcmnet.models.backbone import ResNetFrameEncoder


class ClassificationHead(nn.Sequential):
    """Small regularized classifier for a pooled video representation."""

    def __init__(
        self,
        input_features: int,
        output_classes: int,
        dropout: float,
    ) -> None:
        hidden_features = max(input_features // 2, 128)
        super().__init__(
            nn.LayerNorm(input_features),
            nn.Linear(input_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, output_classes),
        )


class HCMMultitaskModel(nn.Module):
    """Predict regional and overall obstruction labels from a single video."""

    REGIONS = ("outflow", "middle", "apex")

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.attention_mode = str(config["attention_mode"])
        if self.attention_mode not in {"global", "hard", "soft"}:
            raise ValueError("attention_mode must be global, hard, or soft.")

        self.pooling_order = str(config.get("pooling_order", "temporal_first"))
        if self.pooling_order not in {"temporal_first", "spatial_first"}:
            raise ValueError(
                "pooling_order must be temporal_first or spatial_first."
            )

        self.backbone = ResNetFrameEncoder(
            name=str(config["backbone"]),
            pretrained=bool(config["pretrained"]),
            input_channels=int(config["input_channels"]),
        )
        self.region_feature = str(config.get("region_feature", "fpn"))
        if self.region_feature not in {"layer4", "layer3", "fpn"}:
            raise ValueError("region_feature must be layer4, layer3, or fpn.")

        if self.region_feature == "fpn":
            region_channels = int(config.get("region_channels", 256))
            self.region_fuse: nn.Module = nn.Sequential(
                nn.Conv2d(
                    self.backbone.mid_channels + self.backbone.out_channels,
                    region_channels,
                    kernel_size=1,
                ),
                nn.BatchNorm2d(region_channels),
                nn.GELU(),
            )
        elif self.region_feature == "layer3":
            region_channels = self.backbone.mid_channels
            self.region_fuse = nn.Identity()
        else:
            region_channels = self.backbone.out_channels
            self.region_fuse = nn.Identity()

        temporal_attention = bool(config["temporal_attention"])
        self.region_pools = nn.ModuleDict(
            {
                region: AnatomyGuidedSpatialPool(
                    region_channels,
                    initial_center=float(config["prior_centers"][region]),
                    sigma=float(config["prior_sigma"]),
                )
                for region in self.REGIONS
            }
        )
        if self.pooling_order == "temporal_first":
            self.temporal_pool = TemporalAttentionPool(
                region_channels,
                enabled=temporal_attention,
            )
            self.overall_temporal_pool = TemporalAttentionPool(
                self.backbone.out_channels,
                enabled=temporal_attention,
            )
        else:
            self.region_temporal_pools = nn.ModuleDict(
                {
                    region: TemporalAttentionPool1D(
                        region_channels,
                        enabled=temporal_attention,
                    )
                    for region in self.REGIONS
                }
            )
            self.overall_temporal_pool = TemporalAttentionPool1D(
                self.backbone.out_channels,
                enabled=temporal_attention,
            )
        dropout = float(config["dropout"])
        self.flat_heads = bool(config.get("flat_heads", False))
        if self.flat_heads:
            # No-hierarchy ablation: one flat 3-class head (none/latent/resting)
            # per hierarchical region instead of the coarse+fine head pair.
            self.heads = nn.ModuleDict(
                {
                    "outflow_flat": ClassificationHead(region_channels, 3, dropout),
                    "middle_flat": ClassificationHead(region_channels, 3, dropout),
                    "apex_coarse": ClassificationHead(region_channels, 2, dropout),
                    "overall_coarse": ClassificationHead(
                        self.backbone.out_channels, 2, dropout
                    ),
                    "overall_fine": ClassificationHead(
                        self.backbone.out_channels, 3, dropout
                    ),
                }
            )
        else:
            self.heads = nn.ModuleDict(
                {
                    "outflow_coarse": ClassificationHead(
                        region_channels, 2, dropout
                    ),
                    "outflow_fine": ClassificationHead(
                        region_channels, 2, dropout
                    ),
                    "middle_coarse": ClassificationHead(
                        region_channels, 2, dropout
                    ),
                    "middle_fine": ClassificationHead(
                        region_channels, 2, dropout
                    ),
                    "apex_coarse": ClassificationHead(region_channels, 2, dropout),
                    "overall_coarse": ClassificationHead(
                        self.backbone.out_channels, 2, dropout
                    ),
                    "overall_fine": ClassificationHead(
                        self.backbone.out_channels, 3, dropout
                    ),
                }
            )

    def _region_feature_map(self, mid: Tensor, high: Tensor) -> Tensor:
        if self.region_feature == "layer3":
            return mid
        if self.region_feature == "layer4":
            return high
        batch_size, frames = mid.shape[0], mid.shape[1]
        upsampled = F.interpolate(
            high.flatten(0, 1),
            size=mid.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fused = self.region_fuse(torch.cat([mid.flatten(0, 1), upsampled], dim=1))
        return fused.reshape(batch_size, frames, *fused.shape[1:])

    def _pool_region(
        self,
        feature_map: Tensor,
        region: str,
    ) -> tuple[Tensor, Tensor]:
        if self.attention_mode == "soft":
            return self.region_pools[region](feature_map)
        if self.attention_mode == "hard":
            return hard_region_pool(feature_map, region)

        batch_size, _, height, width = feature_map.shape
        attention = feature_map.new_full(
            (batch_size, 1, height, width),
            1.0 / (height * width),
        )
        return feature_map.mean(dim=(-2, -1)), attention

    def _forward_temporal_first(
        self,
        mid: Tensor,
        high: Tensor,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], Tensor]:
        """Collapse time per spatial location, then pool each region."""
        region_map = self.temporal_pool(self._region_feature_map(mid, high))
        overall_map = self.overall_temporal_pool(high)

        representations: dict[str, Tensor] = {}
        attention_maps: dict[str, Tensor] = {}
        for region in self.REGIONS:
            representations[region], attention_maps[region] = self._pool_region(
                region_map, region
            )
        overall = overall_map.mean(dim=(-2, -1))
        return representations, attention_maps, overall

    def _forward_spatial_first(
        self,
        mid: Tensor,
        high: Tensor,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], Tensor]:
        """Pool each region per frame, then model each region over time.

        Every region keeps its own frame-embedding sequence, so each branch
        can learn region-specific temporal emphasis (for example systolic
        frames for the outflow tract) instead of sharing one collapsed map.
        """
        region_map = self._region_feature_map(mid, high)
        batch_size, frames = region_map.shape[0], region_map.shape[1]
        flat_map = region_map.flatten(0, 1)

        representations: dict[str, Tensor] = {}
        attention_maps: dict[str, Tensor] = {}
        for region in self.REGIONS:
            pooled, attention = self._pool_region(flat_map, region)
            sequence = pooled.view(batch_size, frames, -1)
            representations[region] = self.region_temporal_pools[region](sequence)
            attention_maps[region] = attention.view(
                batch_size, frames, 1, *region_map.shape[-2:]
            ).mean(dim=1)

        overall_sequence = high.mean(dim=(-2, -1))
        overall = self.overall_temporal_pool(overall_sequence)
        return representations, attention_maps, overall

    def forward(self, video: Tensor) -> dict[str, Any]:
        mid, high = self.backbone(video)
        if self.pooling_order == "spatial_first":
            representations, attention_maps, overall = self._forward_spatial_first(
                mid, high
            )
        else:
            representations, attention_maps, overall = self._forward_temporal_first(
                mid, high
            )

        if self.flat_heads:
            logits = {
                "outflow_flat": self.heads["outflow_flat"](
                    representations["outflow"]
                ),
                "middle_flat": self.heads["middle_flat"](
                    representations["middle"]
                ),
                "apex_coarse": self.heads["apex_coarse"](representations["apex"]),
                "overall_coarse": self.heads["overall_coarse"](overall),
                "overall_fine": self.heads["overall_fine"](overall),
            }
        else:
            logits = {
                "outflow_coarse": self.heads["outflow_coarse"](
                    representations["outflow"]
                ),
                "outflow_fine": self.heads["outflow_fine"](
                    representations["outflow"]
                ),
                "middle_coarse": self.heads["middle_coarse"](
                    representations["middle"]
                ),
                "middle_fine": self.heads["middle_fine"](
                    representations["middle"]
                ),
                "apex_coarse": self.heads["apex_coarse"](representations["apex"]),
                "overall_coarse": self.heads["overall_coarse"](overall),
                "overall_fine": self.heads["overall_fine"](overall),
            }
        return {"logits": logits, "attention_maps": attention_maps}
