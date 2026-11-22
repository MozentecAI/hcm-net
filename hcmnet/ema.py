"""Exponential moving average of network weights for evaluation."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
from torch import Tensor


class EmaCallback(pl.Callback):
    """Maintain EMA weights and evaluate with them during validation.

    The shadow weights are swapped in before each validation epoch (which is
    also the window in which checkpoints are saved) and swapped back before
    the next training epoch, so saved checkpoints carry the EMA weights.
    """

    def __init__(self, decay: float = 0.999) -> None:
        super().__init__()
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}.")
        self.decay = float(decay)
        self.shadow: dict[str, Tensor] = {}
        self._backup: dict[str, Tensor] = {}

    @staticmethod
    def _tensors(pl_module: pl.LightningModule) -> dict[str, Tensor]:
        model = pl_module.model
        return {
            **dict(model.named_parameters()),
            **dict(model.named_buffers()),
        }

    def on_fit_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if not self.shadow:
            self.shadow = {
                name: value.detach().clone()
                for name, value in self._tensors(pl_module).items()
            }

    @torch.no_grad()
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if not self.shadow:
            return
        step = trainer.global_step
        decay = min(self.decay, (1.0 + step) / (10.0 + step))
        for name, value in self._tensors(pl_module).items():
            shadow = self.shadow[name]
            if shadow.device != value.device:
                shadow = shadow.to(value.device)
                self.shadow[name] = shadow
            if not torch.is_floating_point(value):
                shadow.copy_(value)
                continue
            shadow.mul_(decay).add_(value.detach(), alpha=1.0 - decay)

    @torch.no_grad()
    def on_validation_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if not self.shadow or self._backup:
            return
        self._backup = {
            name: value.detach().clone()
            for name, value in self._tensors(pl_module).items()
        }
        for name, value in self._tensors(pl_module).items():
            value.copy_(self.shadow[name].to(value.device))

    @torch.no_grad()
    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if not self._backup:
            return
        for name, value in self._tensors(pl_module).items():
            value.copy_(self._backup[name])
        self._backup = {}

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.shadow = {
            name: value for name, value in state_dict["shadow"].items()
        }
