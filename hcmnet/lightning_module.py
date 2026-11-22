"""PyTorch Lightning 2.6 training module."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from torch import Tensor

from hcmnet.losses import (
    effective_number_weights,
    focal_cross_entropy,
    regional_to_global_consistency,
)
from hcmnet.metrics import (
    ValidationMetrics,
    cascade_probabilities,
    derived_overall_probabilities,
    flat_to_coarse_fine_logits,
)
from hcmnet.models import HCMMultitaskModel


class HCMLitModule(pl.LightningModule):
    """Optimize all hierarchical heads and report validation metrics."""

    COARSE_TASKS = (
        "outflow_coarse",
        "middle_coarse",
        "apex_coarse",
        "overall_coarse",
    )
    FINE_TASKS = ("outflow_fine", "middle_fine")

    def __init__(
        self,
        model_config: dict[str, Any],
        loss_config: dict[str, Any],
        optimizer_config: dict[str, Any],
        class_counts: dict[str, list[int]],
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = HCMMultitaskModel(model_config)
        self.flat_heads = bool(model_config.get("flat_heads", False))
        self.loss_config = loss_config
        self.optimizer_config = optimizer_config
        self.validation_metrics = ValidationMetrics()

        beta = float(loss_config["effective_number_beta"])
        for task, counts in class_counts.items():
            self.register_buffer(
                f"alpha_{task}",
                effective_number_weights(counts, beta),
            )

    def forward(self, video: Tensor) -> dict[str, Any]:
        return self.model(video)

    def _video_logits(self, video: Tensor) -> dict[str, Tensor]:
        """Run the model, averaging logits over the clip dimension if present."""
        if video.dim() == 6:
            batch_size, num_clips = video.shape[0], video.shape[1]
            flat_logits = self.model(video.flatten(0, 1))["logits"]
            return {
                task: values.reshape(batch_size, num_clips, -1).mean(dim=1)
                for task, values in flat_logits.items()
            }
        return self.model(video)["logits"]

    def _task_loss(
        self,
        task: str,
        logits: Tensor,
        target: Tensor,
    ) -> Tensor:
        return focal_cross_entropy(
            logits,
            target,
            alpha=getattr(self, f"alpha_{task}"),
            gamma=float(self.loss_config["focal_gamma"]),
        )

    def _hierarchical_logits(
        self, logits: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """Expand flat ablation heads into the standard coarse/fine keys."""
        if not self.flat_heads:
            return logits
        expanded = dict(logits)
        for region in ("outflow", "middle"):
            coarse, fine = flat_to_coarse_fine_logits(logits[f"{region}_flat"])
            expanded[f"{region}_coarse"] = coarse
            expanded[f"{region}_fine"] = fine
        return expanded

    def _compute_losses(
        self,
        logits: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        losses: dict[str, Tensor] = {}
        hierarchical = self._hierarchical_logits(logits)
        if self.flat_heads:
            # No-hierarchy ablation: a single flat 3-class loss per region
            # replaces the coarse+fine pair; it is weighted like a coarse loss.
            coarse_losses = []
            for region in ("outflow", "middle"):
                task = f"{region}_flat"
                losses[task] = self._task_loss(
                    task, logits[task], targets[task]
                )
                coarse_losses.append(losses[task])
            for task in ("apex_coarse", "overall_coarse"):
                losses[task] = self._task_loss(task, logits[task], targets[task])
                coarse_losses.append(losses[task])
            fine_losses = []
        else:
            coarse_losses = []
            for task in self.COARSE_TASKS:
                losses[task] = self._task_loss(task, logits[task], targets[task])
                coarse_losses.append(losses[task])

            fine_losses = []
            for task in self.FINE_TASKS:
                mask = targets[task] >= 0
                if mask.any():
                    losses[task] = self._task_loss(
                        task, logits[task][mask], targets[task][mask]
                    )
                    fine_losses.append(losses[task])
                else:
                    losses[task] = logits[task].sum() * 0.0

        losses["overall_fine"] = self._task_loss(
            "overall_fine",
            logits["overall_fine"],
            targets["overall_fine"],
        )
        losses["consistency"] = regional_to_global_consistency(hierarchical)

        fine_total = (
            torch.stack(fine_losses).sum()
            if fine_losses
            else torch.zeros((), device=logits["overall_fine"].device)
        )
        total = (
            float(self.loss_config["coarse_weight"]) * torch.stack(coarse_losses).sum()
            + float(self.loss_config["fine_weight"]) * fine_total
            + float(self.loss_config["overall_fine_weight"])
            * losses["overall_fine"]
            + float(self.loss_config["consistency_weight"])
            * losses["consistency"]
        )
        losses["total"] = total
        return total, losses

    def _shared_step(
        self,
        batch: dict[str, Any],
        stage: str,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], Tensor]:
        logits = self._video_logits(batch["video"])
        total_loss, losses = self._compute_losses(logits, batch["targets"])
        batch_size = batch["video"].shape[0]
        self.log(
            f"{stage}/loss",
            total_loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        for name, loss in losses.items():
            if name == "total":
                continue
            self.log(
                f"{stage}/loss_{name}",
                loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        return logits, batch["targets"], total_loss

    def training_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> Tensor:
        logits, targets, total_loss = self._shared_step(batch, "train")
        del logits, targets
        return total_loss

    def validation_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> None:
        logits, targets, _ = self._shared_step(batch, "val")
        self.validation_metrics.update(
            self._hierarchical_logits(logits), targets
        )

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            self.validation_metrics.reset()
            return
        metrics, _ = self.validation_metrics.compute_and_reset()
        self.log_dict(metrics, prog_bar=False, sync_dist=False)
        if self.trainer.is_global_zero:
            self._print_and_save_validation_metrics(metrics)

    def _print_and_save_validation_metrics(
        self,
        metrics: dict[str, Tensor],
    ) -> None:
        """Print validation metrics and append the same report to the run log."""
        grouped: dict[str, dict[str, float]] = {}
        for key, value in metrics.items():
            parts = key.split("/")
            if len(parts) == 3:
                _, task, metric_name = parts
            else:
                task, metric_name = "summary", parts[-1]
            grouped.setdefault(task, {})[metric_name] = float(
                value.detach().cpu()
            )

        callback_metrics = self.trainer.callback_metrics
        for key, value in callback_metrics.items():
            if not key.startswith("val/loss") or not isinstance(value, Tensor):
                continue
            grouped.setdefault("losses", {})[key.removeprefix("val/")] = float(
                value.detach().cpu()
            )

        lines = [
            "",
            f"[Validation metrics] epoch={self.current_epoch + 1}",
        ]
        for task in sorted(grouped):
            values = ", ".join(
                f"{name}={value:.6f}"
                for name, value in sorted(grouped[task].items())
            )
            lines.append(f"{task}: {values}")
        report = "\n".join(lines)
        print(report, flush=True)

        log_dir = getattr(self.trainer.logger, "log_dir", None)
        if log_dir is None:
            return
        output_path = Path(log_dir) / "validation_metrics.log"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{report}\n")

    def predict_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Any]:
        logits = {
            task: values.float()
            for task, values in self._hierarchical_logits(
                self._video_logits(batch["video"])
            ).items()
            if not task.endswith("_flat")
        }
        probabilities = {
            task: task_logits.softmax(dim=-1)
            for task, task_logits in logits.items()
        }
        for region in ("outflow", "middle"):
            probabilities[f"{region}_cascade"] = cascade_probabilities(
                logits[f"{region}_coarse"],
                logits[f"{region}_fine"],
            )
        derived = derived_overall_probabilities(logits)
        probabilities["overall_derived"] = derived
        logits["overall_derived"] = derived.clamp_min(
            torch.finfo(derived.dtype).eps
        ).log()
        return {
            "image_id": batch["image_id"],
            "patient_id": batch["patient_id"],
            "probabilities": probabilities,
            "targets": batch["targets"],
            "logits": logits,
        }

    def configure_optimizers(self) -> dict[str, Any]:
        backbone_parameters = list(self.model.backbone.parameters())
        backbone_parameter_ids = {id(parameter) for parameter in backbone_parameters}
        other_parameters = [
            parameter
            for parameter in self.model.parameters()
            if id(parameter) not in backbone_parameter_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": backbone_parameters,
                    "lr": float(self.optimizer_config["backbone_learning_rate"]),
                },
                {
                    "params": other_parameters,
                    "lr": float(self.optimizer_config["learning_rate"]),
                },
            ],
            weight_decay=float(self.optimizer_config["weight_decay"]),
        )

        total_steps = max(int(self.trainer.estimated_stepping_batches), 1)
        max_epochs = max(int(self.trainer.max_epochs), 1)
        warmup_steps = round(
            total_steps
            * float(self.optimizer_config["warmup_epochs"])
            / max_epochs
        )

        def learning_rate_multiplier(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(
                total_steps - warmup_steps, 1
            )
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, learning_rate_multiplier
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
