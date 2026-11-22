"""Validation metrics for hierarchical and cascaded predictions."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAccuracy,
    BinaryAveragePrecision,
    BinaryCalibrationError,
    BinaryF1Score,
    BinaryNegativePredictiveValue,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
    MulticlassAUROC,
    MulticlassAccuracy,
    MulticlassAveragePrecision,
    MulticlassCalibrationError,
    MulticlassF1Score,
    MulticlassNegativePredictiveValue,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassSpecificity,
)


BINARY_TASKS = (
    "outflow_coarse",
    "outflow_fine",
    "middle_coarse",
    "middle_fine",
    "apex_coarse",
    "overall_coarse",
    "overall_derived",
)
MULTICLASS_TASKS = {
    "overall_fine": 3,
    "outflow_cascade": 3,
    "middle_cascade": 3,
}
COARSE_AUC_TASKS = (
    "outflow_coarse",
    "middle_coarse",
    "apex_coarse",
    "overall_derived",
)
# v2 clinical definition: the overall obstruction status is defined by LVOT
# (outflow) and midventricular (middle) obstruction only; the mid-apical
# phenotype is nested within the MVO spectrum and does not contribute
# independently. Used by the soft-OR derived overall score.
REGIONS = ("outflow", "middle")


def _binary_metrics() -> MetricCollection:
    return MetricCollection(
        {
            "accuracy": BinaryAccuracy(),
            "auroc": BinaryAUROC(),
            "auprc": BinaryAveragePrecision(),
            "f1": BinaryF1Score(),
            "sensitivity": BinaryRecall(),
            "specificity": BinarySpecificity(),
            "ppv": BinaryPrecision(),
            "npv": BinaryNegativePredictiveValue(),
            "ece": BinaryCalibrationError(n_bins=15, norm="l1"),
        }
    )


def _multiclass_metrics(num_classes: int) -> MetricCollection:
    return MetricCollection(
        {
            "accuracy": MulticlassAccuracy(
                num_classes=num_classes, average="micro"
            ),
            "auroc": MulticlassAUROC(
                num_classes=num_classes, average="macro"
            ),
            "auprc": MulticlassAveragePrecision(
                num_classes=num_classes, average="macro"
            ),
            "macro_f1": MulticlassF1Score(
                num_classes=num_classes, average="macro"
            ),
            "sensitivity": MulticlassRecall(
                num_classes=num_classes, average="macro"
            ),
            "specificity": MulticlassSpecificity(
                num_classes=num_classes, average="macro"
            ),
            "ppv": MulticlassPrecision(
                num_classes=num_classes, average="macro"
            ),
            "npv": MulticlassNegativePredictiveValue(
                num_classes=num_classes, average="macro"
            ),
            "ece": MulticlassCalibrationError(
                num_classes=num_classes, n_bins=15, norm="l1"
            ),
        }
    )


def cascade_probabilities(coarse_logits: Tensor, fine_logits: Tensor) -> Tensor:
    """Compose coarse and conditional fine probabilities into three classes."""
    coarse = coarse_logits.softmax(dim=-1)
    fine = fine_logits.softmax(dim=-1)
    return torch.stack(
        [
            coarse[:, 1] * fine[:, 0],
            coarse[:, 1] * fine[:, 1],
            coarse[:, 0],
        ],
        dim=1,
    )


def flat_to_coarse_fine_logits(flat: Tensor) -> tuple[Tensor, Tensor]:
    """Expand flat 3-class regional logits into hierarchical coarse/fine pairs.

    The flat head orders classes as (none, latent, resting); it is used by the
    no-hierarchy ablation. Coarse obstruction is the latent+resting margin and
    the fine pair renormalizes within obstruction, matching CLASS_NAMES.
    """
    epsilon = torch.finfo(torch.float32).eps
    probabilities = flat.float().softmax(dim=-1)
    p_none, p_latent, p_resting = (
        probabilities[:, 0],
        probabilities[:, 1],
        probabilities[:, 2],
    )
    p_obstruct = p_latent + p_resting
    coarse = torch.stack([p_none, p_obstruct], dim=1).clamp_min(epsilon).log()
    denominator = p_obstruct.clamp_min(epsilon)
    fine = torch.stack(
        [p_resting / denominator, p_latent / denominator], dim=1
    ).clamp_min(epsilon).log()
    return coarse, fine


def derived_overall_probabilities(logits: dict[str, Tensor]) -> Tensor:
    """Soft-OR the regional obstruction probabilities into an overall pair.

    Mirrors the deterministic clinical rule (any regional obstruction means
    unhealthy) with a differentiable product t-norm, so the deployed overall
    score inherits the regional heads' discrimination directly.
    """
    regional_obstruction = [
        logits[f"{region}_coarse"].float().softmax(dim=-1)[:, 1]
        for region in REGIONS
    ]
    derived_obstruction = 1.0 - torch.stack(
        [1.0 - probability for probability in regional_obstruction],
        dim=1,
    ).prod(dim=1)
    return torch.stack(
        [1.0 - derived_obstruction, derived_obstruction],
        dim=1,
    )


def derived_overall_logits(logits: dict[str, Tensor]) -> Tensor:
    """Log-probabilities of the soft-OR overall prediction."""
    epsilon = torch.finfo(torch.float32).eps
    return derived_overall_probabilities(logits).clamp_min(epsilon).log()


class ValidationMetrics(nn.Module):
    """Accumulate distributed-safe metrics over a validation epoch."""

    def __init__(self) -> None:
        super().__init__()
        self.updated_tasks: set[str] = set()
        self.binary = nn.ModuleDict(
            {task: _binary_metrics() for task in BINARY_TASKS}
        )
        self.binary_macro_f1 = nn.ModuleDict(
            {
                task: MulticlassF1Score(num_classes=2, average="macro")
                for task in BINARY_TASKS
            }
        )
        self.multiclass = nn.ModuleDict(
            {
                task: _multiclass_metrics(num_classes)
                for task, num_classes in MULTICLASS_TASKS.items()
            }
        )
        self.brier = nn.ModuleDict(
            {
                **{task: MeanMetric() for task in BINARY_TASKS},
                **{task: MeanMetric() for task in MULTICLASS_TASKS},
            }
        )
        self.logic_disagreement = MeanMetric()

    def _update_binary(
        self,
        task: str,
        logits: Tensor,
        target: Tensor,
    ) -> None:
        probability = logits.softmax(dim=-1)[:, 1]
        self.updated_tasks.add(task)
        self.binary[task].update(probability, target)
        self.binary_macro_f1[task].update(logits.softmax(dim=-1), target)
        self.brier[task].update((probability - target.float()).square())

    def _update_multiclass(
        self,
        task: str,
        probabilities: Tensor,
        target: Tensor,
    ) -> None:
        self.updated_tasks.add(task)
        self.multiclass[task].update(probabilities, target)
        one_hot = F.one_hot(
            target, num_classes=probabilities.shape[1]
        ).float()
        self.brier[task].update((probabilities - one_hot).square().sum(dim=1))

    def update(
        self,
        logits: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> None:
        for task in BINARY_TASKS:
            if task == "overall_derived":
                continue
            target = targets[task]
            if task.endswith("_fine"):
                mask = target >= 0
                if not mask.any():
                    continue
                self._update_binary(task, logits[task][mask], target[mask])
            else:
                self._update_binary(task, logits[task], target)

        self._update_binary(
            "overall_derived",
            derived_overall_logits(logits),
            targets["overall_derived"],
        )
        self._update_multiclass(
            "overall_fine",
            logits["overall_fine"].softmax(dim=-1),
            targets["overall_fine"],
        )
        for region in ("outflow", "middle"):
            cascade = cascade_probabilities(
                logits[f"{region}_coarse"],
                logits[f"{region}_fine"],
            )
            cascade_target = torch.where(
                targets[f"{region}_coarse"] == 0,
                torch.full_like(targets[f"{region}_coarse"], 2),
                targets[f"{region}_fine"],
            )
            self._update_multiclass(
                f"{region}_cascade", cascade, cascade_target
            )

        regional_prediction = torch.stack(
            [
                logits[f"{region}_coarse"].argmax(dim=1).bool()
                for region in REGIONS
            ],
            dim=1,
        ).any(dim=1)
        overall_prediction = logits["overall_coarse"].argmax(dim=1).bool()
        self.logic_disagreement.update(
            (regional_prediction != overall_prediction).float()
        )

    def reset(self) -> None:
        """Discard all accumulated validation state without computing it."""
        for task, metrics in self.binary.items():
            metrics.reset()
        for task, metrics in self.binary_macro_f1.items():
            metrics.reset()
        for task, metrics in self.multiclass.items():
            metrics.reset()
        for task, metrics in self.brier.items():
            metrics.reset()
        self.logic_disagreement.reset()
        self.updated_tasks.clear()

    def compute_and_reset(self) -> tuple[dict[str, Tensor], Tensor]:
        results: dict[str, Tensor] = {}
        f1_values: list[Tensor] = []
        for task, metrics in self.binary.items():
            if task not in self.updated_tasks:
                continue
            computed = metrics.compute()
            for name, value in computed.items():
                results[f"val/{task}/{name}"] = value
            macro_f1 = self.binary_macro_f1[task].compute()
            results[f"val/{task}/macro_f1"] = macro_f1
            results[f"val/{task}/brier"] = self.brier[task].compute()
            f1_values.append(macro_f1)

        for task, metrics in self.multiclass.items():
            if task not in self.updated_tasks:
                continue
            computed = metrics.compute()
            for name, value in computed.items():
                results[f"val/{task}/{name}"] = value
            results[f"val/{task}/brier"] = self.brier[task].compute()
            f1_values.append(computed["macro_f1"])

        results["val/logic_disagreement"] = self.logic_disagreement.compute()
        coarse_auroc = [
            results[f"val/{task}/auroc"]
            for task in COARSE_AUC_TASKS
            if f"val/{task}/auroc" in results
        ]
        if coarse_auroc:
            results["val/mean_coarse_auroc"] = torch.stack(coarse_auroc).mean()
        if not f1_values:
            raise RuntimeError("No validation samples were accumulated.")
        mean_f1 = torch.stack(f1_values).mean()
        results["val/mean_f1"] = mean_f1
        self.reset()
        return results, mean_f1
