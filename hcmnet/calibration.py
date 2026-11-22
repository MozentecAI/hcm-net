"""Temperature scaling and decision-threshold calibration for binary tasks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import functional as F

from hcmnet.metrics import (
    BINARY_TASKS,
    cascade_probabilities,
    derived_overall_logits,
)
from hcmnet.prediction import CLASS_NAMES

POSITIVE_CLASS = {
    "outflow_coarse": "obstruct",
    "outflow_fine": "latent",
    "middle_coarse": "obstruct",
    "middle_fine": "latent",
    "apex_coarse": "obstruct",
    "overall_coarse": "unhealthy",
    "overall_derived": "unhealthy",
}

CALIBRATED_TASKS = (*BINARY_TASKS, "overall_fine")


def collect_logits_and_targets(
    batches: list[dict[str, Any]],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Stack per-sample logits and targets from Lightning predict batches."""
    logits: dict[str, list[Tensor]] = {task: [] for task in CALIBRATED_TASKS}
    targets: dict[str, list[Tensor]] = {task: [] for task in CALIBRATED_TASKS}
    for batch in batches:
        for task in CALIBRATED_TASKS:
            logits[task].append(batch["logits"][task].detach().cpu().float())
            targets[task].append(batch["targets"][task].detach().cpu())
    return (
        {task: torch.cat(values) for task, values in logits.items()},
        {task: torch.cat(values) for task, values in targets.items()},
    )


def fit_temperature(logits: Tensor, targets: Tensor) -> float:
    """Fit a single temperature by minimizing NLL with LBFGS."""
    if logits.shape[0] < 2:
        return 1.0
    log_temperature = torch.zeros(1, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], max_iter=50, line_search_fn="strong_wolfe"
    )

    def closure() -> Tensor:
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_temperature.exp(), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().item())
    return float(np.clip(temperature, 0.1, 10.0))


def fit_threshold(
    probabilities: Tensor,
    targets: Tensor,
    criterion: str = "youden",
) -> float:
    """Scan candidate thresholds and pick the best Youden's J or F1."""
    candidates = torch.linspace(0.01, 0.99, 197)
    predictions = probabilities.unsqueeze(1) >= candidates.unsqueeze(0)
    truth = targets.bool().unsqueeze(1)
    tp = (predictions & truth).sum(dim=0).float()
    fp = (predictions & ~truth).sum(dim=0).float()
    fn = (~predictions & truth).sum(dim=0).float()
    tn = (~predictions & ~truth).sum(dim=0).float()
    if criterion == "youden":
        sensitivity = tp / (tp + fn).clamp_min(1.0)
        specificity = tn / (tn + fp).clamp_min(1.0)
        score = sensitivity + specificity - 1.0
    elif criterion == "f1":
        score = 2.0 * tp / (2.0 * tp + fp + fn).clamp_min(1.0)
    else:
        raise ValueError(f"Unsupported threshold criterion '{criterion}'.")
    return float(candidates[int(score.argmax().item())].item())


def fit_calibration(
    batches: list[dict[str, Any]],
    criterion: str = "youden",
    fit_temperatures: bool = True,
) -> dict[str, Any]:
    """Fit per-task temperatures and binary thresholds on predict outputs."""
    logits, targets = collect_logits_and_targets(batches)
    tasks: dict[str, Any] = {}
    for task in BINARY_TASKS:
        mask = targets[task] >= 0
        task_logits = logits[task][mask]
        task_targets = targets[task][mask]
        # overall_derived is recomputed from the regional logits when
        # calibration is applied, so an independent temperature is unused.
        temperature = (
            fit_temperature(task_logits, task_targets)
            if fit_temperatures and task != "overall_derived"
            else 1.0
        )
        positive = (task_logits / temperature).softmax(dim=-1)[:, 1]
        tasks[task] = {
            "temperature": temperature,
            "threshold": fit_threshold(positive, task_targets, criterion),
            "num_samples": int(mask.sum().item()),
        }
    overall_logits = logits["overall_fine"]
    overall_targets = targets["overall_fine"]
    tasks["overall_fine"] = {
        "temperature": (
            fit_temperature(overall_logits, overall_targets)
            if fit_temperatures
            else 1.0
        ),
        "num_samples": int(overall_targets.shape[0]),
    }
    return {"criterion": criterion, "tasks": tasks}


def apply_calibration_to_batches(
    batches: list[dict[str, Any]],
    calibration: dict[str, Any],
) -> list[dict[str, Any]]:
    """Temperature-scale logits and rebuild probabilities for every batch."""
    task_calibration = calibration.get("tasks", calibration)
    calibrated = []
    for batch in batches:
        logits = batch["logits"]
        scaled = {}
        for task, values in logits.items():
            if task == "overall_derived":
                continue
            entry = task_calibration.get(task)
            temperature = float(entry["temperature"]) if entry else 1.0
            scaled[task] = values / temperature
        scaled["overall_derived"] = derived_overall_logits(scaled)
        probabilities = {
            task: values.softmax(dim=-1) for task, values in scaled.items()
        }
        for region in ("outflow", "middle"):
            probabilities[f"{region}_cascade"] = cascade_probabilities(
                scaled[f"{region}_coarse"],
                scaled[f"{region}_fine"],
            )
        calibrated.append(
            {**batch, "probabilities": probabilities, "logits": scaled}
        )
    return calibrated


def apply_thresholds_to_frame(
    frame: pd.DataFrame,
    calibration: dict[str, Any],
) -> pd.DataFrame:
    """Add threshold-based calibrated predictions for every binary task."""
    task_calibration = calibration.get("tasks", calibration)
    result = frame.copy()
    for task in BINARY_TASKS:
        entry = task_calibration.get(task)
        if entry is None or "threshold" not in entry:
            continue
        positive = POSITIVE_CLASS[task]
        negative = next(name for name in CLASS_NAMES[task] if name != positive)
        threshold = float(entry["threshold"])
        result[f"pred_{task}_calibrated"] = np.where(
            result[f"p_{task}_{positive}"] >= threshold,
            positive,
            negative,
        )
        result[f"threshold_{task}"] = threshold
    return result


def thresholded_metrics_from_frame(
    frame: pd.DataFrame,
    calibration: dict[str, Any],
) -> dict[str, float]:
    """Recompute threshold-dependent binary metrics at calibrated thresholds."""
    task_calibration = calibration.get("tasks", calibration)
    metrics: dict[str, float] = {}
    for task in BINARY_TASKS:
        entry = task_calibration.get(task)
        if entry is None or "threshold" not in entry:
            continue
        target_column = f"target_{task}"
        mask = frame[target_column].notna()
        if not mask.any():
            continue
        truth = frame.loc[mask, target_column].astype(int).to_numpy()
        probability = frame.loc[mask, f"p_{task}_{POSITIVE_CLASS[task]}"].to_numpy()
        prediction = (probability >= float(entry["threshold"])).astype(int)

        tp = float(((prediction == 1) & (truth == 1)).sum())
        tn = float(((prediction == 0) & (truth == 0)).sum())
        fp = float(((prediction == 1) & (truth == 0)).sum())
        fn = float(((prediction == 0) & (truth == 1)).sum())
        sensitivity = tp / max(tp + fn, 1.0)
        specificity = tn / max(tn + fp, 1.0)
        ppv = tp / max(tp + fp, 1.0)
        npv = tn / max(tn + fn, 1.0)
        f1_positive = 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)
        f1_negative = 2.0 * tn / max(2.0 * tn + fn + fp, 1.0)
        prefix = f"calibrated/{task}"
        metrics[f"{prefix}/accuracy"] = (tp + tn) / max(tp + tn + fp + fn, 1.0)
        metrics[f"{prefix}/sensitivity"] = sensitivity
        metrics[f"{prefix}/specificity"] = specificity
        metrics[f"{prefix}/ppv"] = ppv
        metrics[f"{prefix}/npv"] = npv
        metrics[f"{prefix}/f1"] = f1_positive
        metrics[f"{prefix}/macro_f1"] = 0.5 * (f1_positive + f1_negative)
        metrics[f"{prefix}/threshold"] = float(entry["threshold"])
    return metrics
