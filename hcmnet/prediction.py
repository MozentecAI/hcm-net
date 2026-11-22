"""Shared sample-level prediction formatting and metric accumulation."""

from __future__ import annotations

from typing import Any

import pandas as pd
from torch import Tensor

from hcmnet.metrics import BINARY_TASKS, ValidationMetrics


CLASS_NAMES = {
    "outflow_coarse": ("none", "obstruct"),
    "outflow_fine": ("resting", "latent"),
    "middle_coarse": ("none", "obstruct"),
    "middle_fine": ("resting", "latent"),
    "apex_coarse": ("none", "obstruct"),
    "overall_coarse": ("healthy", "unhealthy"),
    "overall_derived": ("healthy", "unhealthy"),
    "overall_fine": ("none", "latent", "resting"),
    "outflow_cascade": ("resting", "latent", "none"),
    "middle_cascade": ("resting", "latent", "none"),
}


def prediction_rows(
    batches: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Flatten Lightning prediction batches and calculate evaluation metrics."""
    rows: list[dict[str, Any]] = []
    metrics = ValidationMetrics()

    for batch in batches:
        probabilities: dict[str, Tensor] = {
            name: value.detach().cpu()
            for name, value in batch["probabilities"].items()
        }
        targets: dict[str, Tensor] = {
            name: value.detach().cpu() for name, value in batch["targets"].items()
        }
        logits = {
            task: probabilities[task].clamp_min(1e-12).log()
            for task in (*BINARY_TASKS, "overall_fine")
        }
        metrics.update(logits, targets)

        image_ids = [str(value) for value in batch["image_id"]]
        patient_ids = [str(value) for value in batch["patient_id"]]
        for index, image_id in enumerate(image_ids):
            row: dict[str, Any] = {
                "image_id": image_id,
                "_predicted_patient_id": patient_ids[index],
            }
            for task, class_names in CLASS_NAMES.items():
                task_probabilities = probabilities[task][index]
                prediction_index = int(task_probabilities.argmax().item())
                row[f"pred_{task}"] = class_names[prediction_index]
                row[f"confidence_{task}"] = float(
                    task_probabilities[prediction_index].item()
                )
                for class_index, class_name in enumerate(class_names):
                    row[f"p_{task}_{class_name}"] = float(
                        task_probabilities[class_index].item()
                    )

            for task, values in targets.items():
                value = int(values[index].item())
                row[f"target_{task}"] = None if value < 0 else value
            for region in ("outflow", "middle"):
                coarse = int(targets[f"{region}_coarse"][index].item())
                fine = int(targets[f"{region}_fine"][index].item())
                row[f"target_{region}_cascade"] = 2 if coarse == 0 else fine

            # v2: the QC flag mirrors the LVOTO∨MVO overall definition; the
            # mid-apical phenotype is nested in MVO and not counted separately.
            regional_obstruction = any(
                row[f"pred_{region}_coarse"] == "obstruct"
                for region in ("outflow", "middle")
            )
            overall_unhealthy = row["pred_overall_coarse"] == "unhealthy"
            row["logic_disagreement"] = int(
                regional_obstruction != overall_unhealthy
            )
            rows.append(row)

    metric_tensors, _ = metrics.compute_and_reset()
    metric_values = {
        name: float(value.detach().cpu().item())
        for name, value in metric_tensors.items()
    }
    return pd.DataFrame(rows), metric_values
