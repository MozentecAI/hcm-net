"""Leakage-safe patient-level multilabel fold generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


STRATIFICATION_COLUMNS = ("outflow", "middle", "apex", "overall_fine")


def _patient_multihot(labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Represent every patient's observed label states as a multilabel vector."""
    feature_names = [
        *(f"outflow={value}" for value in (1, 2, 3)),
        *(f"middle={value}" for value in (1, 2, 3)),
        *(f"apex={value}" for value in (1, 3)),
        *(f"overall_fine={value}" for value in ("none", "latent", "resting")),
    ]
    patients = labels["patient_id"].drop_duplicates().to_numpy()
    patient_rows = labels.groupby("patient_id", sort=False)
    vectors = np.zeros((len(patients), len(feature_names)), dtype=np.int64)

    for row_index, patient_id in enumerate(patients):
        group = patient_rows.get_group(patient_id)
        values = (
            {f"outflow={value}" for value in group["outflow"].unique()}
            | {f"middle={value}" for value in group["middle"].unique()}
            | {f"apex={value}" for value in group["apex"].unique()}
            | {
                f"overall_fine={value}"
                for value in group["overall_fine"].unique()
            }
        )
        vectors[row_index] = [int(name in values) for name in feature_names]

    return patients, vectors


def make_patient_folds(
    labels: pd.DataFrame,
    num_folds: int,
    seed: int,
) -> pd.DataFrame:
    """Assign one fold to each patient using iterative multilabel stratification."""
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2.")
    missing = set(STRATIFICATION_COLUMNS) - set(labels.columns)
    if missing:
        raise ValueError(f"Missing stratification columns: {sorted(missing)}")

    patients, targets = _patient_multihot(labels)
    splitter = MultilabelStratifiedKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=seed,
    )
    patient_to_fold: dict[str, int] = {}
    dummy_features = np.zeros((len(patients), 1), dtype=np.float32)
    for fold, (_, validation_indices) in enumerate(
        splitter.split(dummy_features, targets)
    ):
        patient_to_fold.update(
            {str(patients[index]): fold for index in validation_indices}
        )

    result = labels.copy()
    result["fold"] = result["patient_id"].map(patient_to_fold)
    if result["fold"].isna().any():
        raise RuntimeError("At least one patient was not assigned to a fold.")
    result["fold"] = result["fold"].astype(int)
    return result
