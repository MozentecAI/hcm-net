"""PyTorch Lightning data module for grouped train/validation folds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from hcmnet.data.dataset import (
    FLAT_LABEL_TO_INDEX,
    HCMVideoDataset,
    OVERALL_FINE_TO_INDEX,
)
from hcmnet.data.splitting import make_patient_folds


class ReplicatedEvalSampler(DistributedSampler):
    """Evaluate the complete validation fold on every rank without padding."""

    def __init__(self, dataset: HCMVideoDataset) -> None:
        super().__init__(
            dataset,
            num_replicas=1,
            rank=0,
            shuffle=False,
            drop_last=False,
        )


class HCMDataModule(pl.LightningDataModule):
    """Create patient-disjoint multilabel-stratified train/validation loaders."""

    def __init__(self, config: dict[str, Any], seed: int) -> None:
        super().__init__()
        self.config = config
        self.seed = seed
        self.train_dataset: HCMVideoDataset | None = None
        self.val_dataset: HCMVideoDataset | None = None
        self.fold_labels: pd.DataFrame | None = None
        self.class_counts: dict[str, list[int]] = {}

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None:
            return

        labels_path = Path(self.config["labels_path"])
        data_dir = Path(self.config["data_dir"])
        labels = pd.read_csv(
            labels_path,
            dtype={"image_id": str, "patient_id": str},
        )
        self._validate_labels_and_arrays(labels, data_dir)
        fixed_folds = self.config.get("fixed_fold_assignments")
        if fixed_folds:
            self.fold_labels = self._load_fixed_fold_assignments(
                labels, Path(fixed_folds), int(self.config["num_folds"])
            )
        else:
            self.fold_labels = make_patient_folds(
                labels,
                num_folds=int(self.config["num_folds"]),
                seed=self.seed,
            )
        fold = int(self.config["fold"])
        if not 0 <= fold < int(self.config["num_folds"]):
            raise ValueError(
                f"fold must be in [0, {int(self.config['num_folds']) - 1}]."
            )

        train_labels = self.fold_labels[self.fold_labels["fold"] != fold].copy()
        val_labels = self.fold_labels[self.fold_labels["fold"] == fold].copy()
        self.class_counts = self._get_class_counts(train_labels)

        augmentation = self.config["augmentation"]
        common_arguments = {
            "data_dir": data_dir,
            "num_frames": int(self.config["num_frames"]),
            "image_size": int(self.config["image_size"]),
            "normalization_mean": float(self.config["normalization"]["mean"]),
            "normalization_std": float(self.config["normalization"]["std"]),
            "resize_mode": str(self.config.get("resize_mode", "stretch")),
            "sampling_mode": str(self.config.get("sampling_mode", "window")),
            "cycle_num_cycles": int(self.config.get("cycle_num_cycles", 2)),
            "cycle_period_range": tuple(
                self.config.get("cycle_period_range", (16, 96))
            ),
            "cycle_curve_stride": int(self.config.get("cycle_curve_stride", 2)),
            "eval_phase_bias_strength": float(
                self.config.get("eval_phase_bias_strength", 0.0)
            ),
        }
        self.train_dataset = HCMVideoDataset(
            train_labels,
            training=True,
            temporal_window=tuple(self.config["train_temporal_window"]),
            rotation_degrees=float(augmentation["rotation_degrees"]),
            brightness=float(augmentation["brightness"]),
            contrast=float(augmentation["contrast"]),
            phase_bias_strength=float(self.config.get("phase_bias_strength", 0.0)),
            phase_candidate_multiplier=int(
                self.config.get("phase_candidate_multiplier", 2)
            ),
            temporal_stride_jitter=float(
                self.config.get("temporal_stride_jitter", 0.0)
            ),
            frame_dropout_prob=float(self.config.get("frame_dropout_prob", 0.0)),
            **common_arguments,
        )
        self.val_dataset = HCMVideoDataset(
            val_labels,
            training=False,
            eval_num_clips=int(self.config.get("eval_num_clips", 1)),
            eval_window_ratio=float(self.config.get("eval_window_ratio", 0.75)),
            **common_arguments,
        )

    @staticmethod
    def _validate_labels_and_arrays(
        labels: pd.DataFrame, data_dir: Path
    ) -> None:
        required_columns = {
            "image_id",
            "patient_id",
            "outflow",
            "middle",
            "apex",
            "outflow_coarse",
            "outflow_fine",
            "middle_coarse",
            "middle_fine",
            "apex_coarse",
            "overall",
            "overall_fine",
        }
        missing_columns = required_columns - set(labels.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required label columns: {sorted(missing_columns)}"
            )
        if labels["image_id"].isna().any() or labels["patient_id"].isna().any():
            raise ValueError("image_id and patient_id must not contain null values.")
        duplicated_ids = labels.loc[
            labels["image_id"].duplicated(), "image_id"
        ].tolist()
        if duplicated_ids:
            raise ValueError(f"Duplicate image_id values: {duplicated_ids[:10]}")

        allowed_values = {
            "outflow": {1, 2, 3},
            "middle": {1, 2, 3},
            "apex": {1, 3},
            "overall": {0, 1},
            "outflow_coarse": {"none", "obstruct"},
            "middle_coarse": {"none", "obstruct"},
            "apex_coarse": {"none", "obstruct"},
            "overall_fine": {"none", "latent", "resting"},
        }
        for column, allowed in allowed_values.items():
            invalid = set(labels[column].dropna().unique()) - allowed
            if invalid or labels[column].isna().any():
                raise ValueError(
                    f"Invalid or null values in {column}: {sorted(invalid)}"
                )

        for region in ("outflow", "middle"):
            expected_coarse = np.where(
                labels[region].isin([1, 2]), "obstruct", "none"
            )
            if not np.array_equal(
                labels[f"{region}_coarse"].to_numpy(), expected_coarse
            ):
                raise ValueError(f"{region}_coarse is inconsistent with {region}.")
        # v2: mid-apical obstructive phenotype 嵌套于 MVO 谱系，
        # 仅当 apex=1 且 middle 梗阻(1/2) 时计为阳性。
        expected_apex_coarse = np.where(
            labels["apex"].eq(1) & labels["middle"].isin([1, 2]),
            "obstruct",
            "none",
        )
        if not np.array_equal(
            labels["apex_coarse"].to_numpy(), expected_apex_coarse
        ):
            raise ValueError(
                "apex_coarse is inconsistent with apex/middle "
                "(v2 MVO-spectrum rule)."
            )
        for region in ("outflow", "middle"):
            obstructed = labels[region].isin([1, 2])
            fine = labels[f"{region}_fine"]
            invalid_fine = (
                (obstructed & (fine != labels[region]))
                | (~obstructed & fine.notna())
            )
            if invalid_fine.any():
                raise ValueError(
                    f"{region}_fine is inconsistent with the hierarchical label."
                )

        # v2: overall 仅由 LVOT(outflow) 与中部(middle) 定义；
        # mid-apical 表型嵌套于 MVO 谱系，不独立计入 overall。
        regional = labels[["outflow", "middle"]]
        expected_overall = regional.isin([1, 2]).any(axis=1).astype(int)
        if not labels["overall"].equals(expected_overall):
            raise ValueError(
                "overall is inconsistent with the LVOT/middle labels (v2 rule)."
            )
        expected_overall_fine = np.select(
            [
                regional.eq(1).any(axis=1),
                regional.eq(2).any(axis=1),
            ],
            ["resting", "latent"],
            default="none",
        )
        if not np.array_equal(
            labels["overall_fine"].to_numpy(), expected_overall_fine
        ):
            raise ValueError(
                "overall_fine is inconsistent with the regional labels."
            )

        missing_arrays = [
            image_id
            for image_id in labels["image_id"]
            if not (data_dir / f"{image_id}.npy").is_file()
        ]
        if missing_arrays:
            preview = ", ".join(missing_arrays[:10])
            raise FileNotFoundError(
                f"{len(missing_arrays)} labeled NPY arrays are missing. "
                f"First missing IDs: {preview}"
            )

    @staticmethod
    def _load_fixed_fold_assignments(
        labels: pd.DataFrame, path: Path, num_folds: int
    ) -> pd.DataFrame:
        """Reuse a frozen patient-level fold split instead of regenerating one.

        Keeps cross-round experiments comparable when label definitions change
        (e.g. the v2 mid-apical/overall redefinition reuses the v1 split).
        """
        if not path.is_file():
            raise FileNotFoundError(f"Fixed fold assignments not found: {path}")
        folds = pd.read_csv(
            path, dtype={"image_id": str, "patient_id": str}
        )
        required = {"image_id", "patient_id", "fold"}
        if not required.issubset(folds.columns):
            raise ValueError(
                f"Fixed fold assignments must contain {sorted(required)}."
            )
        if folds["image_id"].duplicated().any():
            raise ValueError("Fixed fold assignments contain duplicate image_id.")
        label_ids = set(labels["image_id"])
        fold_ids = set(folds["image_id"])
        if label_ids != fold_ids:
            only_labels = sorted(label_ids - fold_ids)[:5]
            only_folds = sorted(fold_ids - label_ids)[:5]
            raise ValueError(
                "Fixed fold assignments do not cover the same examinations: "
                f"only in labels {only_labels}, only in folds {only_folds}."
            )
        per_patient = folds.groupby("patient_id")["fold"].nunique()
        if (per_patient > 1).any():
            raise ValueError(
                "Fixed fold assignments split a patient across folds: "
                f"{sorted(per_patient[per_patient > 1].index.tolist())[:5]}"
            )
        fold_values = set(folds["fold"].astype(int).unique())
        if not fold_values <= set(range(num_folds)):
            raise ValueError(
                f"Fixed fold ids {sorted(fold_values)} exceed num_folds={num_folds}."
            )
        merged = labels.merge(
            folds[["image_id", "fold"]], on="image_id", validate="one_to_one"
        )
        return merged

    @staticmethod
    def _get_class_counts(labels: pd.DataFrame) -> dict[str, list[int]]:
        targets = {
            "outflow_coarse": (labels["outflow_coarse"] == "obstruct").astype(int),
            "middle_coarse": (labels["middle_coarse"] == "obstruct").astype(int),
            "apex_coarse": (labels["apex_coarse"] == "obstruct").astype(int),
            "overall_coarse": labels["overall"].astype(int),
            "outflow_fine": labels["outflow_fine"].dropna().astype(int) - 1,
            "middle_fine": labels["middle_fine"].dropna().astype(int) - 1,
            "overall_fine": labels["overall_fine"].map(OVERALL_FINE_TO_INDEX),
            "outflow_flat": labels["outflow"].astype(str).map(FLAT_LABEL_TO_INDEX),
            "middle_flat": labels["middle"].astype(str).map(FLAT_LABEL_TO_INDEX),
        }
        return {
            name: np.bincount(
                values,
                minlength=3 if name in ("overall_fine", "outflow_flat",
                                        "middle_flat") else 2,
            )
            .astype(int)
            .tolist()
            for name, values in targets.items()
        }

    def save_fold_assignments(self, path: str | Path) -> None:
        if self.fold_labels is None:
            raise RuntimeError("setup() must be called before saving folds.")
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fold_labels[["image_id", "patient_id", "fold"]].to_csv(
            output_path,
            index=False,
        )

    def _loader(
        self,
        dataset: HCMVideoDataset,
        shuffle: bool,
        sampler: Sampler[int] | None = None,
    ) -> DataLoader:
        num_workers = int(self.config["num_workers"])
        return DataLoader(
            dataset,
            batch_size=int(self.config["batch_size"]),
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=bool(self.config["pin_memory"]),
            persistent_workers=bool(
                self.config["persistent_workers"] and num_workers > 0
            ),
            drop_last=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Data module has not been set up.")
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Data module has not been set up.")
        sampler = None
        if self.trainer is not None and self.trainer.world_size > 1:
            sampler = ReplicatedEvalSampler(self.val_dataset)
        return self._loader(self.val_dataset, shuffle=False, sampler=sampler)
