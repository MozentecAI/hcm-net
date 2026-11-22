"""Run an independent external test set from a trained HCM-Net checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from hcmnet.calibration import (
    apply_calibration_to_batches,
    apply_thresholds_to_frame,
    thresholded_metrics_from_frame,
)
from hcmnet.data.datamodule import HCMDataModule
from hcmnet.data.dataset import HCMVideoDataset
from hcmnet.lightning_module import HCMLitModule
from hcmnet.prediction import prediction_rows
from main import PROJECT_ROOT, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on an independent external dataset."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "train.yaml",
        help="Training configuration used for input shape and normalization.",
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV path. Defaults to "
            "independent_test_external_predictions.csv in the checkpoint "
            "run directory."
        ),
    )
    parser.add_argument(
        "--accelerator",
        choices=("auto", "cpu", "gpu"),
        default="gpu",
        help="Override the configured accelerator. Testing always uses one device.",
    )
    parser.add_argument(
        "--num-clips",
        type=int,
        default=5,
        help="Number of temporal clips per video for multi-clip evaluation.",
    )
    parser.add_argument(
        "--clip-window-ratio",
        type=float,
        default=0.75,
        help="Temporal window covered by each clip when --num-clips > 1.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help=(
            "Calibration JSON produced by validate_fold.py. Applies temperature "
            "scaling and adds threshold-calibrated prediction columns."
        ),
    )
    return parser.parse_args()


def _run_directory(checkpoint: Path) -> Path:
    if checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent
    return checkpoint.parent


def _build_test_loader(
    labels: pd.DataFrame,
    data_dir: Path,
    data_config: dict,
    num_clips: int,
    clip_window_ratio: float,
) -> DataLoader:
    HCMDataModule._validate_labels_and_arrays(labels, data_dir)
    dataset = HCMVideoDataset(
        labels=labels,
        data_dir=data_dir,
        num_frames=int(data_config["num_frames"]),
        image_size=int(data_config["image_size"]),
        training=False,
        normalization_mean=float(data_config["normalization"]["mean"]),
        normalization_std=float(data_config["normalization"]["std"]),
        resize_mode=str(data_config.get("resize_mode", "stretch")),
        sampling_mode=str(data_config.get("sampling_mode", "window")),
        cycle_num_cycles=int(data_config.get("cycle_num_cycles", 2)),
        cycle_period_range=tuple(data_config.get("cycle_period_range", (16, 96))),
        cycle_curve_stride=int(data_config.get("cycle_curve_stride", 2)),
        eval_phase_bias_strength=float(
            data_config.get("eval_phase_bias_strength", 0.0)
        ),
        eval_num_clips=num_clips,
        eval_window_ratio=clip_window_ratio,
    )
    num_workers = int(data_config["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(data_config["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(data_config["pin_memory"]),
        persistent_workers=bool(
            data_config["persistent_workers"] and num_workers > 0
        ),
        drop_last=False,
    )


def _merge_with_test_labels(
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    if predictions["image_id"].duplicated().any():
        duplicates = predictions.loc[
            predictions["image_id"].duplicated(), "image_id"
        ].tolist()
        raise ValueError(f"Duplicate predictions for image_id: {duplicates[:10]}")

    expected_ids = set(labels["image_id"].astype(str))
    predicted_ids = set(predictions["image_id"])
    if expected_ids != predicted_ids:
        missing = sorted(expected_ids - predicted_ids)
        unexpected = sorted(predicted_ids - expected_ids)
        raise ValueError(
            "Prediction IDs do not match the independent test labels. "
            f"Missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    merged = labels.merge(
        predictions,
        on="image_id",
        how="left",
        validate="one_to_one",
    )
    mismatch = (
        merged["patient_id"].astype(str) != merged["_predicted_patient_id"]
    )
    if mismatch.any():
        raise ValueError("Predicted patient_id values do not match the label CSV.")
    merged.drop(columns="_predicted_patient_id", inplace=True)
    merged.insert(2, "split", "independent_test")
    return merged


def _print_metrics(metrics: dict[str, Any]) -> None:
    print("\nIndependent test metrics", flush=True)
    for name, value in sorted(metrics.items()):
        if isinstance(value, dict):
            for sub_name, sub_value in sorted(value.items()):
                print(f"{name}/{sub_name}: {sub_value:.6f}", flush=True)
        else:
            print(f"{name}: {value:.6f}", flush=True)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    labels_path = args.labels.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if not labels_path.is_file():
        raise FileNotFoundError(f"Label CSV does not exist: {labels_path}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Test data directory does not exist: {data_dir}")

    calibration = None
    if args.calibration is not None:
        calibration_path = args.calibration.expanduser().resolve()
        if not calibration_path.is_file():
            raise FileNotFoundError(
                f"Calibration JSON does not exist: {calibration_path}"
            )
        with calibration_path.open("r", encoding="utf-8") as stream:
            calibration = json.load(stream)
    if args.num_clips < 1:
        raise ValueError("--num-clips must be at least 1.")
    if not 0.0 < args.clip_window_ratio <= 1.0:
        raise ValueError("--clip-window-ratio must be in the interval (0, 1].")

    config = load_config(args.config.expanduser().resolve())
    pl.seed_everything(int(config["seed"]), workers=True)
    labels = pd.read_csv(
        labels_path,
        dtype={"image_id": str, "patient_id": str},
    )
    loader = _build_test_loader(
        labels,
        data_dir,
        config["data"],
        num_clips=args.num_clips,
        clip_window_ratio=args.clip_window_ratio,
    )

    module = HCMLitModule.load_from_checkpoint(
        str(checkpoint),
        map_location="cpu",
    )
    accelerator = args.accelerator or str(config["trainer"]["accelerator"])
    precision = config["trainer"]["precision"] if accelerator != "cpu" else "32-true"
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        precision=precision,
        deterministic=bool(config["trainer"]["deterministic"]),
        benchmark=bool(config["trainer"]["benchmark"]),
        logger=False,
        enable_checkpointing=False,
    )
    batches = trainer.predict(module, dataloaders=loader)
    if not batches:
        raise RuntimeError("Independent testing produced no prediction batches.")

    if calibration is not None:
        batches = apply_calibration_to_batches(batches, calibration)

    predictions, metrics = prediction_rows(batches)
    output_frame = _merge_with_test_labels(labels, predictions)
    if calibration is not None:
        output_frame = apply_thresholds_to_frame(output_frame, calibration)
        metrics["calibrated"] = thresholded_metrics_from_frame(
            output_frame, calibration
        )
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else _run_directory(checkpoint)
        / "independent_test_external_predictions.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(output_path, index=False)

    metrics_path = output_path.with_name(f"{output_path.stem}_metrics.json")
    with metrics_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "split": "independent_test",
                "dataset": "external",
                "checkpoint_path": str(checkpoint),
                "labels_path": str(labels_path),
                "data_dir": str(data_dir),
                "num_clips": args.num_clips,
                "clip_window_ratio": args.clip_window_ratio,
                "calibration_path": (
                    str(args.calibration.expanduser().resolve())
                    if args.calibration is not None
                    else None
                ),
                "num_samples": len(output_frame),
                "num_patients": int(output_frame["patient_id"].nunique()),
                "metrics": metrics,
            },
            stream,
            indent=2,
            sort_keys=True,
        )

    _print_metrics(metrics)
    print(f"\nPredictions: {output_path}", flush=True)
    print(f"Metrics: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
