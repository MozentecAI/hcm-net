"""Run one fold's held-out validation and export sample-level predictions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pytorch_lightning as pl

from hcmnet.calibration import (
    apply_thresholds_to_frame,
    fit_calibration,
    thresholded_metrics_from_frame,
)
from hcmnet.data import HCMDataModule
from hcmnet.lightning_module import HCMLitModule
from hcmnet.prediction import prediction_rows
from main import PROJECT_ROOT, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate one held-out fold from a Lightning checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "train.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Prediction CSV path. Defaults to validation_predictions_fold_<k>.csv "
            "beside the checkpoint directory."
        ),
    )
    parser.add_argument(
        "--accelerator",
        choices=("auto", "cpu", "gpu"),
        default=None,
        help="Override the configured accelerator. Validation always uses one device.",
    )
    parser.add_argument(
        "--num-clips",
        type=int,
        default=5,
        help=(
            "Number of temporal clips per video for multi-clip evaluation. "
            "Defaults to 5; pass 1 for a single deterministic full-video pass."
        ),
    )
    parser.add_argument(
        "--calibration-criterion",
        choices=("youden", "f1"),
        default="youden",
        help="Criterion used to pick per-task decision thresholds.",
    )
    parser.add_argument(
        "--no-temperature",
        action="store_true",
        help="Skip temperature fitting and calibrate thresholds only.",
    )
    parser.add_argument(
        "--calibration-output",
        type=Path,
        default=None,
        help=(
            "Calibration JSON path. Defaults to calibration_fold_<k>.json "
            "beside the checkpoint directory."
        ),
    )
    return parser.parse_args()


def _run_directory(checkpoint: Path) -> Path | None:
    if checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent
    return None


def _validate_checkpoint_fold(checkpoint: Path, fold: int) -> None:
    run_directory = _run_directory(checkpoint)
    if run_directory is None:
        return
    fold_directory = run_directory.parent.name
    match = re.fullmatch(r"fold_(\d+)", fold_directory)
    if match and int(match.group(1)) != fold:
        raise ValueError(
            f"Checkpoint is under {fold_directory}, but --fold {fold} was requested."
        )


def _validate_saved_assignments(
    checkpoint: Path,
    generated: pd.DataFrame,
) -> None:
    run_directory = _run_directory(checkpoint)
    if run_directory is None:
        return
    assignment_path = run_directory / "fold_assignments.csv"
    if not assignment_path.is_file():
        return

    saved = pd.read_csv(
        assignment_path,
        dtype={"image_id": str, "patient_id": str},
    )
    required = {"image_id", "patient_id", "fold"}
    if not required.issubset(saved.columns):
        raise ValueError(f"{assignment_path} is missing columns: {sorted(required)}")
    current = generated[["image_id", "patient_id", "fold"]].copy()
    saved = saved[["image_id", "patient_id", "fold"]].copy()
    for frame in (current, saved):
        frame["fold"] = frame["fold"].astype(int)
        frame.sort_values("image_id", inplace=True, ignore_index=True)
    if not current.equals(saved):
        raise ValueError(
            "The configured labels/seed produce fold assignments different from "
            f"the training run: {assignment_path}"
        )


def _merge_with_labels(
    predictions: pd.DataFrame,
    fold_labels: pd.DataFrame,
    fold: int,
) -> pd.DataFrame:
    if predictions["image_id"].duplicated().any():
        duplicates = predictions.loc[
            predictions["image_id"].duplicated(), "image_id"
        ].tolist()
        raise ValueError(f"Duplicate predictions for image_id: {duplicates[:10]}")

    validation_labels = fold_labels.loc[fold_labels["fold"] == fold].copy()
    expected_ids = set(validation_labels["image_id"].astype(str))
    predicted_ids = set(predictions["image_id"])
    if expected_ids != predicted_ids:
        missing = sorted(expected_ids - predicted_ids)
        unexpected = sorted(predicted_ids - expected_ids)
        raise ValueError(
            "Prediction IDs do not match the validation fold. "
            f"Missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    merged = validation_labels.merge(
        predictions,
        on="image_id",
        how="left",
        validate="one_to_one",
    )
    patient_mismatch = (
        merged["patient_id"].astype(str) != merged["_predicted_patient_id"]
    )
    if patient_mismatch.any():
        raise ValueError("Predicted patient_id values do not match the label CSV.")
    merged.drop(columns="_predicted_patient_id", inplace=True)
    merged.rename(columns={"fold": "validation_fold"}, inplace=True)
    merged.insert(2, "split", "validation")
    return merged


def _print_metrics(metrics: dict[str, Any]) -> None:
    print("\nValidation metrics", flush=True)
    for name, value in sorted(metrics.items()):
        if isinstance(value, dict):
            for sub_name, sub_value in sorted(value.items()):
                print(f"{name}/{sub_name}: {sub_value:.6f}", flush=True)
        else:
            print(f"{name}: {value:.6f}", flush=True)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    _validate_checkpoint_fold(checkpoint, args.fold)

    config = load_config(args.config.expanduser().resolve())
    num_folds = int(config["data"]["num_folds"])
    if not 0 <= args.fold < num_folds:
        raise ValueError(f"--fold must be between 0 and {num_folds - 1}.")
    config["data"]["fold"] = args.fold
    if args.num_clips is not None:
        if args.num_clips < 1:
            raise ValueError("--num-clips must be at least 1.")
        config["data"]["eval_num_clips"] = args.num_clips

    pl.seed_everything(int(config["seed"]), workers=True)
    data_module = HCMDataModule(config["data"], seed=int(config["seed"]))
    data_module.setup("fit")
    if data_module.fold_labels is None:
        raise RuntimeError("Data module did not create fold assignments.")
    _validate_saved_assignments(checkpoint, data_module.fold_labels)

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
    batches = trainer.predict(module, dataloaders=data_module.val_dataloader())
    if not batches:
        raise RuntimeError("Validation produced no prediction batches.")

    calibration = fit_calibration(
        batches,
        criterion=args.calibration_criterion,
        fit_temperatures=not args.no_temperature,
    )
    calibration["fold"] = args.fold
    calibration["checkpoint_path"] = str(checkpoint)

    predictions, metrics = prediction_rows(batches)
    output_frame = _merge_with_labels(
        predictions,
        data_module.fold_labels,
        args.fold,
    )
    output_frame = apply_thresholds_to_frame(output_frame, calibration)
    metrics["calibrated_in_sample"] = thresholded_metrics_from_frame(
        output_frame, calibration
    )

    run_directory = _run_directory(checkpoint) or checkpoint.parent
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else run_directory / f"validation_predictions_fold_{args.fold}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(output_path, index=False)

    calibration_path = (
        args.calibration_output.expanduser().resolve()
        if args.calibration_output is not None
        else run_directory / f"calibration_fold_{args.fold}.json"
    )
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    with calibration_path.open("w", encoding="utf-8") as stream:
        json.dump(calibration, stream, indent=2, sort_keys=True)
        stream.write("\n")

    metrics_path = output_path.with_name(f"{output_path.stem}_metrics.json")
    with metrics_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "fold": args.fold,
                "checkpoint_path": str(checkpoint),
                "calibration_path": str(calibration_path),
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
    print(f"Calibration: {calibration_path}", flush=True)
    print(f"Metrics: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
