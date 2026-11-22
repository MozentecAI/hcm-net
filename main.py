"""Training entry point for HCM echocardiography video classification."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from hcmnet.data import HCMDataModule
from hcmnet.ema import EmaCallback
from hcmnet.lightning_module import HCMLitModule


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the hierarchical HCM video classifier."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "train.yaml",
        help="Path to a YAML training configuration.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Override the validation fold from the configuration.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help='Override devices, for example "1", "0,1", or "auto".',
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume training from a Lightning checkpoint.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    for key in ("labels_path", "data_dir"):
        value = Path(config["data"][key])
        if not value.is_absolute():
            config["data"][key] = str(PROJECT_ROOT / value)
    fixed_folds = config["data"].get("fixed_fold_assignments")
    if fixed_folds:
        value = Path(fixed_folds)
        if not value.is_absolute():
            config["data"]["fixed_fold_assignments"] = str(PROJECT_ROOT / value)
    output_dir = Path(config["trainer"]["output_dir"])
    if not output_dir.is_absolute():
        config["trainer"]["output_dir"] = str(PROJECT_ROOT / output_dir)
    return config


def parse_devices(value: str | int | list[int]) -> str | int | list[int]:
    if not isinstance(value, str):
        return value
    if value == "auto":
        return value
    if "," in value:
        return [int(device.strip()) for device in value.split(",")]
    return int(value)


@rank_zero_only
def save_run_metadata(
    data_module: HCMDataModule,
    logger: TensorBoardLogger,
    config: dict[str, Any],
) -> None:
    """Persist split assignments and hyperparameters once under DDP."""
    data_module.save_fold_assignments(Path(logger.log_dir) / "fold_assignments.csv")
    logger.log_hyperparams(config)


def create_logger(
    config: dict[str, Any],
    experiment_name: str,
    fold: int,
    resume_path: Path | None,
) -> TensorBoardLogger:
    """Create a new run logger or reuse the original resumed run directory."""
    if resume_path is None:
        return TensorBoardLogger(
            save_dir=str(config["trainer"]["output_dir"]),
            name=f"{experiment_name}/fold_{fold}",
        )

    checkpoint_path = resume_path.resolve()
    if checkpoint_path.parent.name != "checkpoints":
        raise ValueError(
            "A resumed checkpoint must be inside its run's checkpoints directory."
        )
    run_directory = checkpoint_path.parent.parent
    return TensorBoardLogger(
        save_dir=str(run_directory.parent),
        name="",
        version=run_directory.name,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.fold is not None:
        config["data"]["fold"] = args.fold
    if args.devices is not None:
        config["trainer"]["devices"] = args.devices

    pl.seed_everything(int(config["seed"]), workers=True)
    data_module = HCMDataModule(config["data"], seed=int(config["seed"]))
    data_module.setup("fit")

    fold = int(config["data"]["fold"])
    experiment_name = str(config["trainer"]["experiment_name"])
    logger = create_logger(
        config,
        experiment_name,
        fold,
        args.resume,
    )
    save_run_metadata(data_module, logger, config)

    module = HCMLitModule(
        model_config=config["model"],
        loss_config=config["loss"],
        optimizer_config=config["optimizer"],
        class_counts=data_module.class_counts,
    )
    monitor = str(config["trainer"].get("monitor", "val/mean_coarse_auroc"))
    checkpoint = ModelCheckpoint(
        dirpath=Path(logger.log_dir) / "checkpoints",
        filename="epoch={epoch:02d}-step={step}",
        monitor=monitor,
        mode="max",
        save_top_k=8,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks: list[pl.Callback] = [
        checkpoint,
        EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=int(config["trainer"]["early_stopping_patience"]),
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    ema_decay = float(config["trainer"].get("ema_decay", 0.0))
    if ema_decay > 0.0:
        callbacks.insert(0, EmaCallback(decay=ema_decay))
    trainer = pl.Trainer(
        max_epochs=int(config["trainer"]["max_epochs"]),
        accelerator=config["trainer"]["accelerator"],
        devices=parse_devices(config["trainer"]["devices"]),
        strategy=config["trainer"]["strategy"],
        precision=config["trainer"]["precision"],
        accumulate_grad_batches=int(
            config["trainer"]["accumulate_grad_batches"]
        ),
        gradient_clip_val=float(config["trainer"]["gradient_clip_val"]),
        deterministic=bool(config["trainer"]["deterministic"]),
        benchmark=bool(config["trainer"]["benchmark"]),
        log_every_n_steps=int(config["trainer"]["log_every_n_steps"]),
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(
        module,
        datamodule=data_module,
        ckpt_path=str(args.resume) if args.resume else None,
    )


if __name__ == "__main__":
    main()
