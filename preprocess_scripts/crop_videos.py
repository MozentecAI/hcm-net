"""Content-aware sector cropping of grayscale cine arrays plus normalization stats."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crop NPZ cine arrays to the bounding box of the ultrasound sector "
            "(estimated from a multi-frame max projection), save uncompressed "
            "NPY files, and compute dataset statistics."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("hcmdata_raw"),
        help="Directory containing source <image_id>.npz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hcmdata"),
        help="Directory for cropped <image_id>.npy files.",
    )
    parser.add_argument(
        "--intensity-threshold",
        type=int,
        default=8,
        help=(
            "Pixels above this uint8 intensity count as sector foreground in "
            "the max projection."
        ),
    )
    parser.add_argument(
        "--projection-ratio",
        type=float,
        default=0.02,
        help=(
            "Row/column is kept while its foreground count exceeds this "
            "fraction of the peak count, expanding outward from the peak."
        ),
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.02,
        help="Fractional padding added around the detected bounding box.",
    )
    parser.add_argument(
        "--projection-frames",
        type=int,
        default=8,
        help="Number of evenly spaced frames used for the max projection.",
    )
    parser.add_argument(
        "--npz-key",
        type=str,
        default=None,
        help="Optional source array key. Standard keys are detected automatically.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Number of parallel preprocessing workers.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate existing NPY outputs instead of reusing them.",
    )
    return parser.parse_args()


def select_npz_array(
    archive: np.lib.npyio.NpzFile,
    path: Path,
    npz_key: str | None,
) -> np.ndarray:
    if npz_key is not None:
        if npz_key not in archive:
            raise KeyError(f"NPZ key '{npz_key}' is missing from {path}.")
        return archive[npz_key]
    if "video" in archive:
        return archive["video"]
    if "arr_0" in archive:
        return archive["arr_0"]
    if len(archive.files) == 1:
        return archive[archive.files[0]]
    raise KeyError(f"{path} contains multiple arrays; provide --npz-key.")


def _dominant_span(counts: np.ndarray, ratio: float) -> tuple[int, int]:
    """Return the contiguous above-threshold span containing the peak.

    Expanding from the peak instead of taking the global first/last index
    keeps detached overlays (ECG traces, text) from widening the box.
    """
    length = len(counts)
    peak = int(np.argmax(counts))
    if counts[peak] <= 0:
        return 0, length
    above = counts > counts[peak] * ratio
    start = peak
    while start > 0 and above[start - 1]:
        start -= 1
    stop = peak
    while stop < length - 1 and above[stop + 1]:
        stop += 1
    return start, stop + 1


def content_bounding_box(
    array: np.ndarray,
    intensity_threshold: int,
    projection_ratio: float,
    margin: float,
    projection_frames: int,
) -> tuple[int, int, int, int]:
    """Estimate the sector bounding box from a multi-frame max projection."""
    frame_count, height, width = array.shape
    probe_count = min(projection_frames, frame_count)
    probe_indices = np.linspace(0, frame_count - 1, probe_count).round().astype(int)
    projection = np.asarray(array[probe_indices]).max(axis=0)
    foreground = projection > intensity_threshold

    row_start, row_stop = _dominant_span(foreground.sum(axis=1), projection_ratio)
    col_start, col_stop = _dominant_span(foreground.sum(axis=0), projection_ratio)

    pad_rows = int(round(margin * height))
    pad_cols = int(round(margin * width))
    row_start = max(0, row_start - pad_rows)
    row_stop = min(height, row_stop + pad_rows)
    col_start = max(0, col_start - pad_cols)
    col_stop = min(width, col_stop + pad_cols)

    min_area = 0.25 * height * width
    box_area = (row_stop - row_start) * (col_stop - col_start)
    if box_area < min_area:
        return 0, height, 0, width
    return row_start, row_stop, col_start, col_stop


def uint8_histogram(array: np.ndarray) -> np.ndarray:
    histogram = np.zeros(256, dtype=np.int64)
    for frame in array:
        histogram += np.bincount(frame.reshape(-1), minlength=256)
    return histogram


def process_one(
    arguments: tuple[str, str, dict[str, Any], str | None, bool],
) -> tuple[np.ndarray, dict[str, Any]]:
    input_path_string, output_dir_string, crop_config, npz_key, overwrite = arguments
    input_path = Path(input_path_string)
    output_path = Path(output_dir_string) / f"{input_path.stem}.npy"

    if output_path.exists() and not overwrite:
        cropped = np.load(output_path, mmap_mode="r", allow_pickle=False)
        box = None
    else:
        with np.load(input_path, allow_pickle=False) as archive:
            source = np.asarray(select_npz_array(archive, input_path, npz_key))
        if source.ndim != 3:
            raise ValueError(
                f"Expected [T, H, W] in {input_path}, got {source.shape}."
            )
        if source.dtype != np.uint8:
            raise TypeError(
                f"Expected uint8 in {input_path}, got {source.dtype}."
            )
        row_start, row_stop, col_start, col_stop = content_bounding_box(
            source,
            intensity_threshold=int(crop_config["intensity_threshold"]),
            projection_ratio=float(crop_config["projection_ratio"]),
            margin=float(crop_config["margin"]),
            projection_frames=int(crop_config["projection_frames"]),
        )
        box = (row_start, row_stop, col_start, col_stop)
        cropped = np.ascontiguousarray(
            source[:, row_start:row_stop, col_start:col_stop]
        )
        temporary_path = output_path.with_suffix(".npy.tmp")
        with temporary_path.open("wb") as stream:
            np.save(stream, cropped, allow_pickle=False)
        temporary_path.replace(output_path)

    if cropped.ndim != 3 or cropped.dtype != np.uint8:
        raise ValueError(f"Invalid output array in {output_path}.")
    metadata = {
        "image_id": input_path.stem,
        "frames": int(cropped.shape[0]),
        "height": int(cropped.shape[1]),
        "width": int(cropped.shape[2]),
        "pixels": int(cropped.size),
    }
    if box is not None:
        metadata["crop_box"] = [int(value) for value in box]
    return uint8_histogram(cropped), metadata


def statistics_from_histogram(histogram: np.ndarray) -> dict[str, Any]:
    values = np.arange(256, dtype=np.float64)
    pixel_count = int(histogram.sum())
    mean_uint8 = float(np.dot(histogram, values) / pixel_count)
    variance_uint8 = float(
        np.dot(histogram, (values - mean_uint8) ** 2) / pixel_count
    )
    return {
        "pixel_count": pixel_count,
        "minimum": int(np.flatnonzero(histogram)[0]),
        "maximum": int(np.flatnonzero(histogram)[-1]),
        "mean": mean_uint8 / 255.0,
        "std": variance_uint8**0.5 / 255.0,
    }


def main() -> None:
    args = parse_args()
    if not 0 <= args.intensity_threshold < 255:
        raise ValueError("--intensity-threshold must be in [0, 255).")
    if not 0.0 < args.projection_ratio < 1.0:
        raise ValueError("--projection-ratio must be in the interval (0, 1).")
    if not 0.0 <= args.margin < 0.5:
        raise ValueError("--margin must be in the interval [0, 0.5).")
    if args.projection_frames < 1:
        raise ValueError("--projection-frames must be at least 1.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")

    input_files = sorted(args.input_dir.glob("*.npz"))
    if not input_files:
        raise FileNotFoundError(f"No NPZ files found in {args.input_dir}.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    crop_config = {
        "intensity_threshold": args.intensity_threshold,
        "projection_ratio": args.projection_ratio,
        "margin": args.margin,
        "projection_frames": args.projection_frames,
    }
    jobs = [
        (
            str(input_path),
            str(args.output_dir),
            crop_config,
            args.npz_key,
            args.overwrite,
        )
        for input_path in input_files
    ]
    histogram = np.zeros(256, dtype=np.int64)
    sample_metadata: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, (sample_histogram, metadata) in enumerate(
            executor.map(process_one, jobs, chunksize=1),
            start=1,
        ):
            histogram += sample_histogram
            sample_metadata.append(metadata)
            if index % 25 == 0 or index == len(jobs):
                print(f"Processed {index}/{len(jobs)} samples", flush=True)

    statistics = statistics_from_histogram(histogram)
    report = {
        "source_directory": str(args.input_dir),
        "output_directory": str(args.output_dir),
        "sample_count": len(sample_metadata),
        "crop_method": "sector_max_projection",
        "crop_config": crop_config,
        "normalization_scale": 255.0,
        **statistics,
    }
    report_path = args.output_dir / "normalization_stats.json"
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")

    boxes_path = args.output_dir / "crop_boxes.json"
    with boxes_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                metadata["image_id"]: metadata.get("crop_box")
                for metadata in sample_metadata
            },
            stream,
            indent=2,
        )
        stream.write("\n")

    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved statistics to {report_path}", flush=True)
    print(f"Saved per-sample crop boxes to {boxes_path}", flush=True)


if __name__ == "__main__":
    main()
