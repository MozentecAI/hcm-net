"""Memory-mapped NPY grayscale echocardiography video dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


OVERALL_FINE_TO_INDEX = {"none": 0, "latent": 1, "resting": 2}
# Raw regional label -> flat 3-class index (none/latent/resting) used by the
# no-hierarchy ablation heads.
FLAT_LABEL_TO_INDEX = {"3": 0, "2": 1, "1": 2}


class HCMVideoDataset(Dataset[dict[str, Any]]):
    """Load one apical-view cine loop and all hierarchical targets per sample."""

    def __init__(
        self,
        labels: pd.DataFrame,
        data_dir: str | Path,
        num_frames: int,
        image_size: int,
        training: bool,
        normalization_mean: float = 0.136364,
        normalization_std: float = 0.219420,
        temporal_window: tuple[float, float] = (0.75, 1.0),
        rotation_degrees: float = 7.0,
        brightness: float = 0.15,
        contrast: float = 0.15,
        phase_bias_strength: float = 0.0,
        phase_candidate_multiplier: int = 2,
        eval_num_clips: int = 1,
        eval_window_ratio: float = 0.75,
        resize_mode: str = "stretch",
        sampling_mode: str = "window",
        cycle_num_cycles: int = 2,
        cycle_period_range: tuple[int, int] = (16, 96),
        cycle_curve_stride: int = 2,
        eval_phase_bias_strength: float = 0.0,
        temporal_stride_jitter: float = 0.0,
        frame_dropout_prob: float = 0.0,
    ) -> None:
        self.labels = labels.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.image_size = image_size
        self.training = training
        self.normalization_mean = [normalization_mean]
        self.normalization_std = [normalization_std]
        self.temporal_window = temporal_window
        self.rotation_degrees = rotation_degrees
        self.brightness = brightness
        self.contrast = contrast
        self.phase_bias_strength = float(phase_bias_strength)
        self.phase_candidate_multiplier = max(1, int(phase_candidate_multiplier))
        self.eval_num_clips = max(1, int(eval_num_clips))
        self.eval_window_ratio = float(eval_window_ratio)
        if resize_mode not in {"stretch", "letterbox"}:
            raise ValueError("resize_mode must be stretch or letterbox.")
        if sampling_mode not in {"window", "cycle"}:
            raise ValueError("sampling_mode must be window or cycle.")
        self.resize_mode = resize_mode
        self.sampling_mode = sampling_mode
        self.cycle_num_cycles = max(1, int(cycle_num_cycles))
        min_period, max_period = (
            int(cycle_period_range[0]),
            int(cycle_period_range[1]),
        )
        if not 2 <= min_period < max_period:
            raise ValueError("cycle_period_range must satisfy 2 <= min < max.")
        self.cycle_period_range = (min_period, max_period)
        self.cycle_curve_stride = max(1, int(cycle_curve_stride))
        self.eval_phase_bias_strength = float(eval_phase_bias_strength)
        self.temporal_stride_jitter = float(temporal_stride_jitter)
        self.frame_dropout_prob = float(frame_dropout_prob)

    def __len__(self) -> int:
        return len(self.labels)

    def _load_array(self, path: Path) -> np.ndarray:
        array = np.load(path, mmap_mode="r", allow_pickle=False)

        if array.ndim != 3:
            raise ValueError(
                f"Expected grayscale [T, H, W] in {path}, "
                f"got shape {array.shape}."
            )
        if array.shape[0] == 0:
            raise ValueError(f"No frames found in {path}.")
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"Expected a numeric array in {path}, got {array.dtype}.")
        return array

    def _phase_cdf(
        self,
        array: np.ndarray,
        start: int,
        stop: int,
        strength: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Candidate frames plus a brightness-weighted sampling CDF.

        Blood pools are dark and myocardium is bright, so the mean intensity
        of a frame peaks around end-systole. Mixing a uniform floor with the
        brightness weights keeps full temporal coverage while oversampling
        high-intensity phases.
        """
        candidate_count = self.num_frames * self.phase_candidate_multiplier
        candidates = np.linspace(start, stop, candidate_count).round().astype(np.int64)
        brightness = array[candidates].mean(axis=(1, 2), dtype=np.float64)
        kernel = np.ones(5, dtype=np.float64) / 5.0
        smoothed = np.convolve(brightness, kernel, mode="same")
        weights = smoothed - smoothed.min()
        total = weights.sum()
        if total <= 0:
            return None
        weights /= total
        weights = (1.0 - strength) / len(weights) + strength * weights
        cdf = np.cumsum(weights)
        cdf /= cdf[-1]
        return candidates, cdf

    def _phase_weighted_indices(
        self,
        array: np.ndarray,
        start: int,
        stop: int,
        strength: float,
        jitter: bool,
    ) -> np.ndarray:
        """Stratified inverse-CDF draw; deterministic at evaluation."""
        weighted = self._phase_cdf(array, start, stop, strength)
        if weighted is None:
            return np.linspace(start, stop, self.num_frames).round().astype(np.int64)
        candidates, cdf = weighted
        if jitter:
            offsets = torch.rand(self.num_frames).numpy()
        else:
            offsets = np.full(self.num_frames, 0.5)
        positions = (np.arange(self.num_frames) + offsets) / self.num_frames
        selected = np.searchsorted(cdf, positions).clip(0, len(candidates) - 1)
        return np.sort(candidates[selected])

    def _apply_temporal_augmentation(
        self,
        indices: np.ndarray,
        start: int,
        stop: int,
    ) -> np.ndarray:
        """Randomize inter-frame stride and drop frames, keeping order."""
        indices = indices.copy()
        if self.temporal_stride_jitter > 0.0 and len(indices) > 2:
            step = max((stop - start) / max(len(indices) - 1, 1), 1.0)
            offsets = (
                torch.rand(len(indices)).numpy() * 2.0 - 1.0
            ) * self.temporal_stride_jitter * step
            offsets[0] = 0.0
            offsets[-1] = 0.0
            jittered = np.clip(
                indices + np.round(offsets), start, stop
            ).astype(np.int64)
            indices = np.maximum.accumulate(jittered)
        if self.frame_dropout_prob > 0.0 and len(indices) > 1:
            keep = torch.rand(len(indices)).numpy() >= self.frame_dropout_prob
            keep[0] = True
            for position in range(1, len(indices)):
                if not keep[position]:
                    indices[position] = indices[position - 1]
        return indices

    def _cycle_profile(self, array: np.ndarray) -> tuple[np.ndarray, int]:
        """Smoothed frame-brightness curve and its temporal stride."""
        stride = self.cycle_curve_stride
        grid = np.arange(0, array.shape[0], stride)
        profile = array[grid][:, ::8, ::8].mean(axis=(1, 2), dtype=np.float64)
        kernel = np.ones(5, dtype=np.float64) / 5.0
        return np.convolve(profile, kernel, mode="same"), stride

    def _detect_cycle(
        self,
        profile: np.ndarray,
        stride: int,
        frame_count: int,
    ) -> tuple[int, int] | None:
        """Estimate the dominant cardiac period and a diastolic anchor.

        The period comes from the strongest autocorrelation lag inside the
        plausible range; the anchor is the darkest (end-diastolic) frame of
        the first detected cycle, giving a deterministic phase reference.
        """
        min_period, max_period = self.cycle_period_range
        lo = max(1, min_period // stride)
        hi = min(max_period // stride, len(profile) - 2)
        if hi <= lo:
            return None
        centered = profile - profile.mean()
        energy = float(np.dot(centered, centered))
        if energy <= 0:
            return None
        autocorr = np.correlate(centered, centered, mode="full")[
            len(centered) - 1:
        ]
        lag = lo + int(np.argmax(autocorr[lo:hi + 1]))
        if autocorr[lag] <= 0.3 * energy:
            return None
        period = lag * stride
        if not min_period <= period <= max_period or period >= frame_count:
            return None
        anchor = int(np.argmin(profile[:lag])) * stride
        return period, anchor

    def _cycle_clip(self, start: int, span: int) -> np.ndarray:
        """Phase-grid indices covering a fixed number of cardiac cycles."""
        cycles = self.cycle_num_cycles
        base_count = self.num_frames // cycles
        extra = self.num_frames % cycles
        segment = span / cycles
        parts = []
        for cycle in range(cycles):
            count = base_count + (extra if cycle == cycles - 1 else 0)
            segment_start = start + round(cycle * segment)
            segment_stop = start + round((cycle + 1) * segment) - 1
            parts.append(
                np.linspace(segment_start, segment_stop, count)
                .round()
                .astype(np.int64)
            )
        return np.concatenate(parts)

    def _cycle_indices(self, array: np.ndarray) -> np.ndarray | None:
        """Phase-aligned clips; None when no reliable cycle is detected."""
        frame_count = array.shape[0]
        profile, stride = self._cycle_profile(array)
        detected = self._detect_cycle(profile, stride, frame_count)
        if detected is None:
            return None
        period, anchor = detected
        span = min(self.cycle_num_cycles * period, frame_count)
        max_start = frame_count - span

        if self.training:
            start = (
                int(torch.randint(max_start + 1, (1,)).item())
                if max_start > 0
                else 0
            )
            return self._cycle_clip(start, span)[np.newaxis, :]

        if self.eval_num_clips == 1:
            starts = [min(max(anchor, 0), max_start)]
        elif max_start > 0:
            starts = (
                np.linspace(0, max_start, self.eval_num_clips)
                .round()
                .astype(np.int64)
                .tolist()
            )
        else:
            # Only one position fits; repeat it so every sample in a batch
            # keeps the same clip dimension.
            starts = [0] * self.eval_num_clips
        return np.stack(
            [self._cycle_clip(start, span) for start in starts], axis=0
        )

    def _sample_indices(self, array: np.ndarray) -> np.ndarray:
        """Return frame indices with shape [num_clips, num_frames]."""
        frame_count = array.shape[0]
        if self.sampling_mode == "cycle":
            cycle_indices = self._cycle_indices(array)
            if cycle_indices is not None:
                return cycle_indices

        if self.training:
            minimum, maximum = self.temporal_window
            ratio = float(torch.empty(1).uniform_(minimum, maximum).item())
            window_size = max(1, round(frame_count * ratio))
            max_start = frame_count - window_size
            start = (
                int(torch.randint(max_start + 1, (1,)).item())
                if max_start > 0
                else 0
            )
            stop = start + window_size - 1
            if self.phase_bias_strength > 0.0:
                indices = self._phase_weighted_indices(
                    array, start, stop, self.phase_bias_strength, jitter=True
                )
            else:
                indices = np.linspace(start, stop, self.num_frames).round().astype(
                    np.int64
                )
            indices = self._apply_temporal_augmentation(indices, start, stop)
            return indices[np.newaxis, :]

        if self.eval_num_clips == 1:
            if self.eval_phase_bias_strength > 0.0:
                indices = self._phase_weighted_indices(
                    array,
                    0,
                    frame_count - 1,
                    self.eval_phase_bias_strength,
                    jitter=False,
                )
            else:
                indices = np.linspace(0, frame_count - 1, self.num_frames).round().astype(
                    np.int64
                )
            return indices[np.newaxis, :]

        window_size = max(1, round(frame_count * self.eval_window_ratio))
        max_start = frame_count - window_size
        starts = np.linspace(0, max_start, self.eval_num_clips).round().astype(np.int64)
        clips = []
        for start in starts:
            stop = int(start) + window_size - 1
            if self.eval_phase_bias_strength > 0.0:
                clip = self._phase_weighted_indices(
                    array,
                    int(start),
                    stop,
                    self.eval_phase_bias_strength,
                    jitter=False,
                )
            else:
                clip = np.linspace(start, stop, self.num_frames).round().astype(
                    np.int64
                )
            clips.append(clip)
        return np.stack(clips, axis=0)

    def _resize(self, video: Tensor) -> Tensor:
        """Resize to the configured input size, preserving aspect on request."""
        if video.shape[-2:] == (self.image_size, self.image_size):
            return video
        if self.resize_mode == "letterbox":
            height, width = video.shape[-2:]
            scale = self.image_size / max(height, width)
            new_height = max(1, round(height * scale))
            new_width = max(1, round(width * scale))
            video = TF.resize(
                video,
                [new_height, new_width],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            pad_height = self.image_size - new_height
            pad_width = self.image_size - new_width
            return TF.pad(
                video,
                [
                    pad_width // 2,
                    pad_height // 2,
                    pad_width - pad_width // 2,
                    pad_height - pad_height // 2,
                ],
                fill=0.0,
            )
        return TF.resize(
            video,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

    def _transform(self, frames: np.ndarray) -> Tensor:
        source_dtype = frames.dtype
        video = torch.from_numpy(np.ascontiguousarray(frames)).float()
        if np.issubdtype(source_dtype, np.integer):
            video.div_(float(np.iinfo(source_dtype).max))
        elif video.numel() > 0 and video.max() > 1.0:
            video.div_(255.0)
        if not torch.isfinite(video).all():
            raise ValueError("The input array contains NaN or infinite values.")
        if video.min() < 0.0 or video.max() > 1.0:
            raise ValueError("Expected grayscale values in the [0, 1] range.")
        video = self._resize(video)
        if self.training:
            angle = float(
                torch.empty(1)
                .uniform_(-self.rotation_degrees, self.rotation_degrees)
                .item()
            )
            brightness_factor = float(
                torch.empty(1)
                .uniform_(1.0 - self.brightness, 1.0 + self.brightness)
                .item()
            )
            contrast_factor = float(
                torch.empty(1)
                .uniform_(1.0 - self.contrast, 1.0 + self.contrast)
                .item()
            )
            video = TF.rotate(
                video,
                angle,
                interpolation=InterpolationMode.BILINEAR,
            )
            video = TF.adjust_brightness(video, brightness_factor)
            video = TF.adjust_contrast(video, contrast_factor)
        return TF.normalize(
            video,
            self.normalization_mean,
            self.normalization_std,
        )

    @staticmethod
    def _fine_target(value: Any) -> int:
        if pd.isna(value):
            return -1
        return int(value) - 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.labels.iloc[index]
        image_id = str(row["image_id"])
        array_path = self.data_dir / f"{image_id}.npy"
        array = self._load_array(array_path)
        clip_indices = self._sample_indices(array)
        num_clips, frames_per_clip = clip_indices.shape
        frames = array[clip_indices.reshape(-1)][:, np.newaxis, :, :]
        transformed = self._transform(frames)
        video = transformed.reshape(
            num_clips, frames_per_clip, *transformed.shape[1:]
        )
        if num_clips == 1:
            video = video.squeeze(0)

        overall_target = torch.tensor(int(row["overall"]), dtype=torch.long)
        targets = {
            "outflow_coarse": torch.tensor(
                int(row["outflow_coarse"] == "obstruct"), dtype=torch.long
            ),
            "outflow_fine": torch.tensor(
                self._fine_target(row["outflow_fine"]), dtype=torch.long
            ),
            # Flat 3-class targets (none=0, latent=1, resting=2) feed the
            # no-hierarchy ablation heads; unused by the hierarchical model.
            "outflow_flat": torch.tensor(
                FLAT_LABEL_TO_INDEX[str(row["outflow"])], dtype=torch.long
            ),
            "middle_coarse": torch.tensor(
                int(row["middle_coarse"] == "obstruct"), dtype=torch.long
            ),
            "middle_fine": torch.tensor(
                self._fine_target(row["middle_fine"]), dtype=torch.long
            ),
            "middle_flat": torch.tensor(
                FLAT_LABEL_TO_INDEX[str(row["middle"])], dtype=torch.long
            ),
            "apex_coarse": torch.tensor(
                int(row["apex_coarse"] == "obstruct"), dtype=torch.long
            ),
            "overall_coarse": overall_target,
            "overall_derived": overall_target.clone(),
            "overall_fine": torch.tensor(
                OVERALL_FINE_TO_INDEX[str(row["overall_fine"])],
                dtype=torch.long,
            ),
        }
        return {
            "video": video,
            "targets": targets,
            "image_id": image_id,
            "patient_id": str(row["patient_id"]),
        }
