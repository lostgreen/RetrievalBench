"""Utility helpers for the video data processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import cv2
import numpy as np


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_frame_as_image(frame: np.ndarray, path: Path) -> Path:
    """Persist a single frame to disk as a JPEG image."""
    ensure_dir(path.parent)
    if not cv2.imwrite(str(path), frame):
        raise IOError(f"Failed to write frame to {path}")
    return path


def frame_index_to_time(frame_idx: int, fps: float) -> float:
    """Convert a frame index to seconds."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    return frame_idx / fps


@dataclass(slots=True)
class VideoMetadata:
    """Basic metadata inferred from an input video."""

    fps: float
    frame_count: int
    duration: float
    width: int
    height: int
    codec: str


def iter_batches(sequence: Sequence, batch_size: int) -> Iterator[Sequence]:
    """Yield fixed-size batches from a sequence."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for idx in range(0, len(sequence), batch_size):
        yield sequence[idx : idx + batch_size]


def hist_diff(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """Compute chi-square distance between two histograms."""
    return cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CHISQR)


def normalize_histogram(hist: np.ndarray) -> np.ndarray:
    """Normalize a histogram so that it sums to one."""
    total = hist.sum()
    if not total:
        return hist
    return hist / total
