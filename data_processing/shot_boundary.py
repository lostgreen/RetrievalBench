"""Shot boundary detection based on histogram differences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import cv2
import numpy as np

from .config import Config, DEFAULT_CONFIG
from .utils import VideoMetadata, hist_diff, normalize_histogram


@dataclass(slots=True)
class ShotSegment:
    """Represents a continuous shot within a video."""

    start_frame: int
    end_frame: int
    score: float


def _frame_histogram(frame: np.ndarray, bins: int) -> np.ndarray:
    """Compute a normalized HSV histogram for a frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv],
        channels=[0, 1, 2],
        mask=None,
        histSize=[bins, bins, bins],
        ranges=[0, 180, 0, 256, 0, 256],
    )
    return normalize_histogram(hist).flatten()


def detect_shot_boundaries(
    frames: Sequence[np.ndarray],
    threshold: float | None = None,
    bins: int | None = None,
    config: Config | None = None,
    metadata: VideoMetadata | None = None,
) -> List[ShotSegment]:
    """
    Split the video into shots using histogram based boundary detection.

    Returns a list of shot segments with their frame ranges and
    the contrast score that triggered the cut.
    """
    cfg = config or DEFAULT_CONFIG
    if not frames:
        raise ValueError("frames must not be empty")

    cut_threshold = threshold if threshold is not None else cfg.shot_boundary_threshold
    hist_bins = bins if bins is not None else cfg.histogram_bins

    shot_segments: List[ShotSegment] = []
    start_idx = 0
    prev_hist = _frame_histogram(frames[0], hist_bins)

    for idx in range(1, len(frames)):
        current_hist = _frame_histogram(frames[idx], hist_bins)
        diff = hist_diff(prev_hist, current_hist)

        if diff > cut_threshold:
            shot_segments.append(ShotSegment(start_frame=start_idx, end_frame=idx - 1, score=diff))
            start_idx = idx
        prev_hist = current_hist

    shot_segments.append(ShotSegment(start_frame=start_idx, end_frame=len(frames) - 1, score=0.0))

    if cfg.verbose:
        duration = metadata.duration if metadata else len(frames)
        print(
            f"Detected {len(shot_segments)} shots "
            f"(threshold={cut_threshold}, bins={hist_bins}, duration={duration:.2f})"
        )

    return shot_segments
