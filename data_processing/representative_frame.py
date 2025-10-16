"""Representative frame selection utilities."""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np

from .config import Config, DEFAULT_CONFIG
from .shot_boundary import ShotSegment

FrameSelector = Callable[[Sequence[np.ndarray], ShotSegment], int]


def _first_frame_selector(frames: Sequence[np.ndarray], shot: ShotSegment) -> int:
    return shot.start_frame


def _last_frame_selector(frames: Sequence[np.ndarray], shot: ShotSegment) -> int:
    return shot.end_frame


def _middle_frame_selector(frames: Sequence[np.ndarray], shot: ShotSegment) -> int:
    return shot.start_frame + (shot.end_frame - shot.start_frame) // 2


def _brightest_frame_selector(frames: Sequence[np.ndarray], shot: ShotSegment) -> int:
    indices = range(shot.start_frame, shot.end_frame + 1)
    brightness = {
        idx: float(frames[idx].mean()) for idx in indices
    }
    return max(brightness, key=brightness.get)


def _motion_frame_selector(frames: Sequence[np.ndarray], shot: ShotSegment) -> int:
    indices = range(shot.start_frame, shot.end_frame + 1)
    prev = None
    scores = {}
    for idx in indices:
        frame = frames[idx].astype(np.float32)
        if prev is None:
            scores[idx] = 0.0
        else:
            scores[idx] = float(np.mean(np.abs(frame - prev)))
        prev = frame
    return max(scores, key=scores.get)


def _mean_frame_selector(frames: Sequence[np.ndarray], shot: ShotSegment) -> int:
    indices = range(shot.start_frame, shot.end_frame + 1)
    shot_frames = np.stack([frames[idx].astype(np.float32) for idx in indices], axis=0)
    mean_frame = np.mean(shot_frames, axis=0).astype(np.float32)
    distances = {
        idx: float(np.mean(np.abs(frames[idx].astype(np.float32) - mean_frame)))
        for idx in indices
    }
    return min(distances, key=distances.get)


STRATEGIES: dict[str, FrameSelector] = {
    "first": _first_frame_selector,
    "last": _last_frame_selector,
    "middle": _middle_frame_selector,
    "brightest": _brightest_frame_selector,
    "motion": _motion_frame_selector,
    "mean": _mean_frame_selector,
}


def select_representative_frame(
    frames: Sequence[np.ndarray],
    shot: ShotSegment,
    strategy: str | None = None,
    selector: FrameSelector | None = None,
    config: Config | None = None,
) -> Tuple[int, np.ndarray]:
    """
    Select a representative frame for a given shot.

    Returns the index of the selected frame along with the frame data.
    """
    cfg = config or DEFAULT_CONFIG
    chosen_strategy = strategy or cfg.representative_frame_strategy
    if selector is not None:
        frame_idx = selector(frames, shot)
    else:
        if chosen_strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy: {chosen_strategy}")
        frame_idx = STRATEGIES[chosen_strategy](frames, shot)

    if frame_idx < shot.start_frame or frame_idx > shot.end_frame:
        raise ValueError("Selected frame index falls outside shot boundaries")

    return frame_idx, frames[frame_idx]


def select_representative_frames(
    frames: Sequence[np.ndarray],
    shots: Sequence[ShotSegment],
    strategy: str | None = None,
    selector: FrameSelector | None = None,
    config: Config | None = None,
) -> List[Tuple[int, np.ndarray]]:
    """Select representative frames for multiple shots."""
    return [
        select_representative_frame(frames, shot, strategy=strategy, selector=selector, config=config)
        for shot in shots
    ]
