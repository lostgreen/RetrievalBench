"""Video loading utilities for frame extraction."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .config import Config, DEFAULT_CONFIG
from .utils import VideoMetadata, ensure_dir, save_frame_as_image


def _compute_sampling_interval(fps: float, target_rate: float) -> int:
    """Compute the frame sampling interval."""
    if target_rate <= 0:
        raise ValueError("target_rate must be positive")
    if fps <= 0:
        return 1
    interval = max(int(round(fps / target_rate)), 1)
    return interval


def _read_metadata(capture: cv2.VideoCapture) -> VideoMetadata:
    """Extract basic metadata from an opened capture."""
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC) or 0)
    codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip()
    duration = float(total_frames / fps) if fps > 0 else 0.0
    return VideoMetadata(
        fps=fps,
        frame_count=total_frames,
        duration=duration,
        width=width,
        height=height,
        codec=codec,
    )


def load_video(
    video_path: str | Path,
    frame_rate: Optional[float] = None,
    config: Config | None = None,
    save_frames: Optional[bool] = None,
    output_dir: Path | None = None,
) -> Tuple[List[np.ndarray], VideoMetadata, List[int]]:
    """
    Load a video file and sample frames according to the desired frame rate.

    Returns the sampled frames, the inferred metadata, and the original frame indices.
    """
    cfg = config or DEFAULT_CONFIG
    path = Path(video_path or cfg.video_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise IOError(f"Could not open video file: {path}")

    metadata = _read_metadata(capture)
    target_rate = frame_rate if frame_rate is not None else cfg.frame_rate
    interval = _compute_sampling_interval(metadata.fps, target_rate)

    should_save_frames = cfg.save_frames if save_frames is None else save_frames
    frame_output_dir = output_dir or cfg.output_frame_dir
    if should_save_frames:
        ensure_dir(frame_output_dir)

    frames: List[np.ndarray] = []
    frame_indices: List[int] = []
    frame_idx = 0
    sampled_count = 0

    while True:
        success, frame = capture.read()
        if not success:
            break
        if frame_idx % interval == 0:
            # copy to avoid OpenCV buffer reuse issues
            sampled_frame = frame.copy()
            frames.append(sampled_frame)
            frame_indices.append(frame_idx)
            if should_save_frames:
                filename = f"{path.stem}_frame_{frame_idx:06d}.jpg"
                save_frame_as_image(sampled_frame, Path(frame_output_dir) / filename)
            sampled_count += 1
        frame_idx += 1

    capture.release()

    if not frames:
        raise RuntimeError(f"No frames sampled from video: {path}")

    if cfg.verbose:
        print(
            f"Loaded {path.name}: {sampled_count} frames sampled "
            f"of {metadata.frame_count} total (fps={metadata.fps:.2f})"
        )

    return frames, metadata, frame_indices
