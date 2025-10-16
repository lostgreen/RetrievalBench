"""Utilities to visualize shot-level representative frames in a grid."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

def _resize_frame(frame: np.ndarray, target_height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = target_height / h
    new_w = max(int(w * scale), 1)
    return cv2.resize(frame, (new_w, target_height))


def _pad_frame(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    pad_y = max(target_height - h, 0)
    pad_x = max(target_width - w, 0)
    top = pad_y // 2
    bottom = pad_y - top
    left = pad_x // 2
    right = pad_x - left
    return cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])


def make_shot_grid(
    frames: Sequence[np.ndarray],
    rows: int | None = None,
    cols: int | None = None,
    target_height: int = 224,
    padding: int = 10,
) -> np.ndarray:
    """Compose a grid of representative frames."""
    if not frames:
        raise ValueError("frames must not be empty")

    count = len(frames)
    if rows is None and cols is None:
        cols = math.ceil(math.sqrt(count))
    if rows is None:
        rows = math.ceil(count / cols)
    if cols is None:
        cols = math.ceil(count / rows)

    resized = [_resize_frame(frame, target_height) for frame in frames]
    max_width = max(frame.shape[1] for frame in resized)
    canvas_height = rows * target_height + (rows + 1) * padding
    canvas_width = cols * max_width + (cols + 1) * padding
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    for idx, frame in enumerate(resized):
        r = idx // cols
        c = idx % cols
        y = padding + r * (target_height + padding)
        x = padding + c * (max_width + padding)
        padded = _pad_frame(frame, max_width, target_height)
        canvas[y : y + target_height, x : x + max_width] = padded

    return canvas


def save_shot_grid(
    frames: Sequence[np.ndarray],
    output_path: Path,
    rows: int | None = None,
    cols: int | None = None,
    target_height: int = 224,
    padding: int = 10,
) -> Path:
    """Save a grid visualization to disk."""
    grid = make_shot_grid(
        frames,
        rows=rows,
        cols=cols,
        target_height=target_height,
        padding=padding,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), grid):
        raise IOError(f"Failed to write grid image to {output_path}")
    return output_path
