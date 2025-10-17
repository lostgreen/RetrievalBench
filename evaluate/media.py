"""Media helper utilities for frame extraction and encoding."""

from __future__ import annotations

import base64
from contextlib import contextmanager
from typing import List, Sequence

try:  # Optional dependency; image embedding will be skipped if unavailable
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional
    cv2 = None  # type: ignore


@contextmanager
def video_capture(path: str):
    """Context manager that opens a video file and releases it on exit.

    Raises ImportError if OpenCV is not installed.
    Raises IOError if the file cannot be opened.
    """
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for image embedding but is not installed.")
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {path}")
        yield cap
    finally:
        try:
            cap.release()
        except Exception:
            pass


def encode_frame_b64(cap: "cv2.VideoCapture", frame_index: int) -> str:
    """Read a frame by index and return a data URL with base64-encoded JPEG."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    success, frame = cap.read()
    if not success or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_index}")
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def encode_crop_b64(cap: "cv2.VideoCapture", frame_index: int, bbox: Sequence[float]) -> str:
    """Read a frame and return a base64 JPEG of the cropped bbox region.

    bbox is [x1, y1, x2, y2] in normalized coordinates.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    success, frame = cap.read()
    if not success or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_index}")
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, int(round(float(bbox[0]) * w))))
    y1 = max(0, min(h - 1, int(round(float(bbox[1]) * h))))
    x2 = max(0, min(w, int(round(float(bbox[2]) * w))))
    y2 = max(0, min(h, int(round(float(bbox[3]) * h))))
    if x2 <= x1 or y2 <= y1:
        crop = frame
    else:
        crop = frame[y1:y2, x1:x2]
    ok, buf = cv2.imencode(".jpg", crop)
    if not ok:
        raise RuntimeError("Failed to encode crop as JPEG")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def evenly_spaced_frames(start_frame: int, end_frame: int, k: int = 3) -> List[int]:
    """Return k indices evenly spaced between start_frame and end_frame (inclusive)."""
    if k <= 1:
        return [start_frame]
    if end_frame < start_frame:
        start_frame, end_frame = end_frame, start_frame
    length = max(1, end_frame - start_frame)
    if length == 0:
        return [start_frame]
    return [int(round(start_frame + i * (end_frame - start_frame) / (k - 1))) for i in range(k)]
