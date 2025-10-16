"""Data processing package for shot boundary detection pipeline."""

from .config import Config, DEFAULT_CONFIG
from .representative_frame import select_representative_frame
from .shot_boundary import detect_shot_boundaries
from .video_loader import load_video

__all__ = [
    "Config",
    "DEFAULT_CONFIG",
    "load_video",
    "detect_shot_boundaries",
    "select_representative_frame",
]
