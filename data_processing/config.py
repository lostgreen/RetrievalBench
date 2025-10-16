"""Centralized configuration for the video data processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Holds runtime parameters for the data processing pipeline."""

    video_path: Path = Path("data/video.mp4")
    output_dir: Path = Path("output")
    frame_rate: float = 1.0  # frames per second to sample
    shot_boundary_threshold: float = 30.0
    histogram_bins: int = 32
    representative_frame_strategy: str = "first"
    save_frames: bool = False
    save_features: bool = False
    feature_model: str | None = None
    random_seed: int = 42
    verbose: bool = True
    _output_frame_dir: Path = field(default=Path("frames"), repr=False)

    @property
    def output_frame_dir(self) -> Path:
        """Directory used to dump sampled frames."""
        return self.output_dir / self._output_frame_dir


DEFAULT_CONFIG = Config()
