"""Create grid visualizations for multiple representative frame strategies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np

from .shot_grid import save_shot_grid


DEFAULT_STRATEGIES = ("first", "middle", "last", "mean", "meanframe")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError("shots JSON must be a dictionary with metadata")
    shots = data.get("shots")
    if shots is None or not isinstance(shots, list):
        raise ValueError("shots JSON missing 'shots' list")
    sampled = data.get("sampled_frame_indices")
    if sampled is None or not isinstance(sampled, list):
        raise ValueError("shots JSON missing 'sampled_frame_indices'")
    return data


def _resolve_video_path(raw_path: str, workspace: Path, shots_dir: Path) -> Path:
    path = Path(raw_path)
    candidates: List[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([workspace / path, shots_dir / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate video file: {raw_path}")


def _read_frame(capture: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    success, frame = capture.read()
    if not success or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_idx}")
    return frame


def _select_frame_first(sample_indices: Sequence[int], shot: dict) -> int:
    return sample_indices[shot["start_frame_sample"]]


def _select_frame_last(sample_indices: Sequence[int], shot: dict) -> int:
    return sample_indices[shot["end_frame_sample"]]


def _select_frame_middle(sample_indices: Sequence[int], shot: dict) -> int:
    start = shot["start_frame_sample"]
    end = shot["end_frame_sample"]
    mid = start + (end - start) // 2
    return sample_indices[mid]


def _select_frame_mean(
    capture: cv2.VideoCapture,
    sample_indices: Sequence[int],
    shot: dict,
) -> tuple[int, np.ndarray]:
    start = shot["start_frame_sample"]
    end = shot["end_frame_sample"]
    indices = sample_indices[start : end + 1]
    raw_frames = []
    float_frames = []
    for idx in indices:
        frame = _read_frame(capture, idx)
        raw_frames.append(frame)
        float_frames.append(frame.astype(np.float32))
    mean_frame = np.mean(float_frames, axis=0)
    distances = [float(np.mean(np.abs(frame - mean_frame))) for frame in float_frames]
    best_idx = int(np.argmin(distances))
    return indices[best_idx], raw_frames[best_idx]


def _select_frame_meanframe(
    capture: cv2.VideoCapture,
    sample_indices: Sequence[int],
    shot: dict,
) -> np.ndarray:
    start = shot["start_frame_sample"]
    end = shot["end_frame_sample"]
    indices = sample_indices[start : end + 1]
    frames = []
    for idx in indices:
        frame = _read_frame(capture, idx).astype(np.float32)
        frames.append(frame)
    if not frames:
        raise ValueError("Shot contains no frames to average")
    mean_frame = np.mean(frames, axis=0)
    mean_frame = np.clip(mean_frame, 0, 255).astype(np.uint8)
    return mean_frame


def _extract_strategy_frames(
    video_path: Path,
    sample_indices: Sequence[int],
    shots: Sequence[dict],
    strategy: str,
) -> List[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    frames: List[np.ndarray] = []
    try:
        for shot in shots:
            if strategy == "first":
                frame_idx = _select_frame_first(sample_indices, shot)
                frame = _read_frame(capture, frame_idx)
            elif strategy == "last":
                frame_idx = _select_frame_last(sample_indices, shot)
                frame = _read_frame(capture, frame_idx)
            elif strategy == "middle":
                frame_idx = _select_frame_middle(sample_indices, shot)
                frame = _read_frame(capture, frame_idx)
            elif strategy == "mean":
                frame_idx, frame = _select_frame_mean(capture, sample_indices, shot)
            elif strategy == "meanframe":
                frame = _select_frame_meanframe(capture, sample_indices, shot)
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")
            frames.append(frame)
    finally:
        capture.release()

    return frames


def _strategy_output_path(base: Path, strategy: str) -> Path:
    if base.suffix:
        return base.with_name(f"{base.stem}_{strategy}{base.suffix}")
    return base / f"shots_grid_{strategy}.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize representative frames with multiple strategies.")
    parser.add_argument("shots_json", type=str, help="Path to the shots.json file.")
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Workspace root to resolve relative paths (defaults to current working directory).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory or base path. Defaults to shots directory.",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="*",
        default=list(DEFAULT_STRATEGIES),
        help="Representative frame strategies to visualize.",
    )
    parser.add_argument("--rows", type=int, default=None, help="Number of rows in the grid.")
    parser.add_argument("--cols", type=int, default=None, help="Number of columns in the grid.")
    parser.add_argument("--height", type=int, default=224, help="Tile height for each frame.")
    parser.add_argument("--padding", type=int, default=10, help="Padding in pixels between tiles.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shots_path = Path(args.shots_json).expanduser().resolve()
    if not shots_path.exists():
        raise FileNotFoundError(f"shots file not found: {shots_path}")

    strategies = args.strategies or list(DEFAULT_STRATEGIES)
    strategies = [s.lower() for s in strategies]
    unknown = sorted(set(strategies) - set(DEFAULT_STRATEGIES))
    if unknown:
        raise ValueError(f"Unsupported strategies requested: {', '.join(unknown)}")

    workspace = Path(args.workspace).expanduser().resolve() if args.workspace else Path.cwd()
    data = _load_json(shots_path)
    shots = data["shots"]
    sample_indices = data["sampled_frame_indices"]

    video_field = data.get("video")
    if not video_field:
        raise ValueError("shots JSON missing 'video' field")
    video_path = _resolve_video_path(video_field, workspace, shots_path.parent)

    output_base = Path(args.output).expanduser() if args.output else shots_path.parent
    output_base = output_base.resolve()
    if not output_base.suffix:
        output_base.mkdir(parents=True, exist_ok=True)
    else:
        output_base.parent.mkdir(parents=True, exist_ok=True)

    for strategy in strategies:
        frames = _extract_strategy_frames(video_path, sample_indices, shots, strategy)
        output_path = _strategy_output_path(output_base, strategy)
        save_shot_grid(
            frames,
            output_path,
            rows=args.rows,
            cols=args.cols,
            target_height=args.height,
            padding=args.padding,
        )
        print(f"Saved {strategy} grid to {output_path}")


if __name__ == "__main__":
    main()
