"""End-to-end pipeline that stitches together the video processing stages."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from tqdm.auto import tqdm

from .config import Config, DEFAULT_CONFIG
from .representative_frame import STRATEGIES, select_representative_frames
from .shot_boundary import ShotSegment, detect_shot_boundaries
from .utils import ensure_dir, frame_index_to_time
from .video_loader import load_video


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def _build_config(args: argparse.Namespace) -> Config:
    """Construct a Config instance using CLI overrides."""
    cfg = DEFAULT_CONFIG
    overrides = {
        "video_path": Path(args.video) if args.video else cfg.video_path,
        "output_dir": Path(args.output_dir) if args.output_dir else cfg.output_dir,
        "frame_rate": args.frame_rate if args.frame_rate else cfg.frame_rate,
        "shot_boundary_threshold": (
            args.threshold if args.threshold is not None else cfg.shot_boundary_threshold
        ),
        "representative_frame_strategy": (
            args.strategy if args.strategy else cfg.representative_frame_strategy
        ),
        "verbose": not args.quiet,
    }
    if args.save_frames:
        overrides["save_frames"] = True
    elif args.no_save_frames:
        overrides["save_frames"] = False
    return replace(cfg, **overrides)


def _serialize_shot(
    shot: ShotSegment,
    representative_index: int,
    fps: float,
    frame_indices: Sequence[int],
) -> Dict[str, Any]:
    """Convert a shot segment into a JSON-serializable dictionary."""
    rep_original_idx = frame_indices[representative_index]
    start_original_idx = frame_indices[shot.start_frame]
    end_original_idx = frame_indices[shot.end_frame]
    return {
        "start_frame": start_original_idx,
        "end_frame": end_original_idx,
        "start_frame_sample": shot.start_frame,
        "end_frame_sample": shot.end_frame,
        "score": shot.score,
        "representative_frame": {
            "index": rep_original_idx,
            "sample_index": representative_index,
            "timestamp": frame_index_to_time(rep_original_idx, fps),
        },
    }


def run_pipeline(cfg: Config) -> Path:
    """Execute the full pipeline and return the path to the summary file."""
    ensure_dir(cfg.output_dir)
    frames, metadata, frame_indices = load_video(
        cfg.video_path,
        frame_rate=cfg.frame_rate,
        config=cfg,
        save_frames=cfg.save_frames,
        output_dir=cfg.output_frame_dir,
    )

    if len(frame_indices) != len(frames):
        raise ValueError("Mismatch between sampled frames and frame indices.")

    shots = detect_shot_boundaries(
        frames,
        threshold=cfg.shot_boundary_threshold,
        bins=cfg.histogram_bins,
        config=cfg,
        metadata=metadata,
    )

    representatives = select_representative_frames(
        frames,
        shots,
        strategy=cfg.representative_frame_strategy,
        config=cfg,
    )

    disable_progress = not cfg.verbose
    serialized_shots: List[Dict[str, Any]] = []
    iterator = range(len(shots))
    for idx in tqdm(iterator, desc="Serializing shots", disable=disable_progress):
        shot = shots[idx]
        frame_idx, _ = representatives[idx]
        serialized_shots.append(
            _serialize_shot(
                shot,
                frame_idx,
                metadata.fps,
                frame_indices,
            )
        )

    video_path_resolved = Path(cfg.video_path).expanduser().resolve()
    summary: Dict[str, Any] = {
        "video": str(video_path_resolved),
        "video_name": video_path_resolved.stem,
        "fps": metadata.fps,
        "sample_rate": cfg.frame_rate,
        "frame_count": metadata.frame_count,
        "duration": metadata.duration,
        "width": metadata.width,
        "height": metadata.height,
        "sampled_frame_indices": frame_indices,
        "shots": serialized_shots,
    }

    summary_path = cfg.output_dir / "shots.json"
    ensure_dir(summary_path.parent)
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    if cfg.verbose:
        print(
            f"Pipeline complete: {len(serialized_shots)} shots written to {summary_path} "
            f"(video={cfg.video_path})"
        )

    return summary_path


def _list_videos(root: Path, recursive: bool) -> List[Path]:
    """Collect video files within a directory."""
    walker: Iterable[Path]
    if recursive:
        walker = root.rglob("*")
    else:
        walker = root.glob("*")
    videos = [
        path
        for path in walker
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(videos)


def run_batch(cfg: Config, videos: Sequence[Path]) -> List[Path]:
    """Run the pipeline for each video in the provided sequence."""
    if not videos:
        raise ValueError("No video files found for batch processing.")

    results: List[Path] = []
    base_output_dir = cfg.output_dir
    disable_progress = not cfg.verbose

    ensure_dir(base_output_dir)

    for video_path in tqdm(videos, desc="Videos", disable=disable_progress):
        output_dir = base_output_dir / video_path.stem
        per_video_cfg = replace(cfg, video_path=video_path, output_dir=output_dir)
        results.append(run_pipeline(per_video_cfg))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the video shot detection pipeline.")
    parser.add_argument("--video", type=str, help="Path to the input video file.")
    parser.add_argument("--video-dir", type=str, help="Directory of videos for batch processing.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when used with --video-dir.",
    )
    parser.add_argument("--output-dir", type=str, help="Directory for pipeline outputs.")
    parser.add_argument("--frame-rate", type=float, help="Sampling rate in frames per second.")
    parser.add_argument(
        "--threshold",
        type=float,
        help="Histogram chi-square threshold used for shot boundary detection.",
    )
    parser.add_argument("--strategy", type=str, choices=sorted(STRATEGIES.keys()), help="Representative frame selection strategy.")
    parser.add_argument(
        "--no-save-frames",
        action="store_true",
        help="Disable writing sampled frames to disk.",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Force writing sampled frames to disk, overriding config defaults.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Silence informative logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _build_config(args)
    if args.video_dir:
        root = Path(args.video_dir)
        if not root.exists():
            raise FileNotFoundError(f"Video directory not found: {root}")
        videos = _list_videos(root, recursive=args.recursive)
        if args.video:
            videos.append(Path(args.video))
        # Deduplicate while preserving order
        unique_videos: List[Path] = []
        seen: set[Path] = set()
        for video in videos:
            if video not in seen:
                seen.add(video)
                unique_videos.append(video)
        if not unique_videos:
            raise ValueError(f"No video files found in {root}.")
        run_batch(cfg, unique_videos)
        return

    if args.video:
        run_pipeline(cfg)
        return

    raise ValueError("Please specify --video or --video-dir.")


if __name__ == "__main__":
    main()
