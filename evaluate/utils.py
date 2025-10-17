"""Utility helpers for evaluation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from evaluate.config import EvalConfig

LOGGER = logging.getLogger("evaluate")


def configure_logging(verbose: bool) -> None:
    """Configure logging.

    - Root logger stays at INFO to avoid third-party DEBUG noise (httpx/openai etc.).
    - Our namespace logger "evaluate" is set to DEBUG when verbose=True.
    - Noisy external loggers are clamped to WARNING.
    """
    # Always initialize root at INFO to suppress third-party DEBUG logs
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Our project logger
    eval_logger = logging.getLogger("evaluate")
    eval_logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Quiet noisy libraries regardless of verbose
    for name in (
        "httpx",
        "httpcore",
        "urllib3",
        "openai",
        "openai._base_client",
        "openai._http_client",
    ):
        ext_logger = logging.getLogger(name)
        ext_logger.setLevel(logging.WARNING)


def read_json(path: Path) -> Any:
    """Read a JSON file."""
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def write_json(data: Any, path: Path) -> None:
    """Persist data to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)


def list_shot_files(root: Path) -> List[Path]:
    """Discover shots.json files under a directory."""
    if root.is_file() and root.name.endswith("shots.json"):
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"Shots root does not exist: {root}")
    return sorted(root.rglob("shots.json"))


def ensure_api_bank(cfg: EvalConfig) -> Path:
    """Verify that the APIBank directory exists."""
    api_path = cfg.api_bank_root.expanduser()
    if not api_path.exists():
        raise FileNotFoundError(
            f"APIBank directory not found at {api_path}. "
            "Please clone https://github.com/VPGTrans/APIBank or adjust EvalConfig."
        )
    return api_path


@dataclass
class ShotSample:
    """Structured representation of a shot entry."""

    shot_id: int
    start_frame: int
    end_frame: int
    score: float
    representative_index: int
    metadata: Mapping[str, Any]


def parse_shots(shots_json: Mapping[str, Any]) -> List[ShotSample]:
    """Convert raw shots JSON into ShotSample objects."""
    samples: List[ShotSample] = []
    for idx, entry in enumerate(shots_json.get("shots", [])):
        rep = entry.get("representative_frame", {})
        samples.append(
            ShotSample(
                shot_id=idx,
                start_frame=int(entry.get("start_frame", -1)),
                end_frame=int(entry.get("end_frame", -1)),
                score=float(entry.get("score", 0.0)),
                representative_index=int(rep.get("index", -1)),
                metadata=entry,
            )
        )
    return samples


def load_videomme_questions(question_file: Path) -> Dict[str, List[Mapping[str, Any]]]:
    """Load Video-MME questions grouped by video identifier."""

    questions_by_video: Dict[str, List[Mapping[str, Any]]] = {}
    with question_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                LOGGER.warning("Failed to parse question JSON line: %s", exc)
                continue
            video_id = item.get("videoID") or item.get("video_id")
            if not video_id:
                continue
            questions_by_video.setdefault(video_id, []).append(item)
    return questions_by_video
