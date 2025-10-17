"""Data loading utilities for evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Sequence

from evaluate.config import EvalConfig
from evaluate.utils import (
    ShotSample,
    list_shot_files,
    load_videomme_questions,
    parse_shots,
    read_json,
)


@dataclass
class QuestionItem:
    """Question metadata used for prompting."""

    question_id: str
    question: str
    options: Sequence[str]
    answer: str | None = None
    meta: Mapping[str, Any] | None = None


@dataclass
class VideoExample:
    """Represents a video and its associated shot metadata."""

    video_path: Path
    shots_path: Path
    shots_json: Mapping[str, Any]
    shots: Sequence[ShotSample]
    questions: Sequence[QuestionItem]
    fps: float


class EvaluationDataset:
    """Collection of video examples used during evaluation."""

    def __init__(self, examples: Sequence[VideoExample]):
        self._examples = list(examples)

    def __len__(self) -> int:
        return len(self._examples)

    def __iter__(self):
        return iter(self._examples)

    def videos(self) -> List[Path]:
        return [example.video_path for example in self._examples]


def _resolve_video_path(shots_json: Mapping[str, Any], shots_file: Path) -> Path:
    raw = shots_json.get("video")
    if not raw:
        raise ValueError(f"Shots JSON missing 'video' field: {shots_file}")
    path = Path(raw)
    if not path.is_absolute():
        path = (shots_file.parent / path).resolve()
    return path


def load_video_data(cfg: EvalConfig, shots_root: Path | None = None) -> EvaluationDataset:
    """Load shot metadata for evaluation."""

    root = (shots_root or cfg.shots_root).expanduser()
    shot_files = list_shot_files(root)
    questions_by_video: dict[str, List[Mapping[str, Any]]] = {}
    if cfg.question_file:
        question_path = cfg.question_file.expanduser()
        if question_path.exists():
            questions_by_video = load_videomme_questions(question_path)
    examples: List[VideoExample] = []

    for shots_file in shot_files:
        shots_json = read_json(shots_file)
        shots = parse_shots(shots_json)
        video_path = _resolve_video_path(shots_json, shots_file)
        video_id = shots_json.get("video_name") or video_path.stem
        raw_questions = questions_by_video.get(str(video_id), [])
        questions: List[QuestionItem] = []
        for entry in raw_questions:
            questions.append(
                QuestionItem(
                    question_id=str(entry.get("question_id")),
                    question=str(entry.get("question")),
                    options=list(entry.get("options", [])),
                    answer=entry.get("answer"),
                    meta=entry,
                )
            )
        fps = float(shots_json.get("fps", 0.0))
        examples.append(
            VideoExample(
                video_path=video_path,
                shots_path=shots_file,
                shots_json=shots_json,
                shots=shots,
                questions=questions,
                fps=fps,
            )
        )

    return EvaluationDataset(examples)
