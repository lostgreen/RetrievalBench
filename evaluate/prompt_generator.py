"""Prompt generation utilities for evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .config import EvalConfig
from .utils import ShotSample
from .data_loader import QuestionItem


@dataclass
class PromptBundle:
    """Holds prompt strings and metadata for evaluation prompting."""

    overview_prompt: str
    question_id: str | None
    question: str | None
    options: Sequence[str]
    shot_summaries: Sequence[str]


def _format_time(frame_idx: int, fps: float) -> float:
    if fps and fps > 0:
        return frame_idx / fps
    return float(frame_idx)


def generate_prompt(
    shots: Sequence[ShotSample],
    cfg: EvalConfig,
    questions: Sequence[QuestionItem] | None = None,
    fps: float | None = None,
) -> PromptBundle:
    """Generate prompts based on shot metadata and dataset questions."""

    primary_question = questions[0] if questions else None
    intro_lines = [
        "You are analyzing key shots extracted from a video.",
        "Shots are identified by their temporal range in seconds. Higher boundary_score indicates stronger scene changes.",
    ]

    if primary_question:
        intro_lines.append(
            f"Answer the following question based on the shots: {primary_question.question}"
        )
        if primary_question.options:
            option_lines = [f"{chr(65 + idx)}. {text}" for idx, text in enumerate(primary_question.options)]
            intro_lines.extend(option_lines)

    overview_prompt = "\n".join(intro_lines)

    shot_summaries: list[str] = []
    effective_fps = fps or 0.0
    for idx, shot in enumerate(shots, start=1):
        start_time = _format_time(shot.start_frame, effective_fps)
        end_time = _format_time(shot.end_frame, effective_fps)
        summary = (
            f"Shot {idx}: time {start_time:.2f}s - {end_time:.2f}s, boundary_score={shot.score:.2f}"
        )
        shot_summaries.append(summary)

    if shot_summaries:
        overview_prompt = overview_prompt + "\n" + "\n".join(shot_summaries)

    return PromptBundle(
        overview_prompt=overview_prompt,
        question_id=primary_question.question_id if primary_question else None,
        question=primary_question.question if primary_question else None,
        options=primary_question.options if primary_question else (),
        shot_summaries=shot_summaries,
    )
