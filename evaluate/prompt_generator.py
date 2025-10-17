"""Prompt generation utilities for evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from evaluate.data_loader import QuestionItem
from evaluate.utils import ShotSample


def _format_time(frame_idx: int, fps: float) -> float:
    """Formats a frame index into a timestamp in seconds."""
    if fps and fps > 0:
        return frame_idx / fps
    return float(frame_idx)


class BasePromptGenerator(ABC):
    """Abstract base class for prompt generators."""

    def __init__(self, prompt_template: str):
        self.template = prompt_template

    @abstractmethod
    def generate_prompt(
        self,
        shots: Sequence[ShotSample],
        questions: Sequence[QuestionItem] | None = None,
        fps: float | None = None,
        **kwargs,
    ) -> str:
        """Generate a prompt string."""
        raise NotImplementedError


class ScenePromptGenerator(BasePromptGenerator):
    """Generates prompts for scene-level (Round 1) analysis."""

    def generate_prompt(
        self,
        shots: Sequence[ShotSample],
        questions: Sequence[QuestionItem] | None = None,
        fps: float | None = None,
        **kwargs,
    ) -> str:
        """Generates a prompt with representative frames for each shot/scene."""
        intro_lines = [
            "You are analyzing key scenes extracted from a video.",
            "Each scene is represented by one or two keyframes and is identified by an index.",
            "Your task is to determine which scenes are most relevant for answering a potential question.",
            "Please output the indices of the scenes you need to inspect more closely.",
        ]

        if questions:
            primary_question = questions[0]
            intro_lines.append(
                f"The question to answer is: '{primary_question.question}'"
            )
            if primary_question.options:
                option_lines = [
                    f"{chr(65 + idx)}. {text}"
                    for idx, text in enumerate(primary_question.options)
                ]
                intro_lines.extend(option_lines)

        scene_summaries: list[str] = []
        effective_fps = fps or 30.0
        # This assumes we get one representative frame per shot/scene for Round 1
        for shot in shots:
            start_time = _format_time(shot.start_frame, effective_fps)
            end_time = _format_time(shot.end_frame, effective_fps)
            summary = (
                f"Scene {shot.shot_id}: time {start_time:.2f}s - {end_time:.2f}s. "
                f"Representative frame index {shot.representative_index}."
            )
            scene_summaries.append(summary)

        prompt = "\n".join(intro_lines)
        if scene_summaries:
            prompt += "\n\n" + "\n".join(scene_summaries)

        return prompt


class ShotPromptGenerator(BasePromptGenerator):
    """Generates prompts for shot-level (Round 2) analysis."""

    def generate_prompt(
        self,
        shots: Sequence[ShotSample],
        questions: Sequence[QuestionItem] | None = None,
        fps: float | None = None,
        **kwargs,
    ) -> str:
        """Generates a prompt with all frames for the selected shots."""
        intro_lines = [
            "You are analyzing the detailed frames from a set of selected video shots.",
            "Use this detailed information to provide a comprehensive answer.",
        ]

        shot_details: list[str] = []
        effective_fps = fps or 30.0
        for shot in shots:
            start_time = _format_time(shot.start_frame, effective_fps)
            end_time = _format_time(shot.end_frame, effective_fps)
            # Here, we would ideally include all frame paths or representations
            # For simplicity, we'll just list the frame indices.
            frames = f"Frames {shot.start_frame} to {shot.end_frame}"
            shot_details.append(
                f"Shot {shot.shot_id} (time {start_time:.2f}s - {end_time:.2f}s):\n{frames}"
            )

        prompt = "\n".join(intro_lines)
        if shot_details:
            prompt += "\n\n" + "\n".join(shot_details)

        return prompt
