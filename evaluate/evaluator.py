"""Model evaluation logic."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

from evaluate.data_loader import QuestionItem
from evaluate.metrics import MetricResult


@dataclass
class ModelInterface:
    """Thin abstraction over a reasoning model."""

    name: str

    def predict(self, prompt: str, **kwargs: Any) -> Mapping[str, Any]:
        raise NotImplementedError


@dataclass
class EvaluationResult:
    """Holds per-video evaluation outcomes."""

    video_path: str
    metrics: Dict[str, MetricResult]
    raw_predictions: Mapping[str, Any]


def _extract_choice(response: str, options: Sequence[str]) -> str | None:
    """Extracts the chosen option from a model's text response."""
    # Sanitize the response
    response_lower = response.lower().strip()

    # 1. Check for an explicit "Answer: X" style declaration
    for i, option in enumerate(options):
        letter = chr(65 + i)
        if re.search(rf"answer\s*[:\-]?\s*{letter.lower()}\b", response_lower):
            return option

    # 2. Check for the option letter followed by a period or parenthesis (e.g., "A.", "(B)")
    for i, option in enumerate(options):
        letter = chr(65 + i)
        patterns = [
            f"^{letter.lower()}[\.\)]",  # A., A), a., a)
            f"\({letter.lower()}\)",      # (A), (a)
            f"choice is {letter.lower()}[\.\s]",
            f"option {letter.lower()}[\.\s]",
        ]
        for pattern in patterns:
            if re.search(pattern, response_lower):
                return option

    # 3. Check if the option text itself is in the response
    for option in options:
        if option.lower() in response_lower:
            return option

    return None


def evaluate_multiple_choice(
    model_response: str,
    question: QuestionItem,
) -> Dict[str, MetricResult]:
    """Evaluates a multiple-choice question response."""
    if not question.options or question.answer is None:
        return {}

    predicted_option = _extract_choice(model_response, question.options)

    is_correct = 0
    if predicted_option and predicted_option == question.answer:
        is_correct = 1

    return {
        "multiple_choice_accuracy": MetricResult(
            name="multiple_choice_accuracy",
            values={"accuracy": is_correct},
        )
    }
