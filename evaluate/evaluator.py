"""Model evaluation logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from .config import EvalConfig
from .metrics import MetricResult, compute_k_at_b, compute_oracle_regret, summarize_metrics
from .prompt_generator import generate_prompt
from .utils import ShotSample
from .data_loader import QuestionItem


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


def evaluate_model(
    model: ModelInterface,
    shots: Sequence[ShotSample],
    cfg: EvalConfig,
    questions: Sequence[QuestionItem] | None = None,
    video_path: str | None = None,
    fps: float | None = None,
) -> EvaluationResult:
    """Run the model against shots and compute metrics."""
    prompts = generate_prompt(shots, cfg, questions=questions, fps=fps)
    video_path_str = video_path or "unknown"
    model_output = model.predict(
        prompts.overview_prompt,
        shot_count=len(shots),
        video_path=video_path_str,
        question=prompts.question,
        question_id=prompts.question_id,
        options=list(prompts.options),
        shot_summaries=list(prompts.shot_summaries),
        prompt_bundle=prompts,
    )

    predicted_order = model_output.get("shot_order", [])
    if not isinstance(predicted_order, list):
        predicted_order = list(predicted_order)
    predicted_indices = [int(idx) for idx in predicted_order]
    if not predicted_indices:
        predicted_indices = list(range(len(shots)))
    ground_truth_indices = [shot.representative_index for shot in shots]
    budgets = list(range(1, min(len(predicted_indices), cfg.max_budget) + 1))

    k_at_b_result = compute_k_at_b(predicted_indices, ground_truth_indices, budgets)
    k_at_b_result = summarize_metrics(k_at_b_result)

    oracle_recall = 1.0
    achieved_recall = k_at_b_result.values.get("final_recall", 0.0)
    regret_value = compute_oracle_regret(achieved_recall, oracle_recall)
    regret_result = MetricResult(name="oracle_regret", values={"value": regret_value})

    metrics = {
        k_at_b_result.name: k_at_b_result,
        regret_result.name: regret_result,
    }

    return EvaluationResult(
        video_path=model_output.get("video_path", video_path_str),
        metrics=metrics,
        raw_predictions=model_output,
    )
