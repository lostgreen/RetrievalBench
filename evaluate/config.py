"""Configuration for the evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence


@dataclass
class EvalConfig:
    """Runtime parameters for evaluation."""

    shots_root: Path = Path("output/videomme_batch")
    api_bank_root: Path = Path("/data4/zgw/APIBank")
    model_name: str = "gemini-2.0-flash-lite"
    cache_dir: Path = Path("runs/evaluation_cache")
    metrics: Sequence[str] = ("k_at_b", "auic", "oracle_regret")
    prompt_template: str = "default"
    question_file: Path = Path("/data5/zgw/video_datasets/Video-MME/videomme/videomme_question.json")
    batch_size: int = 4
    max_budget: int = 10
    budget_steps: int = 5
    verbose: bool = True
    request_timeout: int = 30
    save_predictions: bool = True
    predictions_dir: Path = Path("runs/evaluation_predictions")
    # Enable action-planning (AIF-V) round-1 with explicit actions under budget
    enable_action_planning: bool = False
    round1_budget_token: float = 20.0
    # Cost model
    cost_table_token: Mapping[str, float] = field(
        default_factory=lambda: {
            "peek_scene": 1.0,
            "peek_shot": 0.5,
            "request_hd_frame": 5.0,
            "request_clip_1s": 15.0,
            "answer": 0.5,
        }
    )
    cost_table_latency: Mapping[str, float] = field(
        default_factory=lambda: {
            "peek_scene": 1.0,
            "peek_shot": 0.5,
            "request_hd_frame": 5.0,
            "request_clip_1s": 15.0,
            "answer": 0.5,
        }
    )
    budgets_token: Sequence[float] = (10.0, 20.0, 30.0, 40.0)
    acc_targets: Sequence[float] = (0.6, 0.7, 0.8, 0.9)

    def resolve_cache_dir(self) -> Path:
        path = self.cache_dir.expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def resolve_predictions_dir(self) -> Path:
        path = self.predictions_dir.expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path


DEFAULT_EVAL_CONFIG = EvalConfig()
