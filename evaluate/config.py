"""Configuration for the evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


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

    def resolve_cache_dir(self) -> Path:
        path = self.cache_dir.expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def resolve_predictions_dir(self) -> Path:
        path = self.predictions_dir.expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path


DEFAULT_EVAL_CONFIG = EvalConfig()
