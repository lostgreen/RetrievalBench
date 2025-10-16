"""Top-level evaluation pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

from .config import DEFAULT_EVAL_CONFIG, EvalConfig
from .data_loader import EvaluationDataset, load_video_data
from .evaluator import EvaluationResult, ModelInterface
from .prompt_generator import ScenePromptGenerator, ShotPromptGenerator
from .utils import LOGGER, configure_logging, ensure_api_bank, write_json


class APIBankModelAdapter(ModelInterface):
    """Adapter that uses APIBank to rotate Gemini API keys."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

    def __init__(self, cfg: EvalConfig):
        super().__init__(name=cfg.model_name)
        self.cfg = cfg

        try:
            from APIBank import build_default_bank  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - configuration issue
            raise ImportError(
                "APIBank package is not available on PYTHONPATH; "
                "add the APIBank directory or install it as a package."
            ) from exc

        self._bank = build_default_bank()

        try:
            from openai import OpenAI  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            LOGGER.warning(
                "The 'openai' package is not installed. Falling back to heuristic predictions."
            )
            self._client_cls = None
        else:
            self._client_cls = OpenAI

    @staticmethod
    def _fallback_order(shot_count: int) -> list[int]:
        return list(range(shot_count))

    @staticmethod
    def _extract_indices(text: str, shot_count: int) -> list[int]:
        indices: list[int] = []
        sanitized = text.replace("\n", " ").replace(",", " ")
        for token in sanitized.split():
            if token.isdigit():
                value = int(token)
                if 0 <= value < shot_count:
                    indices.append(value)
        if not indices:
            indices = list(range(shot_count))
        return indices

    def predict(self, prompt: str, **kwargs):  # type: ignore[override]
        shot_count = int(kwargs.get("shot_count", 0)) or 0
        video_path = kwargs.get("video_path", "unknown")
        messages = kwargs.get("messages", None)

        if self._client_cls is None:
            return {
                "video_path": video_path,
                "shot_order": self._fallback_order(shot_count),
                "explanation": "openai package missing; using fallback ordering",
            }

        if messages is None:
            # Original behavior if messages are not provided
            question = kwargs.get("question")
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that selects relevant video shots.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
            if question:
                messages.append({"role": "user", "content": f"Question: {question}"})

        with self._bank.checkout() as key:
            client = self._client_cls(api_key=key.secret, base_url=self.BASE_URL)

            try:
                response = client.chat.completions.create(  # type: ignore[attr-defined]
                    model=self.cfg.model_name,
                    messages=messages,
                    max_tokens=1024,  # Increased max_tokens for potentially longer answers
                )
            except Exception as error:  # pragma: no cover - depends on external API
                LOGGER.error("Model request failed: %s", error)
                self._bank.report_failure(key)
                return {
                    "video_path": video_path,
                    "shot_order": self._fallback_order(shot_count),
                    "error": str(error),
                }

            choice = response.choices[0].message  # type: ignore[index]
            content = getattr(choice, "content", "") or ""

            # In round 1, we expect indices. In round 2, we expect a textual response.
            if kwargs.get("is_round_1", False):
                indices = self._extract_indices(content, shot_count)
                return {
                    "video_path": video_path,
                    "shot_order": indices,
                    "raw_completion": content,
                    "used_key": key.alias or key.fingerprint,
                }
            else:
                return {
                    "video_path": video_path,
                    "final_response": content,
                    "raw_completion": content,
                    "used_key": key.alias or key.fingerprint,
                }


def _load_model(cfg: EvalConfig) -> ModelInterface:
    api_root = ensure_api_bank(cfg)
    parent_dir = api_root.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    return APIBankModelAdapter(cfg)


def run_single_video(
    model: ModelInterface,
    dataset: EvaluationDataset,
    cfg: EvalConfig,
) -> List[EvaluationResult]:
    results: List[EvaluationResult] = []
    for example in dataset:
        # ROUND 1: Scene-level analysis for shot selection
        scene_prompt_generator = ScenePromptGenerator(cfg.prompt)
        scene_prompt = scene_prompt_generator.generate_prompt(
            example.shots, representative_frame_method="middle"
        )

        round_1_prediction = model.predict(
            scene_prompt,
            shot_count=len(example.shots),
            video_path=str(example.video_path),
            is_round_1=True,  # Flag for round 1
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant. Your task is to analyze video scenes and identify which shots are most relevant to answer a potential user query. Please output the indices of the shots you select as a comma-separated list.",
                },
                {"role": "user", "content": scene_prompt},
            ],
        )

        selected_shot_indices = round_1_prediction.get("shot_order", [])
        round_1_explanation = round_1_prediction.get("raw_completion", "")

        # Select shots based on model's choice
        selected_shots = [example.shots[i] for i in selected_shot_indices]

        # ROUND 2: Shot-level analysis for final answer
        shot_prompt_generator = ShotPromptGenerator(cfg.prompt)
        shot_prompt = shot_prompt_generator.generate_prompt(selected_shots)

        # Prepare messages for round 2, including CoT from round 1
        round_2_messages = [
            {
                "role": "system",
                "content": "You are an AI assistant. You have previously selected a number of relevant shots from a video. Now, you will be given the detailed frames from those shots. Your task is to use this information to provide a comprehensive answer.",
            },
            {
                "role": "user",
                "content": f"Here is my reasoning for selecting these shots:\n---\n{round_1_explanation}\n---\nNow, here are the detailed frames from the shots I selected:",
            },
            {"role": "user", "content": shot_prompt},
        ]

        if example.questions:
            round_2_messages.append(
                {
                    "role": "user",
                    "content": f"Please answer the following question based on the video content: {example.questions[0]}",
                }
            )

        round_2_prediction = model.predict(
            shot_prompt,  # prompt is not the main input here, messages are
            video_path=str(example.video_path),
            messages=round_2_messages,
            is_round_1=False,  # Flag for round 2
        )

        # Create a result object. Metrics calculation will be handled later.
        result = EvaluationResult(
            video_path=str(example.video_path),
            metrics={},  # Metrics would be calculated here
            raw_predictions={
                "round_1": round_1_prediction,
                "round_2": round_2_prediction,
            },
        )
        results.append(result)

    return results


def aggregate_metrics(results: Sequence[EvaluationResult]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for result in results:
        for metric in result.metrics.values():
            for key, value in metric.values.items():
                summary.setdefault(f"{metric.name}/{key}", 0.0)
                summary[f"{metric.name}/{key}"] += value
    count = max(len(results), 1)
    return {key: value / count for key, value in summary.items()}


def run_evaluation(cfg: EvalConfig, shots_root: Path | None = None) -> Dict[str, float]:
    configure_logging(cfg.verbose)
    dataset = load_video_data(cfg, shots_root=shots_root)
    model = _load_model(cfg)
    results = run_single_video(model, dataset, cfg)

    if cfg.save_predictions:
        predictions_dir = cfg.resolve_predictions_dir()
        for result in results:
            stem = Path(result.video_path).stem or "unknown"
            write_json(
                {
                    "video": result.video_path,
                    "metrics": {name: metric.values for name, metric in result.metrics.items()},
                    "raw_predictions": result.raw_predictions,
                },
                predictions_dir / f"{stem}_predictions.json",
            )

    summary = aggregate_metrics(results)
    write_json(summary, cfg.resolve_cache_dir() / "metrics_summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the video reasoning evaluation pipeline.")
    parser.add_argument("--shots-root", type=str, help="Directory containing shots.json files.")
    parser.add_argument("--model-name", type=str, help="Model identifier for logging.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging verbosity.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DEFAULT_EVAL_CONFIG
    overrides = {}
    if args.model_name:
        overrides["model_name"] = args.model_name
    if args.quiet:
        overrides["verbose"] = False
    if overrides:
        cfg = EvalConfig(**{**cfg.__dict__, **overrides})

    shots_root = Path(args.shots_root).expanduser() if args.shots_root else cfg.shots_root
    summary = run_evaluation(cfg, shots_root=shots_root)
    print("Evaluation summary:")
    for key, value in summary.items():
        print(f" - {key}: {value:.4f}")


if __name__ == "__main__":
    main()
