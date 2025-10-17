"""Top-level evaluation pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.config import DEFAULT_EVAL_CONFIG, EvalConfig
from evaluate.data_loader import EvaluationDataset, load_video_data
from evaluate.evaluator import EvaluationResult, ModelInterface, evaluate_multiple_choice
from evaluate.prompt_generator import ScenePromptGenerator, ShotPromptGenerator
from evaluate.utils import LOGGER, configure_logging, write_json
from evaluate.evaluator import _extract_choice  # reuse choice extraction
from evaluate.model_adapter import load_model
from evaluate.media import (
    video_capture as _video_capture,
    encode_frame_b64 as _encode_frame_b64,
    evenly_spaced_frames as _evenly_spaced_frames,
)
from evaluate.costs import CostTable, CostLedger, k_at_b, c_at_a, oracle_regret
from evaluate.action_schema import build_planning_system_prompt, parse_action_plan, validate_actions
from evaluate.action_runtime import execute_actions


## Model adapter and media helpers moved to dedicated modules for clarity


def run_single_video(
    model: ModelInterface,
    dataset: EvaluationDataset,
    cfg: EvalConfig,
) -> List[EvaluationResult]:
    results: List[EvaluationResult] = []
    for example in dataset:
        # Initialize cost ledger for this example
        table = CostTable(token=cfg.cost_table_token, latency=cfg.cost_table_latency)
        ledger = CostLedger(table)
        # ROUND 1: Planning or Scene-level selection
        if cfg.enable_action_planning:
            # Build a planning prompt asking the model to choose actions under budget
            budget_token = float(cfg.round1_budget_token)
            sys_prompt = build_planning_system_prompt(cfg, budget_token, shot_count=len(example.shots))
            shot_lines = []
            fps = example.fps or 30.0
            for s in example.shots:
                shot_lines.append(
                    f"shot_id={s.shot_id}: frames[{s.start_frame}-{s.end_frame}] time[{s.start_frame/fps:.2f}-{s.end_frame/fps:.2f}]s"
                )
            user_intro = (
                "Available shots (use shot_id when planning):\n" + "\n".join(shot_lines)
            )
            if example.questions:
                q0 = example.questions[0]
                user_intro += f"\nQuestion: {q0.question}\nOptions: " + "; ".join(q0.options or [])

            round_1_messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_intro},
            ]

            round_1_prediction = model.predict(
                "plan_actions",
                shot_count=len(example.shots),
                video_path=str(example.video_path),
                is_round_1=True,
                messages=round_1_messages,
            )
            round_1_explanation = round_1_prediction.get("raw_completion", "")

            # Parse and validate plan
            plan_obj = parse_action_plan(round_1_explanation) or {"steps": []}
            steps = validate_actions(plan_obj, shot_count=len(example.shots))
            # Enforce budget by truncating steps if needed
            remaining = float(cfg.round1_budget_token)
            bounded_steps: List[Dict] = []
            for st in steps:
                act = st.get("act")
                unit_cost = float(cfg.cost_table_token.get(str(act), 0.0))
                if unit_cost <= remaining:
                    bounded_steps.append(st)
                    remaining -= unit_cost
                else:
                    break
            steps = bounded_steps
            # Execute actions to gather evidence
            evidence_content, executed_steps = execute_actions(example, steps, ledger)
            selected_shots = []  # not used in planning mode
        else:
            scene_prompt_generator = ScenePromptGenerator(cfg.prompt_template)
            scene_prompt = scene_prompt_generator.generate_prompt(
                example.shots,
                questions=example.questions,
                fps=example.fps,
            )

        if not cfg.enable_action_planning:
            # Build multimodal messages for Round 1: include representative frame per shot
            budget_limit = min(cfg.max_budget, len(example.shots))
            round_1_messages: List[Dict] = [
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant selecting the most relevant scenes (shots).\n"
                        "Rules:\n"
                        f"- Enclose all analysis within <think></think>.\n"
                        f"- After your analysis, output exactly one line: Shots: [i, j, ...] with at most {budget_limit} unique 0-based indices within [0, {len(example.shots)-1}].\n"
                        "- Do not add any other text after the Shots line.\n"
                        "Cost model:\n"
                        f"- Each selected shot incurs 1 cost; you are rewarded more for using fewer shots while still being correct.\n"
                        f"- Aim to minimize the count (ideally 3-5) while not exceeding {budget_limit}.\n"
                    ),
                }
            ]

            user_content_round1: List[Dict] = [{"type": "text", "text": scene_prompt}]
            try:
                with _video_capture(str(example.video_path)) as cap:
                    for shot in example.shots:
                        # Each shot summary counts as a low-cost peek at the shot
                        ledger.add("peek_shot", args={"shot_id": int(shot.shot_id)})
                        rep = shot.metadata.get("representative_frame", {})
                        frame_index = int(rep.get("index", shot.representative_index))
                        # Add a small label then the image for each scene
                        user_content_round1.append(
                            {
                                "type": "text",
                                "text": f"Scene {shot.shot_id} representative frame (index {frame_index}):",
                            }
                        )
                        try:
                            data_url = _encode_frame_b64(cap, frame_index)
                            user_content_round1.append(
                                {"type": "image_url", "image_url": {"url": data_url}}
                            )
                        except Exception as _err:  # pragma: no cover - depends on video IO
                            # If frame extraction fails, keep the label so model has context
                            user_content_round1.append(
                                {
                                    "type": "text",
                                    "text": f"[Image unavailable for frame {frame_index}]",
                                }
                            )
            except Exception as exc:  # pragma: no cover - depends on video IO
                LOGGER.warning("Round 1 image embedding failed: %s", exc)

            round_1_messages.append({"role": "user", "content": user_content_round1})

            round_1_prediction = model.predict(
                scene_prompt,
                shot_count=len(example.shots),
                video_path=str(example.video_path),
                is_round_1=True,  # Flag for round 1
                messages=round_1_messages,
            )

            selected_shot_indices = round_1_prediction.get("shot_order", [])
            round_1_explanation = round_1_prediction.get("raw_completion", "")

            # Select shots based on model's choice
            valid_indices: List[int] = []
            for idx in selected_shot_indices:
                if isinstance(idx, int) and 0 <= idx < len(example.shots):
                    if idx not in valid_indices:
                        valid_indices.append(idx)
            if not valid_indices:
                valid_indices = list(range(min(len(example.shots), cfg.max_budget)))
            # Enforce the budget limit explicitly
            if len(valid_indices) > cfg.max_budget:
                valid_indices = valid_indices[: cfg.max_budget]

            selected_shots = [example.shots[i] for i in valid_indices]
            evidence_content = []

        # ROUND 2: evidence-based final answer
        shot_prompt_generator = ShotPromptGenerator(cfg.prompt_template)
        shot_prompt = ""
        if not cfg.enable_action_planning:
            shot_prompt = shot_prompt_generator.generate_prompt(
                selected_shots,
                questions=example.questions,
                fps=example.fps,
            )

        # Prepare messages for round 2, including stitching the Round 1 <think> with new evidence
        round_2_messages: List[Dict] = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant answering a question based on selected shots.\n"
                    "Rules:\n"
                    "- First, stitch and refine the prior reasoning from Round 1 with the new frames; put this reasoning within <think></think>.\n"
                    "- Then output exactly one line: Answer: ... (for MCQ, output the single letter, e.g., Answer: B).\n"
                    "- Do not add extra text after the Answer line.\n"
                ),
            },
            {"role": "user", "content": f"Here is your Round 1 output to stitch with:\n---\n{round_1_explanation}\n---"},
        ]
        user_content_round2: List[Dict] = []
        if cfg.enable_action_planning:
            user_content_round2.append({"type": "text", "text": "Here is the evidence you requested:"})
            user_content_round2.extend(evidence_content)
        else:
            user_content_round2 = [{"type": "text", "text": shot_prompt}]
            try:
                with _video_capture(str(example.video_path)) as cap:
                    per = 3
                    if selected_shots:
                        per = max(1, int(cfg.round2_max_frames_total // len(selected_shots)))
                    for shot in selected_shots:
                        frame_indices = _evenly_spaced_frames(
                            int(shot.start_frame), int(shot.end_frame), k=per
                        )
                        user_content_round2.append(
                            {"type": "text", "text": f"Shot {shot.shot_id} frames {frame_indices}:"}
                        )
                        for fi in frame_indices:
                            ledger.add("request_hd_frame", args={"shot_id": int(shot.shot_id), "frame": int(fi)})
                            try:
                                data_url = _encode_frame_b64(cap, int(fi))
                                user_content_round2.append({"type": "image_url", "image_url": {"url": data_url}})
                            except Exception as _err:
                                user_content_round2.append({"type": "text", "text": f"[Image unavailable for frame {fi}]"})
            except Exception as exc:
                LOGGER.warning("Round 2 image embedding failed: %s", exc)

        round_2_messages.append({"role": "user", "content": user_content_round2})

        metrics = {}
        if example.questions:
            question = example.questions[0]
            round_2_messages.append(
                {
                    "role": "user",
                    "content": f"Please answer the following question based on the video content: {question.question}",
                }
            )

        round_2_prediction = model.predict(
            shot_prompt,  # prompt is not the main input here, messages are
            video_path=str(example.video_path),
            messages=round_2_messages,
            is_round_1=False,  # Flag for round 2
        )

        # Account for the 'answer' submission action
        ledger.add("answer")

        if example.questions:
            metrics = evaluate_multiple_choice(
                model_response=round_2_prediction.get("final_response", ""),
                question=example.questions[0],
            )

        # Create a result object.
        result = EvaluationResult(
            video_path=str(example.video_path),
            metrics=metrics,
            raw_predictions={
                "round_1": round_1_prediction,
                "round_2": round_2_prediction,
            },
        )
        results.append(result)

        # Persist intermediate prediction for this video if requested
        if cfg.save_predictions:
            predictions_dir = cfg.resolve_predictions_dir()
            stem = Path(result.video_path).stem or "unknown"

            # Derive predicted answer and reward when we have a question
            predicted_letter = None
            predicted_option = None
            reward = None
            if example.questions:
                q = example.questions[0]
                final_text = round_2_prediction.get("final_response", "")
                # Try to extract letter first from an "Answer: X" pattern
                import re
                m = re.search(r"answer\s*[:\-]?\s*([A-Za-z])\b", final_text, flags=re.IGNORECASE)
                if m:
                    predicted_letter = m.group(1).upper()
                # Extract option text using helper
                if q.options:
                    predicted_option = _extract_choice(final_text, q.options)
                # Compute reward using whichever representation matches gold
                if q.answer is not None:
                    gold = str(q.answer).strip()
                    if len(gold) == 1 and gold.upper().isalpha():
                        if predicted_letter is not None:
                            reward = int(predicted_letter.upper() == gold.upper())
                        else:
                            # fallback: map predicted_option to its letter and compare
                            if predicted_option in q.options:
                                letter_idx = q.options.index(predicted_option)
                                letter = chr(65 + letter_idx)
                                reward = int(letter.upper() == gold.upper())
                    else:
                        # gold is option text
                        if predicted_option is not None:
                            reward = int(predicted_option == gold)
                if reward is None:
                    reward = 0

            # Attach original question metadata (first question if present)
            q_payload = None
            if example.questions:
                q0 = example.questions[0]
                q_payload = {
                    "question_id": q0.question_id,
                    "question": q0.question,
                    "options": list(q0.options or []),
                    "answer": q0.answer,
                }

            # Save predictions with question + reward; omit metrics here to avoid duplication/confusion
            to_save = {
                "video": result.video_path,
                "question": q_payload,
                "predictions": {
                    **result.raw_predictions,
                    "predicted_letter": predicted_letter,
                    "predicted_option": predicted_option,
                    "reward": reward,
                    "budget_limit": budget_limit,
                    "costs": ledger.to_dict(),
                    "oracle_cost": None,
                    "action_plan": plan_obj if cfg.enable_action_planning else None,
                    "executed_actions": executed_steps if cfg.enable_action_planning else None,
                },
            }
            write_json(to_save, predictions_dir / f"{stem}_predictions.json")

            # Also attach these fields to in-memory result for metric aggregation
            result.raw_predictions = {
                **result.raw_predictions,
                "predicted_letter": predicted_letter,
                "predicted_option": predicted_option,
                "reward": reward,
                "costs": ledger.to_dict(),
                "oracle_cost": None,
            }

    return results


def aggregate_metrics(results: Sequence[EvaluationResult]) -> Dict[str, float]:
    """Aggregate metric values across evaluation results."""

    aggregated: Dict[str, List[float]] = {}
    for result in results:
        for metric in result.metrics.values():
            for key, value in metric.values.items():
                aggregated.setdefault(f"{metric.name}/{key}", []).append(float(value))

    return {
        name: (sum(values) / len(values)) if values else 0.0
        for name, values in aggregated.items()
    }


def run_evaluation(cfg: EvalConfig, shots_root: Path | None = None) -> Dict[str, float]:
    """Execute the two-stage evaluation pipeline."""

    configure_logging(cfg.verbose)

    dataset = load_video_data(cfg, shots_root=shots_root)
    if not len(dataset):
        LOGGER.warning("No videos found for evaluation under %s", shots_root or cfg.shots_root)
        return {}

    # If saving predictions, skip videos that already have saved predictions
    if cfg.save_predictions:
        pred_dir = cfg.resolve_predictions_dir()
        existing = {
            p.name.replace("_predictions.json", "")
            for p in pred_dir.glob("*_predictions.json")
        }
        filtered_examples = []
        for example in dataset:
            stem = example.video_path.stem
            if stem not in existing:
                filtered_examples.append(example)
        if len(filtered_examples) != len(list(dataset)):
            LOGGER.info(
                "Skipping %d already-predicted videos.",
                len(list(dataset)) - len(filtered_examples),
            )
        dataset = EvaluationDataset(filtered_examples)

    model = load_model(cfg)
    results = run_single_video(model, dataset, cfg)

    # Per-video predictions are saved during processing; no need to duplicate here

    # Build cost-aware samples from saved predictions (for metrics)
    samples = []
    for r in results:
        raw = r.raw_predictions
        round2 = raw.get("round_2", {}) or {}
        pred_letter = raw.get("predicted_letter")
        pred_option = raw.get("predicted_option")
        reward = raw.get("reward")
        costs = raw.get("costs") or {}
        oracle_cost = raw.get("oracle_cost")
        samples.append({
            "reward": reward if reward is not None else 0,
            "costs": costs,
            "oracle_cost": oracle_cost,
        })

    # Aggregate standard and cost-based metrics
    summary = aggregate_metrics(results)
    # K@B and C@A for token
    summary.update(k_at_b(samples, cfg.budgets_token, unit="token"))
    summary.update(c_at_a(samples, cfg.acc_targets, unit="token"))
    # K@B and C@A for latency
    summary.update(k_at_b(samples, cfg.budgets_token, unit="latency"))
    summary.update(c_at_a(samples, cfg.acc_targets, unit="latency"))
    # Oracle regret (if available)
    summary.update(oracle_regret(samples, unit="token"))
    summary.update(oracle_regret(samples, unit="latency"))
    summary_path = cfg.resolve_cache_dir() / "metrics_summary.json"
    write_json(summary, summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the two-stage video reasoning evaluation pipeline."
    )
    parser.add_argument("--shots-root", type=str, help="Directory containing shots.json files.")
    parser.add_argument("--model-name", default="gemini-2.5-flash",type=str, help="Model identifier for logging.")
    parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="Disable saving per-video prediction files.",
    )
    parser.add_argument("--quiet", action="store_true", help="Silence informative logging output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_dict = dict(DEFAULT_EVAL_CONFIG.__dict__)

    if args.shots_root:
        cfg_dict["shots_root"] = Path(args.shots_root).expanduser()
    if args.model_name:
        cfg_dict["model_name"] = args.model_name
    if args.no_save_predictions:
        cfg_dict["save_predictions"] = False
    if args.quiet:
        cfg_dict["verbose"] = False

    cfg = EvalConfig(**cfg_dict)

    summary = run_evaluation(cfg, shots_root=cfg.shots_root)
    if summary:
        print("Evaluation summary:")
        for key, value in summary.items():
            print(f" - {key}: {value:.4f}")
    else:
        print("No results produced.")


if __name__ == "__main__":
    main()
