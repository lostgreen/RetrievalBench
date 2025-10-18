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
from evaluate.utils import LOGGER, configure_logging, write_json, read_json
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
from evaluate.metrics import compute_k_at_b as _compute_k_at_b, summarize_metrics as _summarize_metrics, MetricResult as _MetricResult


## Model adapter and media helpers moved to dedicated modules for clarity


def _candidate_video_ids(ex) -> List[str]:
    cands: List[str] = []
    try:
        vid = ex.shots_json.get("video_name")
        if vid:
            cands.append(str(vid))
    except Exception:
        pass
    try:
        cands.append(ex.video_path.stem)
    except Exception:
        pass
    try:
        cands.append(ex.shots_path.stem)
        cands.append(ex.shots_path.parent.name)
    except Exception:
        pass
    out: List[str] = []
    for s in cands:
        if s and s not in out:
            out.append(s)
    return out


def _find_gt_path_for_example(ex, gt_dir: Path) -> Path | None:
    for vid in _candidate_video_ids(ex):
        p = (gt_dir / f"{vid}.json").expanduser()
        if p.exists():
            return p
    return None


def _selection_budgets(cfg: EvalConfig) -> List[int]:
    # Build integer budgets from 1..max_budget with ~budget_steps steps, always include 1 and max_budget
    budgets: List[int] = []
    if cfg.max_budget <= 0:
        return budgets
    step = max(1, int(round(cfg.max_budget / max(int(cfg.budget_steps or 1), 1))))
    budgets = list(range(step, int(cfg.max_budget) + 1, step))
    if 1 not in budgets:
        budgets = [1] + budgets
    if budgets[-1] != int(cfg.max_budget):
        budgets.append(int(cfg.max_budget))
    # Deduplicate and sort
    budgets = sorted(set(int(b) for b in budgets))
    return budgets


def run_single_video(
    model: ModelInterface,
    dataset: EvaluationDataset,
    cfg: EvalConfig,
) -> List[EvaluationResult]:
    results: List[EvaluationResult] = []
    for example in dataset:
        # Load GT (if available) for this example
        gt_selected: List[int] | None = None
        gt_path = None
        try:
            if cfg.gt_dir:
                gt_dir = cfg.gt_dir.expanduser()
                gt_path = _find_gt_path_for_example(example, gt_dir)
                if gt_path and gt_path.exists():
                    try:
                        gt_json = read_json(gt_path)
                        raw = gt_json.get("selected_shots") or []
                        if isinstance(raw, list):
                            gt_selected = [int(x) for x in raw if isinstance(x, int) or (isinstance(x, str) and str(x).isdigit())]
                    except Exception:
                        gt_selected = None
        except Exception:
            gt_selected = None

        # Initialize cost ledger for this example
        table = CostTable(token=cfg.cost_table_token, latency=cfg.cost_table_latency)
        ledger = CostLedger(table)
        # ROUND 1: Scene-level selection (Shots)
        # Precompute budget limit
        budget_limit = min(cfg.max_budget, len(example.shots))
        # Build scene prompt and Round 1 messages
        scene_prompt_generator = ScenePromptGenerator(cfg.prompt_template)
        scene_prompt = scene_prompt_generator.generate_prompt(
            example.shots,
            questions=example.questions,
            fps=example.fps,
        )

        # Build multimodal messages for Round 1: include representative frame per shot
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
                    f"- Aim to minimize the count (ideally 3-5) while not exceeding {budget_limit}.\n\n"
                    "Examples (format only):\n"
                    "<think> Consider which scenes contain the requested object. </think>\n"
                    "Shots: [2, 5, 7]\n\n"
                    "<think> Focus on the driving scenes, avoid duplicates. </think>\n"
                    "Shots: [1, 3]\n"
                ),
            }
        ]

        user_content_round1: List[Dict] = [{"type": "text", "text": scene_prompt}]
        # Charge an optional fixed overview fee once (default 0)
        try:
            ov_tok = float(cfg.cost_table_token.get("overview", 0.0))
            ov_lat = float(cfg.cost_table_latency.get("overview", 0.0))
        except Exception:
            ov_tok, ov_lat = 0.0, 0.0
        if (ov_tok > 0.0) or (ov_lat > 0.0):
            ledger.add("overview", note="round1_overview")
        try:
            with _video_capture(str(example.video_path)) as cap:
                for shot in example.shots:
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

        # ROUND 2: evidence-based final answer (with optional micro-actions within selected shots)
        shot_prompt_generator = ShotPromptGenerator(cfg.prompt_template)
        shot_prompt = shot_prompt_generator.generate_prompt(
            selected_shots,
            questions=example.questions,
            fps=example.fps,
        )

        # Round 2A: within selected shots, request targeted evidence via lightweight actions
        round2_plan_obj = None
        executed_steps = None
        evidence_content: List[Dict] = []
        round_2a_prediction = None
        try:
            # Build a simple planning prompt restricted to selected shots
            budget_token = float(cfg.round1_budget_token)
            sys_prompt = build_planning_system_prompt(cfg, budget_token, shot_count=len(example.shots))
            # Emphasize restriction to selected shots only
            sys_prompt += ("\nConstraints:\n"
                            f"- Use ONLY shot_id from this set: {[int(s.shot_id) for s in selected_shots]}.\n"
                            f"- Long shots (> {cfg.long_shot_sec}s) are split into {cfg.clip_win_sec}s clips with indices 0..K-1.\n"
                            f"- You may use peek_clip(shot_id, clip_index) up to {cfg.max_clips_per_shot} per shot.\n"
                            f"- You may use at most {cfg.max_zooms} zoom_hd(shot_id, frame, intent=words).\n"
                            "- Prefer peeking clips before requesting any HD frame/crop; keep the plan short.")

            fps = example.fps or 30.0
            shot_lines = []
            for s in selected_shots:
                shot_lines.append(
                    f"shot_id={s.shot_id}: frames[{s.start_frame}-{s.end_frame}] time[{s.start_frame/fps:.2f}-{s.end_frame/fps:.2f}]s"
                )
            # Build clip listings for long shots
            clip_lines = []
            for s in selected_shots:
                t0 = s.start_frame / fps
                t1 = s.end_frame / fps
                dur = max(0.0, t1 - t0)
                if dur >= float(cfg.long_shot_sec):
                    # list first few clip indices and their time ranges
                    win = float(cfg.clip_win_sec)
                    count = int(max(1, int(dur // win) + (1 if (dur % win) > 1e-6 else 0)))
                    preview = min(6, count)
                    spans = []
                    for cidx in range(preview):
                        st = t0 + cidx * win
                        en = min(t1, st + win)
                        spans.append(f"clip[{cidx}]: {st:.2f}-{en:.2f}s")
                    clip_lines.append(f"shot {s.shot_id} clips: " + "; ".join(spans) + (" ..." if count > preview else ""))

            user_intro = ("Selected shots (use shot_id when planning):\n" + "\n".join(shot_lines))
            if clip_lines:
                user_intro += "\n\nLong-shot temporal clips (indices):\n" + "\n".join(clip_lines)
            if example.questions:
                q0 = example.questions[0]
                user_intro += f"\nQuestion: {q0.question}\nOptions: " + "; ".join(q0.options or [])

            round_2a_messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_intro},
            ]

            # Ask the model for a minimal plain-text plan of micro-actions
            round_2a_prediction = model.predict(
                "round2_plan_actions",
                shot_count=len(example.shots),
                video_path=str(example.video_path),
                is_round_1=False,
                messages=round_2a_messages,
            )
            raw_plan = round_2a_prediction.get("raw_completion") or round_2a_prediction.get("final_response", "")
            plan_obj = parse_action_plan(raw_plan) or {"steps": []}
            steps = validate_actions(plan_obj, shot_count=len(example.shots))
            # Enforce budget by truncating steps if needed
            remaining = float(cfg.round1_budget_token)
            bounded_steps: List[Dict] = []
            # enforce per-shot clip limits and global zoom limit
            clip_counts: Dict[int, int] = {}
            zooms_used = 0
            for st in steps:
                act = st.get("act")
                unit_cost = float(cfg.cost_table_token.get(str(act), 0.0))
                if unit_cost > remaining:
                    break
                # Restrict shot_id to selected set
                sid = st.get("args", {}).get("shot_id")
                if sid is not None and int(sid) not in [int(s.shot_id) for s in selected_shots]:
                    continue
                # enforce limits
                if act == "peek_clip":
                    sid_i = int(sid) if sid is not None else -1
                    cnt = clip_counts.get(sid_i, 0)
                    if cnt >= int(cfg.max_clips_per_shot):
                        continue
                    clip_counts[sid_i] = cnt + 1
                if act == "zoom_hd":
                    if zooms_used >= int(cfg.max_zooms):
                        continue
                    zooms_used += 1
                bounded_steps.append(st)
                remaining -= unit_cost
            steps = bounded_steps

            if steps:
                evidence_content, executed_steps = execute_actions(example, steps, ledger, clip_win_sec=cfg.clip_win_sec)
                round2_plan_obj = plan_obj
        except Exception as _err:  # safety net
            evidence_content = []

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
        if evidence_content:
            user_content_round2.append({"type": "text", "text": "Here is the targeted evidence you requested:"})
            user_content_round2.extend(evidence_content)
        else:
            # Fallback to evenly-sampled frames within selected shots
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
                            # Fallback frames are part of free/default evidence; do not charge cost here
                            try:
                                data_url = _encode_frame_b64(cap, int(fi))
                                user_content_round2.append({"type": "image_url", "image_url": {"url": data_url}})
                            except Exception as _err:
                                user_content_round2.append({"type": "text", "text": f"[Image unavailable for frame {fi}]"})
            except Exception as exc:
                LOGGER.warning("Round 2 image embedding failed: %s", exc)

        round_2_messages.append({"role": "user", "content": user_content_round2})

        # Optional: evaluate selection vs GT (if available) using K@B + AUIC
        selection_metric: _MetricResult | None = None
        if gt_selected is not None:
            try:
                budgets = _selection_budgets(cfg)
                curve = _compute_k_at_b(valid_indices, gt_selected, budgets)
                curve = _summarize_metrics(curve)
                selection_metric = _MetricResult(name="shot_selection_k_at_b", values=dict(curve.values), curve=curve.curve)
            except Exception as _err:
                selection_metric = None

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
        if selection_metric is not None:
            metrics[selection_metric.name] = selection_metric

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
                "version": "0.3.0",
                "video": result.video_path,
                "question": q_payload,
                "budget": {
                    "cost_table": {
                        "peek_shot": cfg.cost_table_token.get("peek_shot", 0.5),
                        "peek_clip": cfg.cost_table_token.get("peek_clip", 1.0),
                        "request_hd_frame": cfg.cost_table_token.get("request_hd_frame", 5.0),
                        "request_clip_1s": cfg.cost_table_token.get("request_clip_1s", 15.0),
                        "zoom_hd": cfg.cost_table_token.get("zoom_hd", 4.0),
                        "answer": cfg.cost_table_token.get("answer", 0.5),
                    },
                    "limits": {
                        "max_shots": int(cfg.max_budget),
                        "max_clips_per_shot": int(cfg.max_clips_per_shot),
                        "max_zooms": int(cfg.max_zooms),
                        "clip_win_sec": float(cfg.clip_win_sec),
                        "long_shot_sec": float(cfg.long_shot_sec),
                    },
                },
                "predictions": {
                    **result.raw_predictions,
                    "predicted_letter": predicted_letter,
                    "predicted_option": predicted_option,
                    "reward": reward,
                    "budget_limit": budget_limit,
                    "costs": ledger.to_dict(),
                    "oracle_cost": None,
                    "action_plan": round2_plan_obj,
                    "executed_actions": executed_steps,
                    "gt_selected_shots": gt_selected,
                    "raw_model_outputs": {
                        "stageA": round_1_prediction,
                        "stageB": round_2a_prediction,
                        "stageC": round_2_prediction,
                    },
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
                "action_plan": round2_plan_obj,
                "executed_actions": executed_steps,
                "raw_model_outputs": {
                    "stageA": round_1_prediction,
                    "stageB": round_2a_prediction,
                    "stageC": round_2_prediction,
                },
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

    # If GT directory is present, filter to only videos that have GT
    try:
        gt_dir = cfg.gt_dir.expanduser() if cfg.gt_dir else None
    except Exception:
        gt_dir = None
    if gt_dir and gt_dir.exists():
        examples_with_gt = []
        missing = 0
        for ex in dataset:
            p = _find_gt_path_for_example(ex, gt_dir)
            if p is not None:
                examples_with_gt.append(ex)
            else:
                missing += 1
        if examples_with_gt:
            if len(examples_with_gt) != len(list(dataset)):
                LOGGER.info("Filtering to %d videos with GT (skipping %d without GT)", len(examples_with_gt), missing)
            dataset = EvaluationDataset(examples_with_gt)
        else:
            LOGGER.info("GT directory %s has no matching files for current dataset; evaluating all videos.", gt_dir)

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
    # Budget spent (means)
    def _mean(lst: List[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0
    summary["BudgetSpent/token_mean"] = _mean([
        float((s.get("costs", {}) or {}).get("total_token", 0.0)) for s in samples
    ])
    summary["BudgetSpent/latency_mean"] = _mean([
        float((s.get("costs", {}) or {}).get("total_latency", 0.0)) for s in samples
    ])
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
        "--round1-budget-token",
        type=float,
        default=None,
        help="Token budget for Round-2 micro-actions (planning within selected shots).",
    )
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
    if args.round1_budget_token is not None:
        cfg_dict["round1_budget_token"] = float(args.round1_budget_token)
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
