"""Action planning prompt, parsing, and validation for AIF-V pipeline."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from evaluate.config import EvalConfig


ALLOWED_ACTIONS = {
    "peek_scene": {"args": {"scene_id": int}},
    "peek_shot": {"args": {"shot_id": int}},
    "request_hd_frame": {"args": {"shot_id": int, "frame": int}},
    "request_clip_1s": {"args": {"shot_id": int, "t": float}},
    "request_hd_crop": {"args": {"shot_id": int, "frame": int, "bbox": list}},  # bbox=[x1,y1,x2,y2] normalized
}


def _action_schema_str(cfg: EvalConfig) -> str:
    price = cfg.cost_table_token
    return (
        "Allowed actions and token-equivalent costs:\n"
        f"- peek_scene(scene_id) : {price.get('peek_scene', 1.0)}\n"
        f"- peek_shot(shot_id)  : {price.get('peek_shot', 0.5)}\n"
        f"- request_hd_frame(shot_id, frame) : {price.get('request_hd_frame', 5.0)}\n"
        f"- request_clip_1s(shot_id, t)      : {price.get('request_clip_1s', 15.0)}\n"
        f"- request_hd_crop(shot_id, frame, bbox=[x1,y1,x2,y2]) : {price.get('request_hd_frame', 5.0)} (same as hd_frame)\n"
        "Note: bbox is normalized coordinates in [0,1]."
    )


def build_planning_system_prompt(cfg: EvalConfig, budget_token: float, shot_count: int) -> str:
    return (
        "You are an information-efficient planner that selects actions under a budget.\n"
        "Rules:\n"
        "- Think inside <think>...</think>.\n"
        "- Then output ONLY a JSON object (no prose).\n"
        "- The JSON schema is: {\n"
        "    \"plan\": string,\n"
        "    \"budget\": number,\n"
        "    \"steps\": [ { \"act\": string, \"args\": object, \"note\": optional string } ]\n"
        "  }\n"
        f"- Budget (token-equivalent) <= {budget_token}. Minimize cost while enabling a correct answer.\n"
        f"- Prefer 3-5 peeks before requesting any HD content. Shot indices are 0..{shot_count-1}.\n"
        f"- Do NOT include an answer in this JSON; you will answer after receiving evidence.\n\n"
        + _action_schema_str(cfg)
    )


def extract_json_block(text: str) -> str | None:
    """Extract a JSON object block from text, preferring fenced code blocks."""
    # Prefer fenced ```json ... ```
    fence = re.findall(r"```json\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fence:
        return fence[-1].strip()
    # Fallback: find the last top-level {...} block
    start_indices = [m.start() for m in re.finditer(r"\{", text)]
    for start in reversed(start_indices):
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    return candidate
    return None


def parse_action_plan(text: str) -> Dict[str, Any] | None:
    block = extract_json_block(text) or text
    try:
        data = json.loads(block)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    # normalize
    plan = {
        "plan": str(data.get("plan", "")),
        "budget": float(data.get("budget", 0.0)),
        "steps": [],
    }
    steps = data.get("steps") or []
    if isinstance(steps, list):
        for s in steps:
            if not isinstance(s, dict):
                continue
            act = str(s.get("act", "")).strip()
            args = s.get("args") or {}
            note = s.get("note")
            plan["steps"].append({"act": act, "args": args, "note": note})
    return plan


def validate_actions(plan: Mapping[str, Any], *, shot_count: int) -> List[Dict[str, Any]]:
    """Validate and sanitize a list of actions according to ALLOWED_ACTIONS."""
    valid: List[Dict[str, Any]] = []
    for step in plan.get("steps", []):
        if not isinstance(step, Mapping):
            continue
        act = str(step.get("act", "")).strip()
        spec = ALLOWED_ACTIONS.get(act)
        if not spec:
            continue
        args_in = step.get("args") or {}
        if not isinstance(args_in, Mapping):
            continue
        args_out: Dict[str, Any] = {}
        ok = True
        for arg_name, arg_type in spec["args"].items():
            if arg_name not in args_in:
                ok = False
                break
            try:
                val = args_in[arg_name]
                # Convert types
                if arg_type is int:
                    val = int(val)
                elif arg_type is float:
                    val = float(val)
                elif arg_name == "bbox":
                    # expect list of 4 numbers
                    bb = list(val)
                    if len(bb) != 4:
                        ok = False
                        break
                    val = [float(x) for x in bb]
                args_out[arg_name] = val
            except Exception:
                ok = False
                break
        if not ok:
            continue
        # basic bounds
        if "shot_id" in args_out and not (0 <= int(args_out["shot_id"]) < shot_count):
            continue
        if "frame" in args_out and int(args_out["frame"]) < 0:
            continue
        if "bbox" in args_out:
            x1, y1, x2, y2 = args_out["bbox"]
            if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
                continue
        valid.append({"act": act, "args": args_out, "note": step.get("note")})
    return valid

