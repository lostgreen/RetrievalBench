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
    "peek_clip": {"args": {"shot_id": int, "clip_index": int}},
    "request_hd_frame": {"args": {"shot_id": int, "frame": int}},
    "request_clip_1s": {"args": {"shot_id": int, "t": float}},
    "request_hd_crop": {"args": {"shot_id": int, "frame": int, "bbox": list}},  # bbox=[x1,y1,x2,y2] normalized
    "zoom_hd": {"args": {"shot_id": int, "frame": int, "intent": str}},  # intent in words
}


def _action_schema_str(cfg: EvalConfig) -> str:
    price = cfg.cost_table_token
    return (
        "Allowed actions and token-equivalent costs:\n"
        f"- peek_scene(scene_id) : {price.get('peek_scene', 1.0)}\n"
        f"- peek_shot(shot_id)  : {price.get('peek_shot', 0.5)}\n"
        f"- peek_clip(shot_id, clip_index) : {price.get('peek_clip', 1.0)}\n"
        f"- request_hd_frame(shot_id, frame) : {price.get('request_hd_frame', 5.0)}\n"
        f"- request_clip_1s(shot_id, t)      : {price.get('request_clip_1s', 15.0)}\n"
        f"- request_hd_crop(shot_id, frame, bbox=[x1,y1,x2,y2]) : {price.get('request_hd_frame', 5.0)} (same as hd_frame)\n"
        f"- zoom_hd(shot_id, frame, intent=words) : {price.get('zoom_hd', 4.0)} (describe region in words; system maps to ROI)\n"
        "Note: bbox is normalized coordinates in [0,1]."
    )


def build_planning_system_prompt(cfg: EvalConfig, budget_token: float, shot_count: int) -> str:
    return (
        "You are an information-efficient planner that selects actions under a budget.\n"
        "Rules:\n"
        "- Think inside <think>...</think>.\n"
        "- Then output a minimal plain-text plan (no extra prose) using this format:\n"
        "  Plan: <one-line summary>\n"
        "  Budget: <number>\n"
        "  Steps:\n"
        "  1) <act> [arg=value ...]\n"
        "  2) <act> [arg=value ...]\n"
        "  ...\n"
        f"- Budget (token-equivalent) <= {budget_token}. Minimize cost while enabling a correct answer.\n"
        f"- Prefer 3-5 peeks before requesting any HD content. Shot indices are 0..{shot_count-1}.\n"
        f"- Do NOT include the final answer; you will answer after receiving evidence.\n\n"
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
    # 1) Try strict/fenced JSON if the model already outputs JSON
    block = extract_json_block(text)
    if block is not None:
        try:
            data = json.loads(block)
        except Exception:
            data = None
        if isinstance(data, dict):
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

    # 2) Parse minimal plain-text plan with regex
    summary = ""
    budget = 0.0
    m_budget = re.search(r"^\s*budget\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\b", text, flags=re.IGNORECASE | re.MULTILINE)
    if m_budget:
        try:
            budget = float(m_budget.group(1))
        except Exception:
            budget = 0.0
    m_plan = re.search(r"^\s*plan\s*[:=]\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if m_plan:
        summary = m_plan.group(1).strip()

    steps: List[Dict[str, Any]] = []
    actions = list(ALLOWED_ACTIONS.keys())
    # Build a combined regex to detect an action token per line
    act_pat = re.compile(r"^(?:\s*\d+[\).\-]\s*)?\s*(" + "|".join(map(re.escape, actions)) + r")\b(.*)$", re.IGNORECASE)
    lines = text.splitlines()
    for line in lines:
        m = act_pat.match(line.strip())
        if not m:
            continue
        act = m.group(1)
        rest = m.group(2) or ""
        act = act.strip()
        spec = ALLOWED_ACTIONS.get(act)
        if not spec:
            continue
        args_out: Dict[str, Any] = {}
        # Prefer named args: key=value or key: value
        for arg_name, arg_type in spec["args"].items():
            # bbox special-case
            if arg_name == "bbox":
                m_bb = re.search(r"bbox\s*[:=]\s*\[([^\]]+)\]", rest, flags=re.IGNORECASE)
                if m_bb:
                    nums = re.findall(r"-?[0-9]+(?:\.[0-9]+)?", m_bb.group(1))
                    if len(nums) >= 4:
                        try:
                            args_out["bbox"] = [float(nums[0]), float(nums[1]), float(nums[2]), float(nums[3])]
                        except Exception:
                            pass
                continue
            # intent (string) special-case: accept quoted or after keyword to EOL
            if arg_type is str:
                m_str = re.search(rf"\b{re.escape(arg_name)}\s*[:=]\s*\"([^\"]+)\"", rest, flags=re.IGNORECASE)
                if not m_str:
                    m_str = re.search(rf"\b{re.escape(arg_name)}\s*[:=]\s*'([^']+)'", rest, flags=re.IGNORECASE)
                if not m_str:
                    m_str = re.search(rf"\b{re.escape(arg_name)}\s*[:=]\s*(.+)$", rest, flags=re.IGNORECASE)
                if m_str:
                    args_out[arg_name] = m_str.group(1).strip()
                continue
            m_named = re.search(rf"\b{re.escape(arg_name)}\s*[:=]\s*(-?[0-9]+(?:\.[0-9]+)?)\b", rest, flags=re.IGNORECASE)
            if m_named:
                val_txt = m_named.group(1)
                try:
                    if arg_type is int:
                        args_out[arg_name] = int(float(val_txt))
                    elif arg_type is float:
                        args_out[arg_name] = float(val_txt)
                    else:
                        args_out[arg_name] = val_txt
                except Exception:
                    pass
        # If some required args missing, collect bare numbers as positional hints
        missing_order = [k for k in spec["args"].keys() if k not in args_out and k != "bbox"]
        if missing_order:
            nums = re.findall(r"-?[0-9]+(?:\.[0-9]+)?", rest)
            # Remove ones already consumed by named matches (best-effort)
            # Use first K numbers to fill missing args in order
            idx = 0
            for k in missing_order:
                if idx >= len(nums):
                    break
                try:
                    if spec["args"][k] is int:
                        args_out[k] = int(float(nums[idx]))
                    elif spec["args"][k] is float:
                        args_out[k] = float(nums[idx])
                    else:
                        args_out[k] = nums[idx]
                    idx += 1
                except Exception:
                    idx += 1
                    continue
        steps.append({"act": act, "args": args_out})

    return {"plan": summary, "budget": budget, "steps": steps}


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
                elif arg_type is str:
                    val = str(val)
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
