"""Execute planned actions to gather evidence content for Round 2."""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

from evaluate.data_loader import VideoExample
from evaluate.media import (
    video_capture,
    encode_frame_b64,
    encode_crop_b64,
    evenly_spaced_frames,
)
from evaluate.utils import LOGGER
from evaluate.costs import CostLedger


def _time_to_frame(t: float, fps: float) -> int:
    if fps and fps > 0:
        return max(0, int(round(t * fps)))
    return int(max(0.0, t))


def _shot_summary_text(example: VideoExample, shot_id: int) -> str:
    shots = example.shots
    if not (0 <= shot_id < len(shots)):
        return f"Shot {shot_id}: out of range"
    shot = shots[shot_id]
    fps = example.fps or 30.0
    t0 = shot.start_frame / fps
    t1 = shot.end_frame / fps
    return (
        f"Shot {shot_id}: frames [{shot.start_frame}-{shot.end_frame}]"
        f" time [{t0:.2f}s - {t1:.2f}s]"
    )


def execute_actions(
    example: VideoExample,
    steps: Sequence[Mapping[str, object]],
    ledger: CostLedger,
    *,
    thumbnail_height: int = 160,
    clip_frames: int = 6,
) -> Tuple[List[Dict], List[Mapping[str, object]]]:
    """Execute actions and return (content_blocks, executed_steps).

    - content_blocks: a list of message content items for OpenAI multimodal [{'type':'text'|'image_url',...}]
    - executed_steps: sanitized steps possibly with extra info (e.g., resolved frames)
    """
    content: List[Dict] = []
    executed: List[Mapping[str, object]] = []

    with video_capture(str(example.video_path)) as cap:
        for step in steps:
            act = step.get("act")
            args = dict(step.get("args") or {})

            if act == "peek_scene":
                # placeholder: scene==shot here
                sid = int(args["scene_id"]) if "scene_id" in args else -1
                content.append({"type": "text", "text": f"[peek_scene] scene={sid} (mapped to shot {sid})"})
                ledger.add("peek_scene", args={"scene_id": sid})
                executed.append({"act": act, "args": {"scene_id": sid}})

            elif act == "peek_shot":
                sid = int(args["shot_id"]) if "shot_id" in args else -1
                content.append({"type": "text", "text": f"[peek_shot] { _shot_summary_text(example, sid) }"})
                # Use representative frame as a thumbnail
                shot = example.shots[sid]
                rep = shot.metadata.get("representative_frame", {})
                frame_index = int(rep.get("index", shot.representative_index))
                # For simplicity, reuse HD encoder; small thumbnails could be added later
                try:
                    data_url = encode_frame_b64(cap, frame_index)
                    content.append({"type": "image_url", "image_url": {"url": data_url}})
                except Exception as exc:
                    LOGGER.warning("peek_shot thumbnail failed: %s", exc)
                ledger.add("peek_shot", args={"shot_id": sid})
                executed.append({"act": act, "args": {"shot_id": sid, "frame": frame_index}})

            elif act == "request_hd_frame":
                sid = int(args["shot_id"]) if "shot_id" in args else -1
                frame = int(args["frame"]) if "frame" in args else 0
                # Clamp frame inside shot
                shot = example.shots[sid]
                frame = max(shot.start_frame, min(shot.end_frame, frame))
                try:
                    data_url = encode_frame_b64(cap, frame)
                    content.append({"type": "text", "text": f"[hd_frame] shot={sid} frame={frame}"})
                    content.append({"type": "image_url", "image_url": {"url": data_url}})
                except Exception as exc:
                    LOGGER.warning("request_hd_frame failed: %s", exc)
                ledger.add("request_hd_frame", args={"shot_id": sid, "frame": frame})
                executed.append({"act": act, "args": {"shot_id": sid, "frame": frame}})

            elif act == "request_clip_1s":
                sid = int(args["shot_id"]) if "shot_id" in args else -1
                t = float(args["t"]) if "t" in args else 0.0
                fps = example.fps or 30.0
                shot = example.shots[sid]
                f0 = _time_to_frame(t, fps)
                f1 = _time_to_frame(t + 1.0, fps)
                # Clamp to shot bounds
                f0 = max(shot.start_frame, min(shot.end_frame, f0))
                f1 = max(shot.start_frame, min(shot.end_frame, f1))
                if f1 < f0:
                    f0, f1 = f1, f0
                frames = evenly_spaced_frames(f0, f1, clip_frames)
                content.append({"type": "text", "text": f"[clip_1s] shot={sid} t={t:.2f}s frames={frames}"})
                for fi in frames:
                    try:
                        data_url = encode_frame_b64(cap, fi)
                        content.append({"type": "image_url", "image_url": {"url": data_url}})
                    except Exception as exc:
                        LOGGER.warning("request_clip_1s frame failed: %s", exc)
                ledger.add("request_clip_1s", args={"shot_id": sid, "t": float(t)}, units=1.0)
                executed.append({"act": act, "args": {"shot_id": sid, "t": float(t), "frames": frames}})

            elif act == "request_hd_crop":
                sid = int(args["shot_id"]) if "shot_id" in args else -1
                frame = int(args["frame"]) if "frame" in args else 0
                bbox = list(args["bbox"]) if "bbox" in args else [0.25, 0.25, 0.75, 0.75]
                # For now, emulate crop by returning the full frame and reporting bbox as text
                shot = example.shots[sid]
                frame = max(shot.start_frame, min(shot.end_frame, frame))
                try:
                    data_url = encode_crop_b64(cap, frame, bbox)
                    content.append({"type": "text", "text": f"[hd_crop] shot={sid} frame={frame} bbox={bbox}"})
                    content.append({"type": "image_url", "image_url": {"url": data_url}})
                except Exception as exc:
                    LOGGER.warning("request_hd_crop failed: %s", exc)
                ledger.add("request_hd_frame", args={"shot_id": sid, "frame": frame, "bbox": bbox})
                executed.append({"act": act, "args": {"shot_id": sid, "frame": frame, "bbox": bbox}})

            else:
                LOGGER.debug("Unknown action ignored: %s", act)
                continue

    return content, executed
