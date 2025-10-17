from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from flask import Flask, Response, jsonify, redirect, render_template, request, send_file, url_for

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:
    cv2 = None  # type: ignore
    np = None  # type: ignore

from evaluate.utils import read_json


app = Flask(__name__, static_folder="static", template_folder="templates")


@dataclass
class ServerConfig:
    shots_root: Path
    question_file: Optional[Path]
    gt_dir: Path
    min_shots: int = 0
    max_shots: int = 1_000_000
    stage2_max_frames: int = 128


CFG = ServerConfig(
    shots_root=Path("output/videomme_batch").resolve(),
    question_file=None,
    gt_dir=Path("annotation/gt").resolve(),
)


def _list_shots_files(root: Path) -> List[Path]:
    if root.is_file() and root.name == "shots.json":
        return [root]
    return sorted(root.rglob("shots.json"))


def _resolve_video_path(raw: str, shots_file: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    cands = [shots_file.parent / p]
    for c in cands:
        if c.exists():
            return c.resolve()
    return (shots_file.parent / p).resolve()


def _load_questions(path: Optional[Path]) -> Dict[str, List[Mapping[str, object]]]:
    if not path:
        return {}
    data: Dict[str, List[Mapping[str, object]]] = {}
    try:
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                vid = obj.get("videoID") or obj.get("video_id") or obj.get("video_name")
                if not vid:
                    continue
                data.setdefault(str(vid), []).append(obj)
    except Exception:
        return {}
    return data


def _read_frame_bytes(video_path: Path, frame_idx: int) -> Optional[bytes]:
    if cv2 is None:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            return None
        return buf.tobytes()
    finally:
        cap.release()


@app.route("/")
def index() -> Response:
    return redirect(url_for("videos"))


@app.route("/videos")
def videos() -> Response:
    files = _list_shots_files(CFG.shots_root)
    items = []
    for f in files:
        try:
            j = read_json(f)
        except Exception:
            continue
        vid = j.get("video_name") or Path(j.get("video", "")).stem or f.parent.name
        nshots = len(j.get("shots", []) or [])
        if CFG.min_shots <= nshots <= CFG.max_shots:
            items.append({"video_id": str(vid), "shots_path": str(f), "shots_count": nshots})
    return render_template("videos.html", items=items)


@app.route("/annotate/<video_id>")
def annotate(video_id: str) -> Response:
    # resolve shots.json by folder match
    shots_file = None
    for f in _list_shots_files(CFG.shots_root):
        if f.parent.name == video_id or f.stem == video_id:
            shots_file = f
            break
    if shots_file is None:
        return Response(f"shots.json for {video_id} not found", status=404)
    data = read_json(shots_file)
    video_path = _resolve_video_path(data.get("video", ""), shots_file)
    fps = float(data.get("fps", 30.0))
    shots = data.get("shots", [])
    # Pick first question if available
    qmap = _load_questions(CFG.question_file)
    qlist = qmap.get(video_id, [])
    question = None
    if qlist:
        q = qlist[0]
        question = {
            "question_id": q.get("question_id"),
            "question": q.get("question"),
            "options": q.get("options", []),
            "answer": q.get("answer"),
        }
    # Prepare minimal shot view: id, start, end, rep_index
    view = []
    for idx, s in enumerate(shots):
        rep = (s.get("representative_frame") or {}).get("index", s.get("start_frame", 0))
        view.append(
            {
                "shot_id": idx,
                "start": int(s.get("start_frame", 0)),
                "end": int(s.get("end_frame", 0)),
                "rep_index": int(rep),
            }
        )
    return render_template(
        "annotate.html",
        video_id=video_id,
        video_path=str(video_path),
        fps=fps,
        shots=view,
        question=question,
        stage2_max_frames=int(CFG.stage2_max_frames),
    )


@app.route("/image/<video_id>/<int:frame>")
def image(video_id: str, frame: int) -> Response:
    # resolve video path and return jpeg
    shots_file = None
    for f in _list_shots_files(CFG.shots_root):
        if f.parent.name == video_id or f.stem == video_id:
            shots_file = f
            break
    if shots_file is None:
        return Response(status=404)
    data = read_json(shots_file)
    vpath = _resolve_video_path(data.get("video", ""), shots_file)
    buf = _read_frame_bytes(vpath, frame)
    if buf is None:
        return Response(status=404)
    return Response(buf, mimetype="image/jpeg")


@app.route("/frames/<video_id>/<int:start>/<int:end>")
def frames(video_id: str, start: int, end: int) -> Response:
    # returns JSON with list of image URLs for evenly spaced frames (k param)
    k = int(request.args.get("k", 6))
    if end < start:
        start, end = end, start
    if k <= 1:
        idxs = [start]
    else:
        span = max(1, end - start)
        idxs = [int(round(start + i * (end - start) / (k - 1))) for i in range(k)]
    urls = [url_for("image", video_id=video_id, frame=i) for i in idxs]
    return jsonify({"frames": idxs, "urls": urls})


@app.route("/simulate/<video_id>")
def simulate(video_id: str) -> Response:
    # Parse selection and total frame budget
    shots_str = request.args.get("shots", "").strip()
    try:
        selected = [int(x) for x in shots_str.split(",") if x.strip().isdigit()]
    except Exception:
        selected = []
    max_frames = int(request.args.get("max", CFG.stage2_max_frames) or CFG.stage2_max_frames)
    if not selected:
        return redirect(url_for('annotate', video_id=video_id))

    # Load shots and compute per-shot frames
    shots_file = None
    for f in _list_shots_files(CFG.shots_root):
        if f.parent.name == video_id or f.stem == video_id:
            shots_file = f
            break
    if shots_file is None:
        return Response(f"shots.json for {video_id} not found", status=404)
    data = read_json(shots_file)
    fps = float(data.get("fps", 30.0))
    shots = data.get("shots", [])
    per = max(1, int(max_frames // max(1, len(selected))))
    items = []
    for sid in selected:
        if not (0 <= sid < len(shots)):
            continue
        s = shots[sid]
        start = int(s.get("start_frame", 0))
        end = int(s.get("end_frame", 0))
        if end < start:
            start, end = end, start
        span = max(1, end - start)
        if per <= 1:
            idxs = [start]
        else:
            idxs = [int(round(start + i * (end - start) / (per - 1))) for i in range(per)]
        urls = [url_for('image', video_id=video_id, frame=i) for i in idxs]
        items.append({"shot_id": sid, "frames": idxs, "urls": urls, "start": start, "end": end})

    # Question
    qmap = _load_questions(CFG.question_file)
    qlist = qmap.get(video_id, [])
    question = None
    if qlist:
        q = qlist[0]
        question = {"question_id": q.get("question_id"), "question": q.get("question"), "options": q.get("options", []), "answer": q.get("answer")}

    return render_template(
        "simulate.html",
        video_id=video_id,
        question=question,
        items=items,
        max_frames=max_frames,
        selected_shots=selected,
    )


@app.route("/save_gt/<video_id>", methods=["POST"])
def save_gt(video_id: str) -> Response:
    payload = request.get_json(force=True, silent=True) or {}
    CFG.gt_dir.mkdir(parents=True, exist_ok=True)
    out = CFG.gt_dir / f"{video_id}.json"
    payload["video_id"] = video_id
    payload["timestamp"] = int(time.time())
    with out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return jsonify({"ok": True, "path": str(out)})


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Annotation Web Server")
    p.add_argument("--shots-root", type=str, default=str(CFG.shots_root), help="Root containing shots.json")
    p.add_argument("--question-file", type=str, default=None, help="Optional Video-MME question file (jsonl)")
    p.add_argument("--min-shots", type=int, default=0, help="Filter: min number of shots (inclusive)")
    p.add_argument("--max-shots", type=int, default=1000000, help="Filter: max number of shots (inclusive)")
    p.add_argument("--stage2-max-frames", type=int, default=128, help="Stage-2 total max frames to distribute across selected shots")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    CFG.shots_root = Path(args.shots_root).expanduser().resolve()
    CFG.question_file = Path(args.question_file).expanduser().resolve() if args.question_file else None
    CFG.gt_dir = Path("annotation/gt").resolve()
    CFG.min_shots = int(args.min_shots)
    CFG.max_shots = int(args.max_shots)
    CFG.stage2_max_frames = int(args.stage2_max_frames)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
