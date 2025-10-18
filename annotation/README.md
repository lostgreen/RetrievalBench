# Annotation Tool (Web)

A lightweight web UI to visualize per-shot frames and questions, let annotators select shots to inspect, and record a ground‑truth (GT) trajectory.

## Features

- List videos (from `shots.json` under a root directory)
- Annotate a video:
  - Show question and options (if available)
  - Grid of shots with representative frame thumbnails
  - Click a shot to expand and preview evenly spaced frames
  - Select shots for GT; choose the final answer; leave notes
  - Save GT to `annotation/gt/{video_id}.json`
- Server logs annotator actions (peek_shot, request_hd_frame) for provenance

## Quickstart

1) Install dependencies (suggested):

```
pip install flask opencv-python
```

2) Run the server:

```
python -m annotation.app \
  --shots-root output/videomme_batch \
  --question-file /data5/zgw/video_datasets/Video-MME/videomme/videomme_question.json \
 --min-shots 8 --max-shots 32 \
  --stage2-max-frames 32
```

3) Open the browser:

- http://127.0.0.1:5000/videos

## File Outputs

- Per‑video GT: `annotation/gt/{video_id}.json`
  - Contains: question, selected_shots, actions (annotator interactions), final answer, notes, timestamp

## Notes

- If OpenCV is not available, images will not render; install `opencv-python` to enable frame decoding.
- The server reads `shots.json` structure produced by this repo and resolves absolute/relative video paths.
- This is a scaffold intended for extension (e.g., ROI crop requests, 1s clip previews, multi‑round annotations).

## Filtering and Two‑Stage Simulation

- Filtering: use `--min-shots` / `--max-shots` to list only videos whose shot count is within the range (e.g., 8–32) for quick testing.

- Two‑Stage Simulation:
  - Stage 1: Select a subset of shots（勾选卡片）。
  - Stage 2: Click “Simulate Stage 2” and set “Stage 2 max frames”（默认 128）。工具会将该总帧预算均分到所选镜头，每个镜头均匀采样对应数量的帧进行展示。
  - 所有 ROI 拖拽框都会以归一化坐标记录在 `proposed_crops` 中，避免再次加载时受显示尺寸影响。
- 只看 “why/为什么” 问题：

```
python -m annotation.app \
  --shots-root output/videomme_batch \
  --question-file /data5/zgw/video_datasets/Video-MME/videomme/videomme_question.json \
  --filter-why
```

- 自定义按问题文本过滤（正则，不区分大小写）：

```
python -m annotation.app \
  --shots-root output/videomme_batch \
  --question-file /data5/zgw/video_datasets/Video-MME/videomme/videomme_question.json \
  --question-contains "(why|为什么)"
```
