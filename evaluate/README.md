# Evaluate Module Overview

本目录包含视频检索/推理评测所需的各个模块，围绕“二阶段（场景选择 + 细粒度回答）”流水线组织。下文说明各文件职责、数据流、提示词格式、以及如何运行与扩展。

## 目录结构与职责

- `evaluate_pipeline.py`
  - 顶层评测流程的编排入口。完成如下任务：
    - 载入数据集（shots.json 与题目）。
    - Round 1：构造场景级提示词并附代表帧图片，请模型选择若干 `shots`。
    - Round 2：基于选中 shots 构造细粒度提示词并附帧图片，缝合第一阶段思维链，生成最终答案。
    - 解析模型输出、计算指标、保存中间结果，并聚合总体指标。
    - 跳过已预测的视频（依据已存在的 `*_predictions.json`）。

- `model_adapter.py`
  - 模型适配层。通过 APIBank 轮换 Gemini/OpenAI 兼容接口的 Key，提供统一的 `predict` 方法。
  - 如缺少 `openai` 包，将回退到启发式（占位）预测。

- `media.py`
  - 媒体/图像辅助工具：
    - `video_capture(path)` 打开视频句柄（需安装 OpenCV）。
    - `encode_frame_b64(cap, idx)` 将帧编码为 base64 Data URL（`data:image/jpeg;base64,...`）。
    - `evenly_spaced_frames(s, e, k)` 在区间内均匀采样索引。

- `data_loader.py`
  - 读取并组织评测样本：
    - 从 `shots.json` 解析镜头信息（见 `utils.parse_shots`）。
    - 解析题目（如 Video-MME 格式）。
    - 解析视频真实路径和 FPS。

- `prompt_generator.py`
  - 提示词生成：
    - Round 1（ScenePromptGenerator）：按镜头聚合的场景级概述。
    - Round 2（ShotPromptGenerator）：选中镜头的细粒度描述。

- `evaluator.py`
  - 统一的模型接口 `ModelInterface` 与评测指标：
    - 多选题解析与准确率计算（`evaluate_multiple_choice`）。
    - 文本中解析选项/答案的工具（内部使用）。

- `utils.py`
  - 常用工具：日志、JSON 读写、shots.json 枚举、APIBank 路径校验等。

- `config.py`
  - 评测配置（数据路径、模型名、预算上限、是否保存预测等）。

## 数据流与执行流程

1. 读取 `shots_root` 下的 `shots.json`，并按视频分组题目（如提供）。
2. Round 1：
   - 生成场景级提示（文本），并为每个镜头附一张代表帧（base64 图片）。
   - 系统提示强约束：
     - 用 `<think>...</think>` 包裹分析；
     - 仅输出一行：`Shots: [i, j, ...]`，索引为 0-based、不超过 `max_budget`、不重复；
     - 不要在该行之后输出其他文本。
   - 发送多模态消息给模型，解析返回的镜头列表（优先解析 `Shots: [...]` 行）。
3. 过滤/截断为有效索引，并限制数量不超过 `max_budget`。
4. Round 2：
   - 生成细粒度提示（文本），并为每个选中镜头附 3 张均匀采样的帧图（base64）。
   - 系统提示强约束：
     - 先将 Round 1 的 `<think>` 与新帧证据缝合，推理写在 `<think>...</think>`；
     - 然后仅输出一行：`Answer: ...`（多选题只输出字母，如 `Answer: B`）。
   - 发送多模态消息给模型，得到最终回答文本。
5. 评测与持久化：
   - 对多选题，解析答案并计算准确率；
   - 从最终回答中抽取 `predicted_letter`/`predicted_option`，与标答比对得到 `reward`（答对为 1，否则 0）。
   - 立即写入 `runs/evaluation_predictions/{video_stem}_predictions.json`（包含 round_1/round_2 原始输出与 reward 等）。
6. 过滤已预测：
   - 如启用保存，则再次运行时会跳过已有 `*_predictions.json` 的视频，以节省推理成本。
7. 汇总：
   - 聚合各视频指标到 `runs/evaluation_cache/metrics_summary.json`。

## 多模态消息格式（OpenAI 兼容）

消息 `messages` 采用 `chat.completions.create` 的内容数组形式：

```jsonc
{
  "role": "user",
  "content": [
    {"type": "text", "text": "...prompt text..."},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
  ]
}
```

其中图片用 Data URL 内联（无需外部存储）。若缺少 OpenCV，系统会降级为仅文本并记录 warning。

## 运行方法

```bash
python -m evaluate.evaluate_pipeline \
  --shots-root output/videomme_batch \
  --model-name gemini-2.0-flash-lite
```

配置项来源于 `config.py`，如需修改默认路径/预算/是否保存等，可编辑配置或通过命令行参数覆盖：

- `--shots-root`：`shots.json` 根目录。
- `--model-name`：模型名（传给 OpenAI 兼容接口）。
- `--no-save-predictions`：不保存每视频预测文件。
- `--quiet`：安静模式（减少日志）。

注意：需要准备好 APIBank 仓库路径（见 `EvalConfig.api_bank_root`），并确保安装了 `openai` 与 `opencv-python` 以启用多模态图片传输。

## 结果产物

- 每视频预测：`runs/evaluation_predictions/{stem}_predictions.json`
  - `predictions.round_1` 与 `predictions.round_2` 原始输出
  - `predicted_letter` / `predicted_option` / `reward`
- 汇总指标：`runs/evaluation_cache/metrics_summary.json`

## 扩展建议

- 自定义模型：实现 `ModelInterface` 并在 `model_adapter.py` 中新增装载逻辑。
- 自定义指标：在 `evaluator.py` 新增计算函数，并在 pipeline 中接入。
- 提示词/帧采样：
  - 调整 Round 2 每镜头帧数：`media.evenly_spaced_frames(k)`。
  - 调整提示词模板：修改 `prompt_generator.py` 或在 pipeline 中更换生成器。
- 中间结果格式：如需导出 CSV 或追加更多诊断信号，可在 `evaluate_pipeline.py` 的写入逻辑上扩展。

