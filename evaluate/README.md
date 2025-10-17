# Evaluate Module Overview

本目录包含视频检索/推理评测所需的各个模块，围绕“二阶段（场景选择 + 细粒度回答）”流水线组织，并可选启用“动作式（Action Planning）+ 成本记账”的 AIF‑V 评测协议。下文说明模块职责、数据流、提示词与解析、成本模型与指标、运行方式与扩展建议。

## 目录结构与职责

- `evaluate_pipeline.py`
  - 顶层评测流程的编排入口。完成如下任务：
    - 载入数据集（shots.json 与题目）。
    - Round 1：
      - 默认模式（选择模式）：构造场景级提示词并附代表帧图片，请模型选择若干 `shots`。
      - 规划模式（动作式）：引导模型输出 JSON 动作计划（在预算内选择 `peek_*`/`request_*` 动作），系统解析与执行后产出证据。
    - Round 2：基于选中 shots 或执行得到的证据，缝合 Round 1 思维链，生成最终答案。
    - 解析模型输出、计算指标、保存中间结果，并聚合总体指标。
    - 跳过已预测的视频（依据已存在的 `*_predictions.json`）。

- `model_adapter.py`
  - 模型适配层。通过 APIBank 轮换 Gemini/OpenAI 兼容接口的 Key，提供统一的 `predict` 方法。
  - 如缺少 `openai` 包，将回退到启发式（占位）预测。

- `media.py`
  - 媒体/图像辅助工具：
    - `video_capture(path)` 打开视频句柄（需安装 OpenCV）。
    - `encode_frame_b64(cap, idx)` 将帧编码为 base64 Data URL（`data:image/jpeg;base64,...`）。
    - `encode_crop_b64(cap, idx, bbox)` 按归一化 bbox 裁剪再编码（局部放大/ROI）。
    - `evenly_spaced_frames(s, e, k)` 在区间内均匀采样索引。

- `costs.py`
  - 成本模型与指标：
    - `CostTable`/`CostLedger` 管理动作成本与记账（token 等价与时延等价）。
    - 评测指标：`k_at_b`（K@B）、`c_at_a`（C@A，下界估计）、`oracle_regret`（神谕悔恨）。

- `action_schema.py`
  - 动作式提示与解析：
    - `build_planning_system_prompt(cfg, budget, shot_count)` 生成规划系统提示（含 JSON schema 与价目表）。
    - `parse_action_plan(text)` 从模型回复中提取 JSON 计划（优先 ```json fenced code```）。
    - `validate_actions(plan, shot_count)` 校验动作与参数（类型/范围/坐标合法性）。

- `action_runtime.py`
  - 动作执行器：
    - 执行 `peek_scene`/`peek_shot`/`request_hd_frame`/`request_clip_1s`/`request_hd_crop`，生成多模态证据块，并在 `CostLedger` 中记账。

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
2. Round 1（两种模式，二选一）：
   - 选择模式（默认）：
     - 生成场景级提示（文本 + 代表帧），系统提示要求：
       - 用 `<think>...</think>` 包裹分析；
       - 最后只输出一行：`Shots: [i, j, ...]`，0-based、不超过 `max_budget`、不重复；
       - 不在该行后输出其他文本；
       - 成本意识：建议 3–5 个镜头，越少越好但不丢正确性。
     - 解析 `Shots` 行，忽略 `<think>` 中的数字（支持 `Shots: [...]`、`Shots: 1,2,3`、`Shots - 1 2 3`）。
   - 规划模式（可选，开启 `enable_action_planning=True`）：
     - 生成动作式系统提示，引导模型仅输出 JSON 计划（含 `plan`/`budget`/`steps`）。
     - 支持动作：`peek_scene`、`peek_shot`、`request_hd_frame`、`request_clip_1s`、`request_hd_crop`。
     - 按 `round1_budget_token` 预算与 `cost_table_token` 价目表进行超预算截断与记账。
     - 执行动作并产生多模态证据块，供 Round 2 使用。
3. 过滤/截断镜头选择（选择模式），并限制数量不超过 `max_budget`。
4. Round 2：
   - 生成细粒度提示或直接使用动作执行得到的证据；
   - 系统提示要求：
     - 先在 `<think>` 中缝合 Round 1 思维链与证据；
     - 然后仅输出一行：`Answer: ...`（多选题只输出字母，如 `Answer: B`）。
   - 发送多模态消息给模型，得到最终回答文本。
5. 评测、成本与持久化：
   - 对多选题，解析答案并计算准确率；
   - 从最终回答中抽取 `predicted_letter`/`predicted_option`，与标答比对得到 `reward∈{0,1}`；
   - 记账动作成本，写入 `costs.total_token/total_latency` 与明细 `actions`；
   - 立即写入 `runs/evaluation_predictions/{video_stem}_predictions.json`（包含问题、原始输出、动作计划、已执行动作、成本等）。
6. 过滤已预测：
   - 如启用保存，则再次运行时会跳过已有 `*_predictions.json` 的视频，以节省推理成本。
7. 汇总：
   - 聚合各视频指标到 `runs/evaluation_cache/metrics_summary.json`；
   - 若启用成本模型，同时输出 K@B、C@A 与 Oracle‑Regret（需提供 oracle_cost）。

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

启用动作式规划（AIF‑V）：在 `evaluate/config.py` 中设置：

- `enable_action_planning=True`
- `round1_budget_token=20.0`（可调）
- `cost_table_token` / `cost_table_latency`：价目表（动作→成本）
- `budgets_token`: K@B 预算列表，例如 `(10.0, 20.0, 30.0, 40.0)`
- `acc_targets`: C@A 的准确率目标，例如 `(0.6, 0.7, 0.8, 0.9)`

## 结果产物

- 每视频预测：`runs/evaluation_predictions/{stem}_predictions.json`
  - `video`: 视频路径
  - `question`: 原始题干/选项/答案
  - `predictions`：
    - `round_1` / `round_2`: 模型原始输出
    - `predicted_letter` / `predicted_option` / `reward`
    - `budget_limit`: Round 1 选择模式的上限（条数）
    - `costs`: `{ total_token, total_latency, actions:[{act,args,units,cost_*}] }`
    - `action_plan`: 规划模式下的原始 JSON 计划（否则为 null）
    - `executed_actions`: 实际执行的动作列表（否则为 null）
    - `oracle_cost`: 若标注了最小必要证据，可填入理论最小成本（用于 Oracle‑Regret）
- 汇总指标：`runs/evaluation_cache/metrics_summary.json`
  - 传统指标：如 `multiple_choice_accuracy/accuracy`
  - 成本相关：`K@B/token/*`、`C@A/token/*`、`OracleRegret/token`（以及 `latency` 同步）

## 提示词与解析要点

- Round 1 选择模式：强约束 `Shots: [...]` 行（支持无括号/短横线变体），解析优先取最后一次出现；忽略 `<think>` 内数字。
- Round 2：答案必须为单行 `Answer: X`（多选仅输出字母）。
- 规划模式：仅输出一个 JSON 对象（可置于 ```json fenced block 内）；系统会按 schema 校验与预算截断。
- ROI 放大：使用 `request_hd_crop(shot_id, frame, bbox=[x1,y1,x2,y2])`，bbox 为归一化坐标。

## 扩展建议

- 自定义模型：实现 `ModelInterface` 并在 `model_adapter.py` 中新增装载逻辑。
- 自定义指标：在 `evaluator.py` 新增计算函数，并在 pipeline 中接入。
- 提示词/帧采样：
  - 调整 Round 2 每镜头帧数：`media.evenly_spaced_frames(k)`。
  - 调整提示词模板：修改 `prompt_generator.py` 或在 pipeline 中更换生成器。
- 中间结果格式：如需导出 CSV 或追加更多诊断信号，可在 `evaluate_pipeline.py` 的写入逻辑上扩展。
 - 场景层级（L0）/索引产物：可新增 `scene_index.py` 生成 `index.json`（scene→shots）、`shots.csv`（起止/摘要/关键帧戳）与 `frames/`（懒加载）。目前 `peek_scene` 以 shot 映射占位。

## 诊断与日志

- 日志：根 logger 设为 INFO，`evaluate` 命名空间按 `verbose` 控制；第三方库（httpx/openai 等）固定压到 WARNING，避免刷屏。
