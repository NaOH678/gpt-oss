# zh_structured_verl

基于本地 `verl` 仓库的最小可跑方案，用于快速迭代：
- `gpt-oss-20b` 中文结构化输出 SFT
- verifier-guided GRPO（RLVR-lite）

## 目录

- `scripts/prepare_massive_zh.py`: MASSIVE zh-CN -> SFT/RL parquet
- `scripts/prepare_coig_cqia.py`: COIG-CQIA -> SFT parquet
- `scripts/build_sft_mix.py`: 70/30 混合 MASSIVE + COIG
- `scripts/prepare_gptoss20b_bf16.py`: `openai/gpt-oss-20b` 转本地 bf16
- `scripts/run_sft.sh`: SFT 训练入口（torchrun + `verl.trainer.sft_trainer`）
- `scripts/run_grpo_structured.sh`: GRPO 训练入口（`verl.trainer.main_ppo`）
- `scripts/run_offline_eval.sh`: 生成 + 离线打分
- `scripts/run_baseline_eval.sh`: 原始模型基线（生成 + 结构化评测）
- `scripts/evaluate_structured_calls.py`: 结构化指标汇总（JSON/Intent/Slot/Schema/EM）
- `rewards/structured_reward.py`: JSON/intent/slot/constraint 组合奖励

## 快速迭代路径（推荐）

### 0) 准备模型（一次性）

```bash
python /Users/naoh/Desktop/gpt-oss/zh_structured_verl/scripts/prepare_gptoss20b_bf16.py \
  --output_dir "$HOME/models/gpt-oss-20b-bf16"
```

### 1) 准备 MASSIVE（主任务）

```bash
python /Users/naoh/Desktop/gpt-oss/zh_structured_verl/scripts/prepare_massive_zh.py \
  --output_dir /Users/naoh/Desktop/gpt-oss/zh_structured_verl/data/massive_zh
```

产物：
- `sft_train/dev/test.parquet`
- `rl_train/dev/test.parquet`

### 1.5) 训练前基线评测（Base）

```bash
export ROLLOUT_TP_SIZE=2

bash /Users/naoh/Desktop/gpt-oss/zh_structured_verl/scripts/run_baseline_eval.sh \
  "$HOME/models/gpt-oss-20b-bf16" \
  /Users/naoh/Desktop/gpt-oss/zh_structured_verl/data/massive_zh/rl_test.parquet \
  /Users/naoh/Desktop/gpt-oss/zh_structured_verl/data/eval/base_test_responses.parquet \
  /Users/naoh/Desktop/gpt-oss/zh_structured_verl/data/eval/base_metrics \
  8
```

评测产物：
- `metrics_summary.json`
- `metrics_by_intent.json`
- `failure_cases.jsonl`

说明：
- 训练前/训练后的结构化指标对比请优先使用 `run_baseline_eval.sh` + `evaluate_structured_calls.py`。
- `run_offline_eval.sh` 仍保留用于原 `verl main_eval` 流程兼容。

### 2) （可选）准备 COIG 并混合到 SFT

```bash
python /Users/naoh/Desktop/gpt-oss/zh_structured_verl/scripts/prepare_coig_cqia.py \
  --output_dir /Users/naoh/Desktop/gpt-oss/zh_structured_verl/data/coig_cqia

python /Users/naoh/Desktop/gpt-oss/zh_structured_verl/scripts/build_sft_mix.py \
  --massive_train /Users/naoh/Desktop/gpt-oss/zh_structured_verl/data/massive_zh/sft_train.parquet \
  --coig_train /Users/naoh/Desktop/gpt-oss/zh_structured_verl/data/coig_cqia/sft_train.parquet \
  --output /Users/naoh/Desktop/gpt-oss/zh_structured_verl/data/sft_mix_70_30.parquet \
  --massive_ratio 0.7 --coig_ratio 0.3
```

### 3) SFT（MASSIVE-only baseline）

```bash
bash /Users/naoh/Desktop/gpt-oss/zh_structured_verl/scripts/run_sft.sh \
  8 \
  "$HOME/models/gpt-oss-20b-bf16" \
  /Users/naoh/Desktop/gpt-oss/zh_structured_verl/data/massive_zh/sft_train.parquet \
  /Users/naoh/Desktop/gpt-oss/zh_structured_verl/data/massive_zh/sft_dev.parquet \
  /Users/naoh/Desktop/gpt-oss/zh_structured_verl/checkpoints/sft_massive
```

### 4) SFT（MASSIVE+COIG）

```bash
bash /Users/naoh/Desktop/gpt-oss/zh_structured_verl/scripts/run_sft.sh \
  8 \
  "$HOME/models/gpt-oss-20b-bf16" \
  /Users/naoh/Desktop/gpt-oss/zh_structured_verl/data/sft_mix_70_30.parquet \
  /Users/naoh/Desktop/gpt-oss/zh_structured_verl/data/massive_zh/sft_dev.parquet \
  /Users/naoh/Desktop/gpt-oss/zh_structured_verl/checkpoints/sft_mix
```

### 5) GRPO（RLVR-lite）

```bash
export ROLLOUT_TP_SIZE=2

bash /Users/naoh/Desktop/gpt-oss/zh_structured_verl/scripts/run_grpo_structured.sh \
  /Users/naoh/Desktop/gpt-oss/zh_structured_verl/checkpoints/sft_massive/actor \
  /Users/naoh/Desktop/gpt-oss/zh_structured_verl/data/massive_zh/rl_train.parquet \
  /Users/naoh/Desktop/gpt-oss/zh_structured_verl/data/massive_zh/rl_dev.parquet \
  /Users/naoh/Desktop/gpt-oss/zh_structured_verl/checkpoints/grpo \
  8
```

### 6) 离线评测（生成 + verifier）

```bash
export ROLLOUT_TP_SIZE=2

bash /Users/naoh/Desktop/gpt-oss/zh_structured_verl/scripts/run_offline_eval.sh \
  /Users/naoh/Desktop/gpt-oss/zh_structured_verl/checkpoints/grpo/actor \
  /Users/naoh/Desktop/gpt-oss/zh_structured_verl/data/massive_zh/rl_test.parquet \
  /Users/naoh/Desktop/gpt-oss/zh_structured_verl/data/eval/grpo_test_responses.parquet \
  4 \
  8
```

### 7) 结果对比顺序（固定）

1. Base（原始模型）
2. SFT(MASSIVE)
3. SFT(MASSIVE+COIG)
4. SFT+GRPO

## 压缩节奏建议（不走四周）

- Day 1: 跑通 MASSIVE-only SFT（小样本 + 1 epoch）
- Day 2: 接 GRPO + verifier，拿到第一版 reward 提升曲线
- Day 3: 加入 COIG 混合对照 + 参数微调（`rollout.n`、batch、KL）
- Day 4+: 扩到 120b 短程 profiling（只做可迁移性验证）

## 关键约束与默认实现

- 结构化 reward = `0.2*json + 0.3*intent + 0.4*slot + 0.1*constraint`
- GRPO 组采样：`actor_rollout_ref.rollout.n=4`
- RL 仅使用 MASSIVE zh-CN
- COIG 仅用于 SFT
