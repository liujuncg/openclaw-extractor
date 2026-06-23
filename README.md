# OpenClaw Extractor

从 OpenClaw Agent 轨迹日志中提取 SFT / DPO / Recovery 训练数据。

## 文件结构

```
openclaw_extractor/
├── models.py              # 数据模型（Trajectory, TrajectoryStep, TrainingValueScore 等）
├── parsers.py             # 轨迹解析器 + 结果推断 + 训练价值打分 + 用途分类
├── aggregate_sessions.py  # .trajectory.jsonl → session-level JSONL 聚合
├── extractor.py           # 主流水线：session JSONL → SFT/Recovery/DPO 候选
├── dpo_pairing.py         # DPO 配对：候选 → (chosen, rejected) pair
└── tests/
    ├── test_extractor.py      # 单元测试（8 个）
    └── test_dpo_pairing.py    # DPO 配对测试（11 个）
```

## 数据流

```
.trajectory.jsonl (1270 files)
  → aggregate_sessions.py → sessions.jsonl (1260 sessions)
  → extractor.py → dataset/
      ├── sft_positive.jsonl      (867 条)
      ├── recovery_training.jsonl  (41 条)
      ├── dpo_candidate.jsonl     (326 条)
      ├── graceful_abort.jsonl     (主动中止轨迹)
      └── manifest.json
  → dpo_pairing.py → dpo_pairs.jsonl (69 对)
      ├── high:   7 对 (success_vs_failure_at_error)
      ├── medium: 44 对 (quality_comparison)
      └── low:    18 对 (global_success_vs_failure)
```

## 用法

### 1. 聚合 session

```bash
python aggregate_sessions.py \
  --input ~/.openclaw/agents/main/sessions/ \
  --output sessions.jsonl
```

### 2. 提取训练数据

```bash
python extractor.py \
  --input sessions.jsonl \
  --output dataset/ \
  --min-score 0.5 \
  --format jsonl
```

### 3. DPO 配对

```bash
python dpo_pairing.py \
  --input dataset/dpo_candidate.jsonl dataset/recovery_training.jsonl \
  --output dataset/dpo_pairs.jsonl \
  --min-similarity 0.5 \
  --max-pairs-per-cluster 10
```

## DPO 配对架构

### 三池分离

| 池 | 来源 | 用途 |
|----|------|------|
| clean_success_pool | DPO candidate (outcome=success, 无 error) | quality_comparison + tool_similarity 匹配 |
| recovery_pool | recovery_training (有 error + recovery) | DPO error-state 配对的主要来源 |
| failure_pool | outcome=failure/unknown | DPO error-state 配对 |

### chosen / rejected 构建原则

- prompt = error 之前的上下文 + error tool_result
- chosen = 同一 session error 后的恢复行为（只含 tool_call，不含 tool_result）
- rejected = synthetic plain_retry（原样重试导致错误的调用）
- 不跨 session 取 chosen/rejected — 跨 session 只用于 tool_similarity 匹配

### retry 分类系统

| recovery_kind | 说明 |
|---------------|------|
| adaptive_retry | 改变了参数/方法/工具后重试 |
| backoff_retry | 加了 delay/sleep 后重试 |
| plain_retry | 原样重试（相同工具相同参数） |
| no_retry | 没有重试 |

### 质量分级

| quality | 条件 |
|---------|------|
| high | success_vs_failure_at_error + adaptive_retry/backoff_retry + resolved=True |
| medium | quality_comparison 或 plain_retry + resolved |
| low | global_success_vs_failure（无明确分叉点） |

### preference_strength

| strength | 条件 |
|----------|------|
| strong | rate_limit + failed_item_eventually_resolved=True + adaptive/backoff retry |
| medium | other_error / server_error，或 rate_limit 但未补齐失败项 |

### 去重

- 同一 near_duplicate_group_id（session:error_type:tool:action）最多保留 2 条
- 优先保留 preference_strength=strong 和 failed_item_eventually_resolved=True

### 关键 metadata 字段

```json
{
  "chosen_source": "same_session_recovery",
  "rejected_source": "synthetic_plain_retry",
  "pair_construction": "same_session_recovery_vs_synthetic_plain_retry",
  "synthetic_rejected": true,
  "recovery_kind": "adaptive_retry",
  "resolved": true,
  "retry_loop": false,
  "failed_item_eventually_resolved": true,
  "near_duplicate_group_id": "5754f150:rate_limit:feishu_doc:update_block",
  "preference_strength": "strong"
}
```

## 训练建议

| 子集 | 条件 | 数量 |
|------|------|------|
| 训练主集 | high + preference_strength=strong | 3 对 |
| 训练扩展集 | high + preference_strength=medium | 4 对 |
| 人工抽样 | 每个 diverge_reason 至少看 2 条 | — |
