#!/usr/bin/env python3
"""
build_eval_set.py — 从 extractor 输出构建私有分层评估集

用法:
  python3 build_eval_set.py \
    --dataset-dir ~/Downloads/dataset/ \
    --output ~/Downloads/eval_set/ \
    --samples-per-stratum 10

输入：extractor 的输出目录（含 rejected.jsonl, needs_review.jsonl,
      sft_positive.jsonl, dpo_candidate.jsonl, error_taxonomy.jsonl 等）

输出目录结构:
  eval_set/
  ├── eval_manifest.json          # 评估集总览
  ├── strata/
  │   ├── tool_light.jsonl        # 轻量工具调用（1-3次）
  │   ├── tool_heavy.jsonl        # 密集工具调用（>10次）
  │   ├── long_plan.jsonl         # 长程规划（steps>20）
  │   ├── error_recovery.jsonl    # 含错误恢复
  │   ├── graceful_abort.jsonl    # 主动中止
  │   ├── multi_agent.jsonl       # 多智体协作
  │   └── domain_*.jsonl          # 按工具域分层
  ├── fault_injection_targets.json # Fault Injection RL 优先场景
  └── label_queue.jsonl           # 待人工标注队列（含标注字段模板）
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


# ─────────────────────────────────────────────────────────────────
# 工具域映射
# ─────────────────────────────────────────────────────────────────

TOOL_DOMAINS = {
    "feishu_doc":     "feishu",
    "feishu_sheet":   "feishu",
    "feishu_message": "feishu",
    "browser":        "web",
    "web_search":     "web",
    "exec":           "code",
    "process":        "code",
    "read":           "file",
    "write":          "file",
    "memory_search":  "memory",
    "sessions_spawn": "agent",
    "sessions_yield": "agent",
}


def get_primary_domain(unique_tools: list[str]) -> str:
    domain_counts: dict[str, int] = defaultdict(int)
    for t in unique_tools:
        d = TOOL_DOMAINS.get(t, "other")
        domain_counts[d] += 1
    if not domain_counts:
        return "other"
    return max(domain_counts, key=lambda d: domain_counts[d])


# ─────────────────────────────────────────────────────────────────
# 分层规则
# ─────────────────────────────────────────────────────────────────

def classify_stratum(sample: dict) -> list[str]:
    """
    一个样本可以属于多个 stratum。
    返回所有匹配的 stratum 名称列表。
    """
    stats    = sample.get("stats", {})
    steps    = stats.get("steps", stats.get("total_steps", 0))
    tool_calls = stats.get("tool_calls", 0)
    tools    = stats.get("unique_tools", [])
    has_recovery = stats.get("has_recovery", False)
    has_multi_agent = any(t in ("sessions_spawn", "sessions_yield") for t in tools)

    strata = []

    # 工具调用密度
    if 0 < tool_calls <= 3:
        strata.append("tool_light")
    elif 4 <= tool_calls <= 10:
        strata.append("tool_medium")
    elif tool_calls > 10:
        strata.append("tool_heavy")

    # 步骤数
    if steps >= 20:
        strata.append("long_plan")

    # 错误恢复
    if has_recovery:
        strata.append("error_recovery")

    # 主动中止
    use = sample.get("training_use", "")
    if use == "graceful_abort" or sample.get("sample_type") == "graceful_abort":
        strata.append("graceful_abort")

    # 多智体
    if has_multi_agent:
        strata.append("multi_agent")

    # 工具域
    domain = get_primary_domain(tools)
    strata.append(f"domain_{domain}")

    # 兜底
    if not strata:
        strata.append("other")

    return strata


# ─────────────────────────────────────────────────────────────────
# 评估协议生成
# ─────────────────────────────────────────────────────────────────

def generate_eval_criteria(sample: dict) -> dict:
    """
    为每个样本生成评估协议。
    包含：expected_outcome、eval_dimensions、自动评估脚本提示。
    """
    stats    = sample.get("stats", {})
    tools    = stats.get("unique_tools", [])
    outcome  = sample.get("outcome", "unknown")
    use      = sample.get("training_use", "")

    # 预期结果
    if outcome in ("success", "likely_success"):
        expected = "task_completed"
    elif outcome in ("failure", "likely_failure"):
        expected = "task_failed"
    elif use == "graceful_abort":
        expected = "graceful_abort_with_reason"
    else:
        expected = "unknown"

    # 评估维度（按样本类型定制）
    dimensions = ["task_completion"]

    if stats.get("tool_calls", 0) > 0:
        dimensions.append("tool_call_accuracy")
    if stats.get("has_recovery", False):
        dimensions.append("error_recovery_quality")
    if stats.get("steps", 0) >= 20:
        dimensions.append("planning_coherence")
    if any(t in ("sessions_spawn", "sessions_yield") for t in tools):
        dimensions.append("subagent_delegation_quality")
    if use == "graceful_abort":
        dimensions.append("boundary_recognition")
        dimensions.append("user_communication_quality")

    # 自动化评估提示（给 LLM judge 用）
    auto_eval_prompt = _make_judge_prompt(sample, expected, dimensions)

    return {
        "expected_outcome":   expected,
        "eval_dimensions":    dimensions,
        "auto_eval_prompt":   auto_eval_prompt,
        "difficulty_estimate":_estimate_difficulty(stats),
        # 人工标注字段（预留）
        "human_label":        None,
        "label_confidence":   None,
        "label_notes":        None,
        "label_time":         None,
        "labeled_by":         None,
    }


def _make_judge_prompt(sample: dict, expected: str, dimensions: list[str]) -> str:
    user_input = (sample.get("user_input") or "")[:200]
    final_out  = (sample.get("final_output") or "")[:200]
    dims_str   = "、".join(dimensions)
    return (
        f"任务描述：{user_input}\n"
        f"预期结果：{expected}\n"
        f"实际输出：{final_out}\n"
        f"请从以下维度评估（1-5分）：{dims_str}\n"
        f"输出格式：JSON，每个维度一个字段，外加 overall_score 和 reasoning。"
    )


def _estimate_difficulty(stats: dict) -> str:
    steps = stats.get("steps", stats.get("total_steps", 0))
    tools = stats.get("tool_calls", 0)
    score = steps * 0.4 + tools * 0.6
    if score >= 20:
        return "hard"
    elif score >= 8:
        return "medium"
    return "easy"


# ─────────────────────────────────────────────────────────────────
# 读取 extractor 输出
# ─────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return samples


def load_all_samples(dataset_dir: Path) -> list[dict]:
    """读取所有有价值的样本（排除 manifest 和 rejected）"""
    all_samples = []
    priority_files = [
        "sft_positive.jsonl",
        "recovery_training.jsonl",
        "dpo_candidate.jsonl",
        "graceful_abort.jsonl",        # 新增通道
        "skill_violation_neg.jsonl",
        "needs_review.jsonl",
        "long_plan.jsonl",
        "subagent_collab.jsonl",
        "retry_sequences.jsonl",
    ]
    for fname in priority_files:
        samples = load_jsonl(dataset_dir / fname)
        for s in samples:
            s["_source_file"] = fname
        all_samples.extend(samples)
        if samples:
            print(f"  读取 {fname}: {len(samples)} 条", file=sys.stderr)

    # 也加载 rejected（用于 label_queue，不进 eval strata）
    rejected = load_jsonl(dataset_dir / "rejected.jsonl")
    for s in rejected:
        s["_source_file"] = "rejected.jsonl"
        s["_rejected"] = True
    all_samples.extend(rejected)

    return all_samples


# ─────────────────────────────────────────────────────────────────
# 分层采样
# ─────────────────────────────────────────────────────────────────

def stratified_sample(
    samples: list[dict],
    samples_per_stratum: int,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """按 stratum 分层，每层最多取 samples_per_stratum 条"""
    rng = random.Random(seed)

    # 分层
    strata_map: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        if s.get("_rejected"):
            continue   # rejected 单独处理
        for stratum in classify_stratum(s):
            strata_map[stratum].append(s)

    # 采样
    result: dict[str, list[dict]] = {}
    for stratum, stratum_samples in strata_map.items():
        if len(stratum_samples) <= samples_per_stratum:
            result[stratum] = stratum_samples
        else:
            result[stratum] = rng.sample(stratum_samples, samples_per_stratum)

    return result


# ─────────────────────────────────────────────────────────────────
# Fault Injection 优先场景生成
# ─────────────────────────────────────────────────────────────────

def build_fault_injection_targets(error_taxonomy_path: Path) -> list[dict]:
    """
    从 error_taxonomy.jsonl 提取 Fault Injection RL 的优先场景。
    输出按优先级排序的错误类型列表，含训练建议和典型样本。
    """
    taxonomy = load_jsonl(error_taxonomy_path)
    if not taxonomy:
        return []

    targets = []
    for row in sorted(taxonomy, key=lambda x: (
        0 if x.get("rl_priority") == "high" else
        1 if x.get("rl_priority") == "medium" else 2
    )):
        targets.append({
            "error_type":          row["error_type"],
            "occurrence_count":    row["count"],
            "session_count":       row.get("session_count", 0),
            "recovery_rate":       row.get("recovery_rate", 0),
            "rl_priority":         row.get("rl_priority", "low"),
            "fi_recommendation":   row.get("fi_recommendation", ""),
            "top_tools":           row.get("top_tools", []),
            # 训练配置建议
            "training_config": {
                "injection_rate":  (
                    0.3 if row.get("rl_priority") == "high" else
                    0.15 if row.get("rl_priority") == "medium" else 0.05
                ),
                "target_recovery_rate": 0.80,
                "current_recovery_rate": row.get("recovery_rate", 0),
                "gap": round(
                    0.80 - row.get("recovery_rate", 0), 3
                ),
            },
            "sample_contexts":     row.get("samples", [])[:3],
        })

    return targets


# ─────────────────────────────────────────────────────────────────
# 标注队列
# ─────────────────────────────────────────────────────────────────

def build_label_queue(
    all_samples: list[dict],
    max_queue_size: int = 200,
    seed: int = 42,
) -> list[dict]:
    """
    构建人工标注队列：优先选 needs_review 和低置信度样本。
    """
    rng = random.Random(seed)

    # 优先级：needs_review > outcome=unknown > 高复杂度样本
    priority_samples = []
    other_samples    = []

    for s in all_samples:
        if s.get("_rejected"):
            continue
        outcome = s.get("outcome", "")
        source  = s.get("_source_file", "")
        if "needs_review" in source or outcome in ("unknown", ""):
            priority_samples.append(s)
        else:
            other_samples.append(s)

    # 组合队列
    queue = priority_samples[:max_queue_size]
    remaining = max_queue_size - len(queue)
    if remaining > 0 and other_samples:
        # 从其他样本里按难度优先补充
        hard_samples = [s for s in other_samples
                        if _estimate_difficulty(s.get("stats", {})) == "hard"]
        queue.extend(rng.sample(hard_samples, min(remaining, len(hard_samples))))

    queue = queue[:max_queue_size]

    # 加标注字段模板
    label_queue = []
    for s in queue:
        eval_criteria = generate_eval_criteria(s)
        label_queue.append({
            "queue_id":          str(uuid4()),
            "session_id":        s.get("session_id"),
            "source_file":       s.get("_source_file"),
            "user_input":        (s.get("user_input") or "")[:500],
            "final_output":      (s.get("final_output") or "")[:500],
            "current_outcome":   s.get("outcome", "unknown"),
            "training_use":      s.get("training_use", "unknown"),
            "stats":             s.get("stats", {}),
            "eval_criteria":     eval_criteria,
            "judge_prompt":      eval_criteria["auto_eval_prompt"],
            # 标注字段（人工填写）
            "human_label":       None,
            "correct_outcome":   None,    # success/failure/partial/abort
            "correct_training_use": None, # 正确的训练用途
            "label_confidence":  None,    # 1-5
            "label_notes":       None,
            "labeled_by":        None,
            "label_time":        None,
        })

    return label_queue


# ─────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="从 extractor 输出构建私有分层评估集",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--dataset-dir", "-d", required=True,
                    help="extractor 输出目录")
    ap.add_argument("--output", "-o", required=True,
                    help="评估集输出目录")
    ap.add_argument("--samples-per-stratum", type=int, default=10,
                    help="每个 stratum 最多取多少样本")
    ap.add_argument("--label-queue-size", type=int, default=200,
                    help="标注队列大小")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir  = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    strata_dir  = output_dir / "strata"
    strata_dir.mkdir(exist_ok=True)

    print("读取 extractor 输出...", file=sys.stderr)
    all_samples = load_all_samples(dataset_dir)
    print(f"共 {len(all_samples)} 条样本", file=sys.stderr)

    # ── 分层采样 ──
    print("\n分层采样...", file=sys.stderr)
    strata = stratified_sample(all_samples, args.samples_per_stratum, args.seed)

    stratum_stats = {}
    total_eval = 0
    for stratum_name, stratum_samples in sorted(strata.items()):
        # 为每个样本加评估协议
        enriched = []
        for s in stratum_samples:
            s_copy = {k: v for k, v in s.items() if not k.startswith("_")}
            s_copy["eval_criteria"] = generate_eval_criteria(s)
            s_copy["stratum"] = stratum_name
            enriched.append(s_copy)

        path = strata_dir / f"{stratum_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for s in enriched:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        stratum_stats[stratum_name] = len(enriched)
        total_eval += len(enriched)
        print(f"  {stratum_name:25s} {len(enriched):4d} 条", file=sys.stderr)

    # ── Fault Injection 优先场景 ──
    fi_targets = build_fault_injection_targets(dataset_dir / "error_taxonomy.jsonl")
    if fi_targets:
        fi_path = output_dir / "fault_injection_targets.json"
        fi_path.write_text(json.dumps(fi_targets, indent=2, ensure_ascii=False))
        print(f"\nFault Injection 目标: {len(fi_targets)} 种错误类型 → {fi_path}",
              file=sys.stderr)

        print("\n── Fault Injection RL 优先级 ──", file=sys.stderr)
        for t in fi_targets[:6]:
            gap  = t["training_config"]["gap"]
            sign = "↑需提升" if gap > 0.1 else "✓已达标"
            print(f"  [{t['rl_priority']:6s}] {t['error_type']:15s}  "
                  f"恢复率 {t['recovery_rate']*100:.0f}%  {sign}  {t['fi_recommendation'][:40]}",
                  file=sys.stderr)

    # ── 标注队列 ──
    label_queue = build_label_queue(all_samples, args.label_queue_size, args.seed)
    lq_path = output_dir / "label_queue.jsonl"
    with open(lq_path, "w", encoding="utf-8") as f:
        for item in label_queue:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n标注队列: {len(label_queue)} 条 → {lq_path}", file=sys.stderr)

    # ── eval_manifest ──
    manifest = {
        "created_at":   datetime.now(tz=timezone.utc).isoformat(),
        "dataset_dir":  str(dataset_dir),
        "total_samples_input":  len(all_samples),
        "total_eval_samples":   total_eval,
        "strata":               stratum_stats,
        "label_queue_size":     len(label_queue),
        "fault_injection_targets": len(fi_targets),
        "usage": {
            "eval_command":  "python3 run_eval.py --eval-set eval_set/ --model hermes-v2",
            "label_command": "python3 label_tool.py --queue eval_set/label_queue.jsonl",
            "fi_command":    "python3 fault_inject.py --targets eval_set/fault_injection_targets.json",
        },
    }
    manifest_path = output_dir / "eval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"\n评估集构建完成:", file=sys.stderr)
    print(f"  总样本: {total_eval} 条（{len(strata)} 个 stratum）", file=sys.stderr)
    print(f"  输出目录: {output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
