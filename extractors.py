"""
OpenClaw / Hermes 专项数据提取器
覆盖三批训练数据类型：

第一批（单 session）:
  ① tool_call_pairs     工具调用配对（SFT tool-use 精度）
  ② long_plan           长程规划轨迹（steps>20 的成功/部分成功）
  ③ graceful_abort      主动中止轨迹（Safety/Boundary）

第二批（需要跨 session）:
  ④ error_taxonomy      错误类型分布统计（指导 Fault Injection RL）
  ⑤ retry_sequences     同任务重试序列（失败→重试→成功）

第三批（Subagent 协作）:
  ⑥ subagent_collab     main + subagent 协作轨迹

用法（在 extractor.py 里已集成，也可单独使用）:
  from extractors import run_all_extractors
  results = run_all_extractors(trajectories, config)
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import combinations
from typing import Any, Optional
from uuid import uuid4

from models import SessionOutcome, StepType, Trajectory


# ─────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are Hermes, an intelligent agent within the OpenClaw framework. "
    "You have access to a set of Skills and tools. Always reason step-by-step, "
    "stay within your assigned Skill boundaries, and handle errors gracefully."
)

# 主动中止的信号词
ABORT_SIGNALS_ZH = [
    "等待", "确认", "需要您", "超出", "无法继续", "请指示",
    "建议", "分批", "筛选", "澄清", "暂停",
]
ABORT_SIGNALS_EN = [
    "waiting", "confirm", "clarif", "out of scope", "unable to proceed",
    "please advise", "suggest", "pause", "need your input",
]
ABORT_SIGNALS = ABORT_SIGNALS_ZH + ABORT_SIGNALS_EN

# Subagent 工具名
SUBAGENT_TOOLS = {"sessions_spawn", "sessions_yield", "subagent"}

# 重试序列的最大时间窗口（分钟）
RETRY_WINDOW_MINUTES = 15

# 长程规划的最低步骤数
LONG_PLAN_MIN_STEPS = 20

# SFT messages 里单条工具调用的最大输出字符
TOOL_OUTPUT_MAX = 2000


# ─────────────────────────────────────────────────────────────────
# 共用工具
# ─────────────────────────────────────────────────────────────────

def _step_to_text(step: dict) -> str:
    """把一个 step dict 转成 SFT assistant 片段"""
    stype = step.get("step_type", "")
    content = step.get("content") or ""

    if stype == "reasoning" and content:
        if content.startswith("[任务理解]"):
            return ""
        return f"<thinking>\n{content}\n</thinking>"

    elif stype == "tool_call":
        tool   = step.get("tool_name", "")
        params = step.get("input_params") or {}
        return (f"<tool_call>\n"
                f"{json.dumps({'tool': tool, 'params': params}, ensure_ascii=False)}\n"
                f"</tool_call>")

    elif stype == "tool_result":
        status = step.get("status", "success")
        out    = step.get("output") or step.get("error") or ""
        if not isinstance(out, str):
            out = json.dumps(out, ensure_ascii=False)
        return (f"<tool_result status=\"{status}\">\n"
                f"{out[:TOOL_OUTPUT_MAX]}\n</tool_result>")

    elif stype == "output" and content:
        return content

    return ""


def _steps_to_assistant(steps: list[dict]) -> str:
    parts = [_step_to_text(s) for s in steps]
    return "\n".join(p for p in parts if p)


def _make_sft_messages(user_input: str, steps: list[dict]) -> list[dict]:
    assistant = _steps_to_assistant(steps)
    msgs = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_input},
    ]
    if assistant:
        msgs.append({"role": "assistant", "content": assistant})
    return msgs


def _traj_steps(traj: Trajectory) -> list[dict]:
    """把 Trajectory.steps 转成 dict 列表（兼容 aggregate_sessions 的输出格式）"""
    from models import _step_to_dict
    return [_step_to_dict(s) for s in traj.steps]


def _parse_ts(ts: Any) -> Optional[datetime]:
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────────

@dataclass
class ExtractorConfig:
    # 第一批
    extract_tool_pairs:    bool = True
    extract_long_plan:     bool = True
    extract_graceful_abort:bool = True
    # 第二批
    extract_error_taxonomy:bool = True
    extract_retry_seq:     bool = True
    retry_window_minutes:  int  = RETRY_WINDOW_MINUTES
    # 第三批
    extract_subagent:      bool = True
    # 通用
    long_plan_min_steps:   int  = LONG_PLAN_MIN_STEPS
    tool_pair_min_calls:   int  = 2
    max_tool_pairs_per_session: int = 20
    # 工具调用配对至少保留多少步上文（防止过度碎片化）
    min_context_steps:     int  = 2


@dataclass
class ExtractorResult:
    tool_call_pairs:   list[dict] = field(default_factory=list)
    long_plan:         list[dict] = field(default_factory=list)
    graceful_abort:    list[dict] = field(default_factory=list)
    error_taxonomy:    list[dict] = field(default_factory=list)
    retry_sequences:   list[dict] = field(default_factory=list)
    subagent_collab:   list[dict] = field(default_factory=list)

    def stats(self) -> dict:
        return {
            "tool_call_pairs":  len(self.tool_call_pairs),
            "long_plan":        len(self.long_plan),
            "graceful_abort":   len(self.graceful_abort),
            "error_taxonomy":   len(self.error_taxonomy),
            "retry_sequences":  len(self.retry_sequences),
            "subagent_collab":  len(self.subagent_collab),
        }


# ─────────────────────────────────────────────────────────────────
# 第一批 ① 工具调用配对
# ─────────────────────────────────────────────────────────────────

def extract_tool_call_pairs(traj: Trajectory, cfg: ExtractorConfig) -> list[dict]:
    """
    从成功轨迹里提取单个工具调用的 SFT 样本。
    格式：给定上下文 → 正确的工具调用 + 结果

    每个样本是一个独立的工具调用决策时刻：
      prompt  = system + user_task + 到此刻之前的所有步骤
      chosen  = 正确的 tool_call + tool_result
    """
    # 只对成功轨迹提取（保证工具调用是正确的）
    if traj.outcome not in (SessionOutcome.SUCCESS, SessionOutcome.LIKELY_SUCCESS):
        return []
    if len(traj.tool_calls) < cfg.tool_pair_min_calls:
        return []

    steps  = _traj_steps(traj)
    pairs  = []
    count  = 0

    for i, step in enumerate(steps):
        if step.get("step_type") != "tool_call":
            continue
        if count >= cfg.max_tool_pairs_per_session:
            break

        tool_name = step.get("tool_name", "")
        if not tool_name:
            continue

        # 上文步骤数不足则跳过（防止过度碎片化，保留 2-3 轮依赖关系）
        if i < cfg.min_context_steps:
            continue

        # 找对应的 tool_result（紧随其后）
        result_step = None
        for j in range(i + 1, min(i + 3, len(steps))):
            if steps[j].get("step_type") == "tool_result":
                result_step = steps[j]
                break

        if result_step is None:
            continue
        if result_step.get("status") == "error":
            continue   # 错误的工具调用不进 SFT

        # 构建 context（此工具调用之前的所有步骤）
        context_steps = steps[:i]
        context_text  = _steps_to_assistant(context_steps)

        prompt_msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": traj.user_input},
        ]
        if context_text:
            prompt_msgs.append({"role": "assistant", "content": context_text})

        # 工具调用 + 结果作为 chosen
        chosen = _steps_to_assistant([step, result_step])
        if not chosen:
            continue

        # 提取 input_params 作为结构化标注
        input_params = step.get("input_params") or {}

        pairs.append({
            "sample_id":    str(uuid4()),
            "sample_type":  "tool_call_pair",
            "session_id":   traj.session_id,
            "tool_name":    tool_name,
            "call_index":   count,
            "input_params": input_params,
            "result_status":result_step.get("status", "success"),
            "context_steps": i,
            "prompt":       prompt_msgs,
            "chosen":       chosen,
            "metadata": {
                "agent":         traj.agent,
                "model_version": traj.model_version,
                "session_steps": len(steps),
                "unique_tools":  list(traj.unique_tools),
            },
        })
        count += 1

    return pairs


# ─────────────────────────────────────────────────────────────────
# 第一批 ② 长程规划轨迹
# ─────────────────────────────────────────────────────────────────

def extract_long_plan(traj: Trajectory, cfg: ExtractorConfig) -> list[dict]:
    """
    提取长程规划轨迹（steps >= long_plan_min_steps）。

    对成功轨迹：提取完整轨迹，标注规划结构。
    对失败轨迹：截取到第一个错误点之前，提取成功的前半段作为正样本。
    """
    steps = _traj_steps(traj)

    # 找第一个错误点
    first_error_idx = next(
        (i for i, s in enumerate(steps)
         if s.get("step_type") == "tool_result" and s.get("status") == "error"),
        None
    )

    # 成功轨迹：全量提取
    if traj.outcome in (SessionOutcome.SUCCESS, SessionOutcome.LIKELY_SUCCESS):
        if len(steps) < cfg.long_plan_min_steps:
            return []
        use_steps = steps
        plan_outcome = "complete"

    # 失败轨迹：截取错误点之前（前半段需足够长）
    elif first_error_idx is not None and first_error_idx >= cfg.long_plan_min_steps:
        use_steps = steps[:first_error_idx]
        plan_outcome = "partial_until_error"

    else:
        return []

    # 提取规划结构：找所有 reasoning 步骤作为规划节点
    plan_nodes = []
    for i, s in enumerate(use_steps):
        if s.get("step_type") == "reasoning" and s.get("content"):
            content = s["content"]
            if content.startswith("[任务理解]"):
                continue
            # 判断是否是规划性 reasoning（含数字列表、步骤词）
            is_planning = bool(re.search(
                r"(\d+[.、)）]|步骤|第[一二三四五六七八九十\d]+步|plan|step \d|first|then|next|finally)",
                content, re.IGNORECASE
            ))
            plan_nodes.append({
                "step_id":    s.get("step_id"),
                "content":    content[:300],
                "is_planning": is_planning,
            })

    # Skill 切换点
    skill_switches = []
    last_skill = None
    for s in use_steps:
        skill = s.get("skill_context")
        if skill and skill != last_skill:
            skill_switches.append({
                "step_id": s.get("step_id"),
                "from":    last_skill,
                "to":      skill,
            })
            last_skill = skill

    messages = _make_sft_messages(traj.user_input, use_steps)

    # SFT 分层标记
    complexity_tier = (
        "tier1" if len(use_steps) >= 30 and len(traj.unique_tools) >= 4
        else "tier2" if len(use_steps) >= 20
        else "tier3"
    )

    return [{
        "sample_id":       str(uuid4()),
        "sample_type":     "long_plan",
        "session_id":      traj.session_id,
        "plan_outcome":    plan_outcome,
        "complexity_tier": complexity_tier,
        "messages":        messages,
        "plan_structure": {
            "total_steps":    len(use_steps),
            "plan_nodes":     plan_nodes,
            "skill_switches": skill_switches,
            "unique_tools":   list(traj.unique_tools),
            "tool_call_count":len(traj.tool_calls),
        },
        "metadata": {
            "agent":         traj.agent,
            "model_version": traj.model_version,
            "original_steps":len(steps),
            "cut_at_step":   first_error_idx,
        },
    }]


# ─────────────────────────────────────────────────────────────────
# 第一批 ③ 主动中止轨迹
# ─────────────────────────────────────────────────────────────────

def extract_graceful_abort(traj: Trajectory, cfg: ExtractorConfig) -> list[dict]:
    """
    提取主动中止轨迹（Safety / Boundary 训练数据）。

    特征：
    - 有工具调用（Agent 真正执行了一些操作）
    - 没有工具错误（不是因为出错才停下）
    - 最终输出含中止信号词
    - outcome 是 partial / unknown（不是成功也不是失败）
    """
    if len(traj.tool_calls) == 0:
        return []
    if len(traj.error_steps) > 0:
        return []   # 有错误的是纠错轨迹，不是主动中止

    # 检查最终输出是否含中止信号
    final = traj.final_output or ""
    has_abort_signal = any(sig in final for sig in ABORT_SIGNALS)

    # 也检查最后一个 output 步骤
    output_steps = [s for s in traj.steps if s.step_type == StepType.OUTPUT]
    if output_steps:
        last_output = output_steps[-1].content or ""
        has_abort_signal = has_abort_signal or any(sig in last_output for sig in ABORT_SIGNALS)

    if not has_abort_signal:
        return []

    # 识别中止原因
    abort_reason = _classify_abort_reason(final or (output_steps[-1].content if output_steps else ""))

    steps    = _traj_steps(traj)
    messages = _make_sft_messages(traj.user_input, steps)

    return [{
        "sample_id":    str(uuid4()),
        "sample_type":  "graceful_abort",
        "session_id":   traj.session_id,
        "abort_reason": abort_reason,
        "messages":     messages,
        "abort_signal": final[:500] if final else "",
        "stats": {
            "steps":       len(traj.steps),
            "tool_calls":  len(traj.tool_calls),
            "unique_tools":list(traj.unique_tools),
        },
        "metadata": {
            "agent":         traj.agent,
            "model_version": traj.model_version,
            "outcome":       traj.outcome.value,
        },
    }]


def _classify_abort_reason(text: str) -> str:
    text_lower = text.lower()
    if any(w in text_lower for w in ["超出", "out of scope", "beyond", "limit", "限额"]):
        return "scope_exceeded"
    if any(w in text_lower for w in ["确认", "clarif", "confirm", "需要您", "need your"]):
        return "needs_confirmation"
    if any(w in text_lower for w in ["分批", "batch", "筛选", "filter", "范围"]):
        return "data_too_large"
    if any(w in text_lower for w in ["等待", "wait", "暂停", "pause"]):
        return "waiting_for_resource"
    return "other"


# ─────────────────────────────────────────────────────────────────
# 第二批 ④ 错误类型分布统计
# ─────────────────────────────────────────────────────────────────

ERROR_PATTERNS = {
    "rate_limit":   ["429", "rate limit", "rate_limit", "too many requests", "quota exceeded"],
    "timeout":      ["timeout", "timed out", "connection timeout", "deadline exceeded"],
    "auth_error":   ["401", "403", "unauthorized", "forbidden", "permission denied", "access denied"],
    "not_found":    ["404", "not found", "does not exist", "no such file", "missing"],
    "server_error": ["500", "502", "503", "504", "internal server error", "bad gateway"],
    "parse_error":  ["json decode", "parse error", "invalid format", "syntax error", "malformed"],
    "network_error":["connection refused", "network error", "dns", "unreachable", "eof"],
    "tool_missing": ["tool not found", "unknown tool", "no such tool"],
    "data_error":   ["type error", "key error", "index error", "attribute error", "valueerror"],
}


def _detect_error_type(error_text: str) -> str:
    low = (error_text or "").lower()
    for etype, patterns in ERROR_PATTERNS.items():
        if any(p in low for p in patterns):
            return etype
    return "other"


def extract_error_taxonomy(trajs: list[Trajectory]) -> list[dict]:
    """
    跨所有轨迹统计错误类型分布。
    输出：每种错误类型的详细统计 + 典型样本。
    用途：指导 Fault Injection RL 的重点覆盖场景。
    """
    # 按工具 + 错误类型聚合
    taxonomy: dict[str, dict] = defaultdict(lambda: {
        "count": 0,
        "tools": defaultdict(int),
        "sessions": [],
        "samples": [],   # 最多保留5个典型样本
        "recovery_count": 0,   # 有多少次成功恢复了
    })

    for traj in trajs:
        steps = _traj_steps(traj)
        for i, step in enumerate(steps):
            if step.get("step_type") != "tool_result":
                continue
            if step.get("status") != "error":
                continue

            error_text = step.get("error") or str(step.get("output", ""))
            error_type = _detect_error_type(error_text)
            tool_name  = ""
            # 找对应的 tool_call
            for j in range(i - 1, max(i - 3, -1), -1):
                if steps[j].get("step_type") == "tool_call":
                    tool_name = steps[j].get("tool_name", "")
                    break

            rec = taxonomy[error_type]
            rec["count"] += 1
            rec["tools"][tool_name] += 1

            if traj.session_id not in rec["sessions"]:
                rec["sessions"].append(traj.session_id)

            # 检查是否有后续恢复
            recovered = False
            for k in range(i + 1, min(i + 5, len(steps))):
                s = steps[k]
                if s.get("step_type") == "tool_result" and s.get("status") == "success":
                    # 同一工具成功了 = 恢复
                    if any(steps[j2].get("tool_name") == tool_name
                           for j2 in range(k-1, max(k-3, -1), -1)
                           if steps[j2].get("step_type") == "tool_call"):
                        recovered = True
                        break
            if recovered:
                rec["recovery_count"] += 1

            # 保留典型样本
            if len(rec["samples"]) < 5:
                # 上下文：错误前2步 + 错误本身 + 后2步
                ctx_start = max(0, i - 2)
                ctx_end   = min(len(steps), i + 3)
                rec["samples"].append({
                    "session_id":  traj.session_id,
                    "tool_name":   tool_name,
                    "error_text":  error_text[:300],
                    "recovered":   recovered,
                    "context":     [_step_to_text(s) for s in steps[ctx_start:ctx_end]],
                })

    # 转换为输出格式
    results = []
    for error_type, data in sorted(taxonomy.items(), key=lambda x: -x[1]["count"]):
        recovery_rate = (
            data["recovery_count"] / data["count"]
            if data["count"] > 0 else 0.0
        )
        results.append({
            "error_type":      error_type,
            "count":           data["count"],
            "session_count":   len(data["sessions"]),
            "recovery_count":  data["recovery_count"],
            "recovery_rate":   round(recovery_rate, 3),
            "top_tools":       sorted(
                data["tools"].items(), key=lambda x: -x[1]
            )[:5],
            "rl_priority": (
                "high"   if data["count"] >= 10 and recovery_rate < 0.3 else
                "medium" if data["count"] >= 5  else
                "low"
            ),
            "fi_recommendation": _fi_recommendation(error_type, recovery_rate),
            "samples":         data["samples"],
        })

    return results


def _fi_recommendation(error_type: str, recovery_rate: float) -> str:
    """生成 Fault Injection RL 的训练建议"""
    recs = {
        "rate_limit":    "注入429错误，训练 sleep+retry 策略；目标恢复率 >80%",
        "timeout":       "注入连接超时，训练检查服务状态后决策是否重试",
        "auth_error":    "注入401/403，训练识别权限问题后主动中止（不重试）",
        "not_found":     "注入404，训练区分资源不存在和路径错误两种场景",
        "server_error":  "注入5xx，训练指数退避重试策略",
        "parse_error":   "注入格式错误，训练fallback解析和格式纠正",
        "network_error": "注入网络中断，训练等待后重试 vs 主动上报的决策",
        "data_error":    "注入类型错误，训练参数校验和类型转换",
        "tool_missing":  "训练识别工具不可用后的替代路径",
        "other":         "人工分析样本后定制训练策略",
    }
    base = recs.get(error_type, "人工分析")
    if recovery_rate > 0.7:
        return f"{base}（当前模型已有较好恢复能力，可降低优先级）"
    return base


# ─────────────────────────────────────────────────────────────────
# 第二批 ⑤ 同任务重试序列
# ─────────────────────────────────────────────────────────────────

def _task_fingerprint_simple(user_input: str) -> str:
    """
    从 user_input 提取任务指纹，用于跨 session 相似度判断。
    策略：提取关键名词和动词，忽略数字和时间。
    """
    # 去除时间戳、数字、标点
    text = re.sub(r"\d{4}[-/]\d{2}[-/]\d{2}|\d+", "", user_input)
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text)
    # 提取前100字符作为指纹基础
    return " ".join(text.split()[:20])


def _input_similarity(a: str, b: str) -> float:
    """简单的字符级 Jaccard 相似度（不依赖外部库）"""
    if not a or not b:
        return 0.0
    # 用 bigram
    def bigrams(s):
        return {s[i:i+2] for i in range(len(s)-1)}
    ba, bb = bigrams(a[:200]), bigrams(b[:200])
    if not ba or not bb:
        return 0.0
    return len(ba & bb) / len(ba | bb)


def extract_retry_sequences(trajs: list[Trajectory], cfg: ExtractorConfig) -> list[dict]:
    """
    跨 session 找同任务重试序列（失败→重试→成功）。

    配对条件：
    1. 时间间隔 < retry_window_minutes
    2. user_input 相似度 > 0.3
    3. 前一条 outcome 为 failure/likely_failure，后一条为 success/likely_success
    """
    # 按时间排序
    timed = []
    for traj in trajs:
        ts = traj.created_at
        if ts:
            timed.append((ts, traj))
    timed.sort(key=lambda x: x[0])

    sequences = []
    window = timedelta(minutes=cfg.retry_window_minutes)

    for i, (ts_i, traj_i) in enumerate(timed):
        # 只看失败轨迹作为起点
        if traj_i.outcome not in (
            SessionOutcome.FAILURE, SessionOutcome.LIKELY_FAILURE
        ):
            continue

        for j in range(i + 1, len(timed)):
            ts_j, traj_j = timed[j]

            # 时间窗口检查
            if ts_j - ts_i > window:
                break

            # 跳过同一 session
            if traj_j.session_id == traj_i.session_id:
                continue

            # 必须是成功轨迹
            if traj_j.outcome not in (
                SessionOutcome.SUCCESS, SessionOutcome.LIKELY_SUCCESS
            ):
                continue

            # 相似度检查
            sim = _input_similarity(traj_i.user_input, traj_j.user_input)
            if sim < 0.3:
                continue

            # 提取失败原因（从错误步骤）
            fail_reason = "unknown"
            if traj_i.error_steps:
                first_err = traj_i.error_steps[0]
                err_text  = first_err.error or str(first_err.output or "")
                fail_reason = _detect_error_type(err_text)

            # 提取用户在重试时补充的指令（user_input 的差异部分）
            # 用相似度的补集来近似
            added_context = _diff_input(traj_i.user_input, traj_j.user_input)

            sequences.append({
                "sequence_id":    str(uuid4()),
                "sample_type":    "retry_sequence",
                "time_gap_sec":   int((ts_j - ts_i).total_seconds()),
                "input_similarity": round(sim, 3),
                "fail_reason":    fail_reason,
                "added_context":  added_context,

                # 失败轨迹
                "attempt_1": {
                    "session_id":  traj_i.session_id,
                    "user_input":  traj_i.user_input[:500],
                    "outcome":     traj_i.outcome.value,
                    "steps":       len(traj_i.steps),
                    "tool_calls":  len(traj_i.tool_calls),
                    "error_steps": len(traj_i.error_steps),
                    "final_output":traj_i.final_output[:300] if traj_i.final_output else "",
                    "messages":    _make_sft_messages(
                        traj_i.user_input, _traj_steps(traj_i)
                    ),
                },

                # 成功轨迹（重试）
                "attempt_2": {
                    "session_id":  traj_j.session_id,
                    "user_input":  traj_j.user_input[:500],
                    "outcome":     traj_j.outcome.value,
                    "steps":       len(traj_j.steps),
                    "tool_calls":  len(traj_j.tool_calls),
                    "final_output":traj_j.final_output[:300] if traj_j.final_output else "",
                    "messages":    _make_sft_messages(
                        traj_j.user_input, _traj_steps(traj_j)
                    ),
                },

                # DPO 价值：attempt_1 作为 rejected，attempt_2 作为 chosen
                "dpo_value": {
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": traj_i.user_input[:500]},
                    ],
                    "chosen":   _steps_to_assistant(_traj_steps(traj_j)[:10]),
                    "rejected": _steps_to_assistant(_traj_steps(traj_i)[:10]),
                },

                "metadata": {
                    "agent_i":  traj_i.agent,
                    "agent_j":  traj_j.agent,
                    "model_i":  traj_i.model_version,
                    "model_j":  traj_j.model_version,
                },
            })

    return sequences


def _diff_input(old: str, new: str) -> str:
    """提取两个 user_input 之间的新增内容（近似）"""
    old_words = set(old.split())
    new_words  = new.split()
    added = [w for w in new_words if w not in old_words]
    return " ".join(added[:30])


# ─────────────────────────────────────────────────────────────────
# 第三批 ⑥ Subagent 协作轨迹
# ─────────────────────────────────────────────────────────────────

def extract_subagent_collab(trajs: list[Trajectory]) -> list[dict]:
    """
    提取包含 Subagent 协作的轨迹。

    识别特征：工具调用里包含 sessions_spawn / sessions_yield。
    提取内容：
    - 任务分解结构（main agent 如何拆解任务）
    - 子任务分配（spawn 的参数）
    - 结果聚合（yield 的处理方式）
    - 完整的 orchestration 轨迹
    """
    results = []

    for traj in trajs:
        # 找所有 subagent 工具调用
        spawn_calls  = [s for s in traj.tool_calls
                        if (s.tool_name or "").lower() in SUBAGENT_TOOLS
                        or "spawn" in (s.tool_name or "").lower()]
        yield_calls  = [s for s in traj.tool_calls
                        if "yield" in (s.tool_name or "").lower()]

        if not spawn_calls and not yield_calls:
            continue

        steps = _traj_steps(traj)

        # 提取任务分解结构
        decomposition = []
        for s in spawn_calls:
            params = s.input_params or {}
            subtask = (
                params.get("task") or
                params.get("prompt") or
                params.get("message") or
                str(params)[:200]
            )
            decomposition.append({
                "step_id":  s.step_id,
                "tool":     s.tool_name,
                "subtask":  subtask[:300] if subtask else "",
                "params":   {k: str(v)[:100] for k, v in params.items()},
            })

        # 提取结果聚合方式（yield 之后的 reasoning）
        aggregation_patterns = []
        for y in yield_calls:
            # 找 yield 之后的 reasoning
            y_idx = next(
                (i for i, s in enumerate(steps) if s.get("step_id") == y.step_id),
                None
            )
            if y_idx is not None:
                for k in range(y_idx + 1, min(y_idx + 3, len(steps))):
                    if steps[k].get("step_type") == "reasoning":
                        aggregation_patterns.append({
                            "after_yield_step": y.step_id,
                            "reasoning": steps[k].get("content", "")[:300],
                        })
                        break

        # 主任务类型判断
        collab_type = _classify_collab_type(traj, spawn_calls)

        messages = _make_sft_messages(traj.user_input, steps)

        results.append({
            "sample_id":     str(uuid4()),
            "sample_type":   "subagent_collab",
            "session_id":    traj.session_id,
            "collab_type":   collab_type,
            "outcome":       traj.outcome.value,
            "messages":      messages,

            "orchestration": {
                "spawn_count":          len(spawn_calls),
                "yield_count":          len(yield_calls),
                "decomposition":        decomposition,
                "aggregation_patterns": aggregation_patterns,
                "unique_subtask_tools": list(traj.unique_tools - SUBAGENT_TOOLS),
            },

            "training_notes": {
                "teaches": [
                    "任务分解策略（何时以及如何拆分子任务）",
                    "子任务参数构造（spawn 的 prompt 设计）",
                    "结果聚合模式（如何整合 subagent 输出）",
                ] if traj.outcome in (SessionOutcome.SUCCESS, SessionOutcome.LIKELY_SUCCESS)
                else [
                    "subagent 协作失败的常见模式",
                    "任务分解粒度不当的负例",
                ],
                "value_tier": (
                    "tier1" if len(spawn_calls) >= 3 and
                    traj.outcome in (SessionOutcome.SUCCESS, SessionOutcome.LIKELY_SUCCESS)
                    else "tier2"
                ),
            },

            "metadata": {
                "agent":         traj.agent,
                "model_version": traj.model_version,
                "total_steps":   len(traj.steps),
                "tool_calls":    len(traj.tool_calls),
            },
        })

    return results


def _classify_collab_type(traj: Trajectory, spawn_calls) -> str:
    """判断协作类型"""
    n = len(spawn_calls)
    tools = traj.unique_tools
    if n == 1:
        return "single_subagent"
    elif n >= 2 and len(tools) >= 3:
        return "parallel_subagents"
    elif n >= 2:
        return "sequential_subagents"
    return "yield_only"


# ─────────────────────────────────────────────────────────────────
# 统一入口
# ─────────────────────────────────────────────────────────────────

def run_all_extractors(
    trajs: list[Trajectory],
    cfg: Optional[ExtractorConfig] = None,
) -> ExtractorResult:
    """
    对所有轨迹运行全部专项提取器。
    返回 ExtractorResult，每个字段对应一类训练数据。
    """
    if cfg is None:
        cfg = ExtractorConfig()

    result = ExtractorResult()

    # ── 第一批：单 session ──
    for traj in trajs:
        if cfg.extract_tool_pairs:
            result.tool_call_pairs.extend(extract_tool_call_pairs(traj, cfg))

        if cfg.extract_long_plan:
            result.long_plan.extend(extract_long_plan(traj, cfg))

        if cfg.extract_graceful_abort:
            result.graceful_abort.extend(extract_graceful_abort(traj, cfg))

    # ── 第二批：跨 session ──
    if cfg.extract_error_taxonomy:
        result.error_taxonomy = extract_error_taxonomy(trajs)

    if cfg.extract_retry_seq:
        result.retry_sequences = extract_retry_sequences(trajs, cfg)

    # ── 第三批：Subagent ──
    if cfg.extract_subagent:
        result.subagent_collab = extract_subagent_collab(trajs)

    return result
