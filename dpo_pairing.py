#!/usr/bin/env python3
"""
DPO 配对模块：从 extractor 输出的 dpo_candidate.jsonl 里
自动提取高价值的 (chosen, rejected) 对。

用法:
  python3 dpo_pairing.py \
    --input  ~/Downloads/dataset/dpo_candidate.jsonl \
    --output ~/Downloads/dataset/dpo_pairs.jsonl \
    --min-similarity 0.5

输出格式（每行一个配对）:
  {
    "pair_id": "...",
    "pair_type": "success_vs_failure | quality_comparison | ...",
    "task_type": "feishu_doc:batch_update",
    "diverge_reason": "rate_limit_retry | tool_error_recovery | ...",
    "prompt":    [...messages 截到分叉点...],
    "chosen":    "分叉点之后的正确行为",
    "rejected":  "分叉点之后的错误行为",
    "metadata":  {...}
  }
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Optional, List
from uuid import uuid4


# ─────────────────────────────────────────────────────────────────
# 任务指纹
# ─────────────────────────────────────────────────────────────────

# 工具名 → 任务域的映射（按 OpenClaw 实际工具名）
TOOL_DOMAIN_MAP = {
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

# 错误类型关键词

def _get_steps(sample):
    """从 sample 中提取 trajectory steps，兼容 4 种格式：
    1. sample["trajectory"] 是 list → 直接返回
    2. sample["trajectory"] 是 dict，含 "trajectory" key → 返回内层
    3. sample["trajectory"] 是 dict，含 "steps" key → 返回 steps
    4. sample["steps"] → 返回 steps
    """
    traj = sample.get("trajectory")
    if isinstance(traj, list):
        return traj
    if isinstance(traj, dict):
        if "trajectory" in traj:
            return traj["trajectory"]
        if "steps" in traj:
            return traj["steps"]
    if "steps" in sample:
        return sample["steps"]
    return []


def _get_user_input(sample):
    traj = sample.get("trajectory")
    if isinstance(traj, dict):
        return traj.get("user_input") or sample.get("user_input", "")
    return sample.get("user_input", "")

ERROR_PATTERNS = {
    "rate_limit":    ["429", "rate limit", "rate_limit", "too many requests", "quota"],
    "timeout":       ["timeout", "timed out", "connection timeout", "read timeout"],
    "auth_error":    ["401", "403", "unauthorized", "forbidden", "permission denied"],
    "not_found":     ["404", "not found", "does not exist", "no such file"],
    "server_error":  ["500", "502", "503", "internal server error", "bad gateway"],
    "parse_error":   ["json decode", "parse error", "invalid format", "syntax error"],
    "tool_missing":  ["tool not found", "unknown tool", "no tool"],
}

# 失败行为模式（说了不做）— reasoning 里的声明
INTENT_WITHOUT_ACTION_PATTERNS = [
    r"(应该|需要|要).{0,20}(重试|retry|再试|try again)",
    r"(let me|let's|i will|i should|i would|i'm going to|i plan to|will|should|would|going to|plan to).{0,20}(retry|try again)",
    r"(等待|wait).{0,15}(重试|retry)",
    r"(标记为|mark(ed)? as).{0,10}(失败|failed|error)",
    r"(无法|can\'?t|cannot|unable to).{0,15}(继续|proceed|continue)",
    r"任务.{0,20}(失败|终止|中止|结束)",
    r"(task|operation).{0,20}(failed|aborted|terminated)",
]


def compute_task_fingerprint(sample: dict) -> str:
    """
    计算任务指纹：工具域 + 主要工具名组合。
    相同指纹的 session 视为同类任务。
    """
    stats = sample.get("stats", {})
    tools = set(stats.get("unique_tools", []))

    # 映射到域
    domains = {TOOL_DOMAIN_MAP.get(t, "other") for t in tools}

    # 指纹 = 排序后的工具名（取前5个）+ 域
    tool_sig   = "|".join(sorted(tools)[:5])
    domain_sig = "|".join(sorted(domains))

    return f"{domain_sig}::{tool_sig}"


def tools_jaccard(a: dict, b: dict) -> float:
    """两个 session 的工具集 Jaccard 相似度"""
    ta = set(a.get("stats", {}).get("unique_tools", []))
    tb = set(b.get("stats", {}).get("unique_tools", []))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ─────────────────────────────────────────────────────────────────
# 错误和失败模式分析
# ─────────────────────────────────────────────────────────────────

def detect_error_type(error_text: str) -> str:
    """从错误信息里检测错误类型"""
    if not error_text:
        return "unknown"
    low = error_text.lower()
    for etype, patterns in ERROR_PATTERNS.items():
        if any(p in low for p in patterns):
            return etype
    return "other_error"


def has_intent_without_action(reasoning_text: str) -> bool:
    """检测 reasoning 里是否有重试/恢复的声明"""
    if not reasoning_text:
        return False
    for pat in INTENT_WITHOUT_ACTION_PATTERNS:
        if re.search(pat, reasoning_text, re.IGNORECASE):
            return True
    return False


def _is_intent_only(steps: list[dict], error_idx: int, next_reasoning) -> bool:
    """
    真正的'说了不做'判断：
    reasoning 或 output 声明了要重试，但后续 3 步内没有实际 tool_call 执行恢复。
    """
    if next_reasoning is None:
        return False
    # 支持 output 步骤（content 字段）和 reasoning 步骤
    reasoning_content = next_reasoning.get("content", "") or next_reasoning.get("text", "")
    if not has_intent_without_action(reasoning_content):
        return False  # 根本没说要重试
    # 找 next_reasoning 在 steps 里的位置
    try:
        r_idx = next((i for i, s in enumerate(steps)
                      if s is next_reasoning), None)
    except Exception:
        r_idx = None
    if r_idx is None:
        return True  # 找不到位置，保守认为是 intent_only
    after = steps[r_idx + 1: r_idx + 4]
    has_tool = any(s.get("step_type") == "tool_call" for s in after)
    return not has_tool  # 没有后续 tool_call = 说了不做


def _check_failed_item_resolved(traj_steps: list[dict], error_idx: int) -> bool:
    """
    检查 error 对应的 tool_call 的目标对象之后是否被成功完成。
    例如：429 失败的 block_id 之后是否被成功 update。
    用于区分 "skip + later resolve" (strong) vs "skip + abandon" (medium)。

    支持跨工具恢复：如 feishu_doc.update_block 失败后用 browser 直接操作，
    只要 key_params (block_id/doc_token/url/path) 匹配即视为 resolved。
    """
    if error_idx <= 0 or error_idx >= len(traj_steps):
        return False
    error_call = traj_steps[error_idx - 1]
    if error_call.get("step_type") != "tool_call":
        return False
    tool = error_call.get("tool_name", "")
    params = error_call.get("input_params", {}) or {}

    # 提取关键参数（block_id, doc_token, url, path, file_id, message_id 等）
    key_params = {}
    for k in ("block_id", "doc_token", "url", "path", "file_id", "message_id", "page_id"):
        if k in params:
            key_params[k] = params[k]
    if not key_params:
        return False

    # 在 error 之后搜索是否有相同 key_params 的成功 tool_call（不要求 tool_name 相同）
    for j in range(error_idx + 1, len(traj_steps)):
        step = traj_steps[j]
        if step.get("step_type") != "tool_call":
            continue
        # 不再要求 step.get("tool_name") == tool，支持跨工具恢复
        step_params = step.get("input_params", {}) or {}
        # 检查关键参数是否匹配
        match = True
        for k, v in key_params.items():
            if step_params.get(k) != v:
                match = False
                break
        if match:
            # 检查这个 tool_call 的 result 是否成功
            if j + 1 < len(traj_steps):
                result = traj_steps[j + 1]
                if result.get("status") == "success":
                    return True
    return False


def _make_near_dedup_id(failure: dict, diverge: dict) -> str:
    """
    构造近重复组 ID，用于去重。
    格式: session_id:error_type:failed_tool:failed_action
    """
    session_id = failure.get("session_id", "")[:8]
    error_type = diverge.get("error_type", "unknown")

    error_idx = diverge.get("error_step_idx", 0)
    traj_steps = _get_steps(failure)
    
    failed_tool = ""
    failed_action = ""
    if error_idx > 0 and error_idx < len(traj_steps):
        call_step = traj_steps[error_idx - 1]
        if call_step.get("step_type") == "tool_call":
            failed_tool = call_step.get("tool_name", "")
            params = call_step.get("input_params", {}) or {}
            failed_action = params.get("action", "")
    
    return f"{session_id}:{error_type}:{failed_tool}:{failed_action}"


def _params_changed(params_a: dict, params_b: dict) -> bool:
    """判断两个 tool_call 的参数是否有实质性变化。
    targetId 变化不算（browser tab ID 每次都不同）。
    但 targetId 的有无变化算（从无到有 = adaptive）。
    """
    if not params_a or not params_b:
        return False
    # 只忽略 call_id/result_call_id（内部通信 ID，无业务含义）
    # 注意：不忽略 targetId，因为 targetId 有无变化是 adaptive 的关键信号
    ignore_keys = {"call_id", "result_call_id"}
    keys_a = set(params_a.keys()) - ignore_keys
    keys_b = set(params_b.keys()) - ignore_keys
    if keys_a != keys_b:
        return True
    for k in keys_a:
        val_a = str(params_a[k])[:200]
        val_b = str(params_b[k])[:200]
        if val_a != val_b:
            return True
    return False


def _classify_retry(traj_steps: list[dict], error_idx: int) -> dict:
    """
    分析 error 之后的 retry 行为，分类为:
    - adaptive_retry: 改变了参数/方法/工具后重试
    - backoff_retry:  加了 delay/sleep 后重试
    - plain_retry:    原样重试（相同工具相同参数）
    - no_retry:       没有重试

    同时检测:
    - resolved:       retry 后是否成功
    - retry_loop:     是否陷入重复失败循环
    """
    error_step = traj_steps[error_idx]

    # 找 error 对应的 tool_call（error_step 是 tool_result，前一个 tool_call 是触发它的）
    error_call_idx = error_idx - 1
    error_tool = ""
    error_params = {}
    if error_call_idx >= 0 and traj_steps[error_call_idx].get("step_type") == "tool_call":
        error_call = traj_steps[error_call_idx]
        error_tool = error_call.get("tool_name", "")
        error_params = error_call.get("input_params", {}) or {}

    # 在 error 后 1-8 步内找 retry 行为
    look_ahead = min(error_idx + 9, len(traj_steps))
    has_backoff = False
    has_param_change = False
    has_same_tool_call = False
    has_success_after = False
    consecutive_failures = 0

    for j in range(error_idx + 1, look_ahead):
        step = traj_steps[j]
        stype = step.get("step_type", "")

        if stype == "tool_call":
            tool = step.get("tool_name", "")
            params = step.get("input_params", {}) or {}

            if tool == error_tool:
                has_same_tool_call = True
                # 检查参数变化
                if _params_changed(error_params, params):
                    has_param_change = True
                # 检查是否是 sleep/backoff
                if tool == "exec":
                    cmd = str(params.get("command", ""))
                    if "sleep" in cmd or "delay" in cmd or "wait" in cmd:
                        has_backoff = True
            elif tool == "exec":
                cmd = str(params.get("command", ""))
                if "sleep" in cmd or "delay" in cmd or "wait" in cmd:
                    has_backoff = True

        elif stype == "tool_result":
            status = step.get("status", "")
            if status == "error" and has_same_tool_call:
                consecutive_failures += 1
            elif status == "success" and has_same_tool_call:
                has_success_after = True

    retry_loop = consecutive_failures >= 2

    # backoff-only（做了 sleep/delay 但没重试原工具）也算 backoff_retry
    if has_backoff:
        return {"recovery_kind": "backoff_retry", "retry_type": "backoff",
                "resolved": has_success_after, "retry_loop": retry_loop}
    elif has_param_change:
        return {"recovery_kind": "adaptive_retry", "retry_type": "param_change",
                "resolved": has_success_after, "retry_loop": retry_loop}
    elif has_same_tool_call:
        return {"recovery_kind": "plain_retry", "retry_type": "same_tool",
                "resolved": has_success_after, "retry_loop": retry_loop}

    return {"recovery_kind": "no_retry", "retry_type": "none",
            "resolved": False, "retry_loop": False}


def find_diverge_point(traj_steps: list[dict]) -> Optional[dict]:
    """
    找到轨迹里的最佳错误点（分叉点）。
    遍历所有 status=error 的 tool_result，
    按优先级选择最佳分叉点：
      1. intent_only=True（显式恢复意图但没执行）→ 最高价值
      2. adaptive_retry/backoff_retry + resolved（隐式恢复成功）→ 高价值
      3. adaptive_retry/backoff_retry（即使未 resolved）→ 中价值
      4. plain_retry + resolved → 中等价值
      5. 最后一个 error → 低价值
    """
    candidates = []
    for i, step in enumerate(traj_steps):
        if step.get("step_type") == "tool_result" and step.get("status") == "error":
            # 找紧随其后的 reasoning 或 output（都可能包含恢复意图声明）
            next_reasoning = None
            for j in range(i + 1, min(i + 5, len(traj_steps))):
                stype = traj_steps[j].get("step_type")
                if stype in ("reasoning", "output"):
                    next_reasoning = traj_steps[j]
                    break

            intent_only = _is_intent_only(traj_steps, i, next_reasoning)

            # 分析 error 后的 retry 行为
            retry_info = _classify_retry(traj_steps, i)

            candidates.append({
                "error_step":      step,
                "error_step_idx":  i,
                "error_type":      detect_error_type(step.get("error") or str(step.get("output", ""))),
                "next_reasoning":  next_reasoning,
                "intent_only":     intent_only,
                "retry_info":      retry_info,
            })

    if not candidates:
        return None

    # 优先级 1: intent_only=True（显式恢复意图但没有执行）→ 最高价值
    for c in candidates:
        if c["intent_only"]:
            return c

    # 优先级 2: 有 adaptive_retry 或 backoff_retry 且 resolved（隐式恢复成功）→ 高价值
    for c in candidates:
        ri = c.get("retry_info", {})
        if ri.get("recovery_kind") in ("adaptive_retry", "backoff_retry") and ri.get("resolved"):
            return c

    # 优先级 3: 有 adaptive_retry 或 backoff_retry（即使未 resolved）→ 中价值
    for c in candidates:
        ri = c.get("retry_info", {})
        if ri.get("recovery_kind") in ("adaptive_retry", "backoff_retry"):
            return c

    # 优先级 4: plain_retry 且 resolved → 中等价值
    for c in candidates:
        ri = c.get("retry_info", {})
        if ri.get("recovery_kind") == "plain_retry" and ri.get("resolved"):
            return c

    # 优先级 5: 返回最后一个 error（最接近恢复的）
    return candidates[-1]


# ─────────────────────────────────────────────────────────────────
# SFT messages 重建工具
# ─────────────────────────────────────────────────────────────────

def steps_to_assistant_text(steps: list[dict], max_chars: int = 6000) -> str:
    """把一段 trajectory steps 转成 assistant message 文本"""
    parts = []
    total = 0
    for step in steps:
        stype   = step.get("step_type", "")
        content = step.get("content") or ""

        if stype == "reasoning" and content:
            if content.startswith("[任务理解]"):
                continue
            chunk = f"<thinking>\n{content}\n</thinking>"
        elif stype == "tool_call":
            tool   = step.get("tool_name", "")
            params = step.get("input_params") or {}
            chunk  = f"<tool_call>\n{json.dumps({'tool': tool, 'params': params}, ensure_ascii=False)}\n</tool_call>"
        elif stype == "tool_result":
            status = step.get("status", "success")
            out    = step.get("output") or step.get("error") or ""
            if not isinstance(out, str):
                out = json.dumps(out, ensure_ascii=False)
            chunk  = f"<tool_result status=\"{status}\">\n{out[:2000]}\n</tool_result>"
        elif stype == "output" and content:
            chunk  = content
        else:
            continue

        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)

    return "\n".join(parts)


def build_dpo_prompt(
    system_prompt: str,
    user_input: str,
    steps_before_diverge: list[dict],
    error_step: dict,
) -> list[dict]:
    """
    构建 DPO 的 prompt 部分：
    system + user + assistant(到分叉点前) + tool_result(错误)
    """
    messages = [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": user_input},
    ]

    # 分叉点之前的 assistant 行为
    pre_text = steps_to_assistant_text(steps_before_diverge)
    if pre_text:
        messages.append({"role": "assistant", "content": pre_text})

    # 错误的 tool_result（这是触发分叉的信号）
    error_out = error_step.get("output") or error_step.get("error") or ""
    if not isinstance(error_out, str):
        error_out = json.dumps(error_out, ensure_ascii=False)
    messages.append({
        "role":    "tool",
        "content": f"[ERROR] {error_out[:1000]}",
    })

    return messages


# ─────────────────────────────────────────────────────────────────
# 配对逻辑
# ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are Hermes, an intelligent agent within the OpenClaw framework. "
    "You have access to a set of Skills and tools. Always reason step-by-step, "
    "stay within your assigned Skill boundaries, and handle errors gracefully."
)

DIVERGE_REASON_LABELS = {
    "rate_limit":   "rate_limit_retry",
    "timeout":      "timeout_recovery",
    "auth_error":   "auth_error_handling",
    "not_found":    "not_found_recovery",
    "server_error": "server_error_recovery",
    "other_error":  "generic_error_recovery",
    "unknown":      "generic_error_recovery",
}


def _is_plain_retry_bad_for_error_type(error_type: str) -> bool:
    """
    判断对某类错误，原样重试是否是坏行为。
    GPT 建议：不能全局认为 plain_retry 一定 rejected，要按 error_type 区分。
    """
    # 429/rate limit: immediate plain retry = bad (应该 backoff)
    if error_type == "rate_limit":
        return True
    # tab not found: same tab_id retry = bad
    if error_type == "not_found":
        return True
    # auth error: plain retry = bad (不会自愈)
    if error_type == "auth_error":
        return True
    # parse error / schema error: plain retry = bad (需要改参数)
    if error_type == "parse_error":
        return True
    # server error: one plain retry may be acceptable, but repeated = bad
    # 返回 False，让 retry_loop 来决定
    if error_type in ("server_error", "timeout"):
        return False
    # other_error: 保守返回 False
    return False


def _extract_recovery_actions(traj_steps: list[dict], error_idx: int, max_steps: int = 8) -> list[dict]:
    """
    从 error 之后的恢复行为中提取 tool_call 步骤（不包含 tool_result）。
    GPT 建议：chosen 不要包含未来 tool_result，只包含模型可控的 assistant 行为。
    """
    actions = []
    for j in range(error_idx + 1, min(error_idx + 1 + max_steps, len(traj_steps))):
        step = traj_steps[j]
        stype = step.get("step_type", "")
        if stype == "tool_call":
            actions.append(step)
        elif stype == "output" and step.get("content"):
            # output 也可以作为 chosen 的一部分（如 "let me retry those two"）
            actions.append(step)
        # 跳过 tool_result — 不泄漏未来信息
    return actions


def _make_synthetic_rejected(error_step: dict, fail_traj: list[dict], error_idx: int) -> tuple[str, str]:
    """
    构造一个 synthetic rejected：原样重试导致错误的 tool_call。
    返回 (rejected_text, rejected_source)。
    """
    # 找到导致错误的 tool_call（error_step 前一步）
    if error_idx > 0 and fail_traj[error_idx - 1].get("step_type") == "tool_call":
        bad_call = fail_traj[error_idx - 1]
        tool = bad_call.get("tool_name", "")
        params = bad_call.get("input_params") or {}
        rejected_text = f"<tool_call>\n{json.dumps({'tool': tool, 'params': params}, ensure_ascii=False)}\n</tool_call>"
        return rejected_text, "synthetic_plain_retry"
    
    # fallback: 用 error 后的前几步
    rejected_text = steps_to_assistant_text(fail_traj[error_idx + 1: error_idx + 3])
    return rejected_text, "error_after_steps"


def try_pair_success_failure(
    success: dict,
    failure: dict,
    min_similarity: float = 0.5,
) -> Optional[dict]:
    """
    尝试把一个成功轨迹和一个失败轨迹配成 DPO 对。
    返回 None 表示不适合配对。
    
    核心原则（GPT 修正版）：
    - prompt = error 之前上下文 + error tool_result
    - chosen = 同一 session error 后的正确恢复行为（只含 tool_call，不含 tool_result）
    - rejected = synthetic plain_retry（原样重试导致错误的调用）或真实坏行为
    - 跨 session 只用于 global pair（低质量）
    """
    sim = tools_jaccard(success, failure)
    if sim < min_similarity:
        return None

    fail_traj   = _get_steps(failure)
    diverge     = find_diverge_point(fail_traj)

    if not diverge:
        # 失败轨迹里没有明确的错误点，降级为全局对比
        return _make_global_pair(success, failure, sim)

    error_idx   = diverge["error_step_idx"]
    error_type  = diverge["error_type"]
    intent_only = diverge["intent_only"]

    # 根据恢复行为类型决定质量
    retry_info = diverge.get("retry_info", {})
    recovery_kind = retry_info.get("recovery_kind", "no_retry")
    resolved = retry_info.get("resolved", False)
    retry_loop = retry_info.get("retry_loop", False)

    # ── 质量分级 ──
    # 检查 failed_item_eventually_resolved（失败项是否最终被补齐）
    item_resolved = _check_failed_item_resolved(fail_traj, error_idx)
    if retry_loop and not resolved:
        pair_quality = "low"
    elif intent_only:
        pair_quality = "high"
    elif recovery_kind in ("adaptive_retry", "backoff_retry") and resolved:
        # rate_limit 场景：只有 failed_item_eventually_resolved=True 才是 high
        # 否则只是 skip-and-abandon，降为 medium
        if error_type == "rate_limit" and not item_resolved:
            pair_quality = "medium"
        else:
            pair_quality = "high"
    elif recovery_kind in ("adaptive_retry", "backoff_retry"):
        pair_quality = "medium"
    elif recovery_kind == "plain_retry" and resolved:
        pair_quality = "medium"
    else:
        pair_quality = "medium"

    # 分叉点之前的步骤（用于构建 prompt）
    steps_before = fail_traj[:error_idx]
    error_step   = diverge["error_step"]

    # ── 构建 chosen 和 rejected ──
    # GPT 原则：chosen 和 rejected 必须来自同一个 error state 的不同恢复行为
    
    # chosen: 从 error 之后的恢复行为中提取 tool_call（不含 tool_result）
    recovery_actions = _extract_recovery_actions(fail_traj, error_idx, max_steps=8)
    chosen_text = steps_to_assistant_text(recovery_actions)
    
    # rejected: 根据情况构造
    if intent_only and diverge.get("next_reasoning"):
        # 策略 1: intent_only — rejected = 只说要做但不做的声明
        next_r = diverge["next_reasoning"]
        rejected_text = (next_r.get("content") or next_r.get("text") or "")[:2000]
        rejected_source = "intent_without_action"
        preference_strength = "strong"
    elif retry_loop:
        # 策略 2: retry_loop — rejected = 重复失败的 retry 序列
        rejected_text = steps_to_assistant_text(fail_traj[error_idx + 1: error_idx + 5])
        rejected_source = "real_retry_loop"
        preference_strength = "strong"
    elif _is_plain_retry_bad_for_error_type(error_type):
        # 策略 3: 对 429/tab_not_found/auth 等错误，synthetic plain_retry 是坏行为
        rejected_text, rejected_source = _make_synthetic_rejected(error_step, fail_traj, error_idx)
        # rate_limit: 只有 failed_item_eventually_resolved=True 才是 strong
        # 否则只是 skip-and-abandon，降为 medium
        if error_type == "rate_limit":
            item_resolved = _check_failed_item_resolved(fail_traj, error_idx)
            preference_strength = "strong" if (item_resolved and recovery_kind in ("adaptive_retry", "backoff_retry")) else "medium"
        else:
            preference_strength = "strong" if recovery_kind in ("adaptive_retry", "backoff_retry") else "medium"
    else:
        # 策略 4: server_error/timeout 等 — plain retry 可能合理，用 medium
        rejected_text, rejected_source = _make_synthetic_rejected(error_step, fail_traj, error_idx)
        preference_strength = "medium"

    if not chosen_text or not rejected_text:
        return _make_global_pair(success, failure, sim)

    if chosen_text == rejected_text:
        return None   # 行为完全一样，配对无意义

    prompt = build_dpo_prompt(
        SYSTEM_PROMPT,
        _get_user_input(failure),
        steps_before,
        error_step,
    )

    # ── 决定 chosen_source ──
    chosen_source = "same_session_recovery"  # 默认：同 session 的恢复行为

    return {
        "pair_id":        str(uuid4()),
        "pair_type":      "success_vs_failure_at_error",
        "pair_quality":   pair_quality,
        "task_type":      compute_task_fingerprint(failure),
        "diverge_reason": DIVERGE_REASON_LABELS.get(error_type, "generic_error_recovery"),
        "error_type":     error_type,
        "intent_only_failure": intent_only,
        "tool_similarity": round(sim, 3),
        "prompt":         prompt,
        "chosen":         chosen_text,
        "rejected":       rejected_text,
        "metadata": {
            "success_session":  success.get("session_id"),
            "failure_session":  failure.get("session_id"),
            "success_score":    success.get("value_score", {}).get("total"),
            "failure_score":    failure.get("value_score", {}).get("total"),
            "success_steps":    success.get("stats", {}).get("steps"),
            "failure_steps":    failure.get("stats", {}).get("steps"),
            "error_at_step":    error_idx + 1,
            "recovery_kind":    recovery_kind,
            "retry_type":       retry_info.get("retry_type", "none"),
            "resolved":         resolved,
            "retry_loop":       retry_loop,
            "chosen_source":    chosen_source,
            "rejected_source":  rejected_source,
            "preference_strength": preference_strength,
            "pair_construction": "same_session_recovery_vs_synthetic_plain_retry",
            "synthetic_rejected": rejected_source.startswith("synthetic"),
            "failed_item_eventually_resolved": _check_failed_item_resolved(fail_traj, error_idx),
            "near_duplicate_group_id": _make_near_dedup_id(failure, diverge),
        },
    }


def _make_global_pair(success: dict, failure: dict, sim: float) -> Optional[dict]:
    """
    降级配对：没有明确分叉点时，用完整轨迹做全局对比。
    价值低于分叉点配对，但仍有训练意义。
    """
    succ_text = steps_to_assistant_text(
        _get_steps(success)[:12]
    )
    fail_text = steps_to_assistant_text(
        _get_steps(failure)[:12]
    )
    if not succ_text or not fail_text or succ_text == fail_text:
        return None

    return {
        "pair_id":        str(uuid4()),
        "pair_type":      "global_success_vs_failure",
        "pair_quality":   "low",
        "task_type":      compute_task_fingerprint(failure),
        "diverge_reason": "unknown",
        "error_type":     "none",
        "intent_only_failure": False,
        "tool_similarity": round(sim, 3),
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": _get_user_input(failure)},
        ],
        "chosen":   succ_text,
        "rejected": fail_text,
        "metadata": {
            "success_session": success.get("session_id"),
            "failure_session": failure.get("session_id"),
        },
    }


def try_pair_quality(high: dict, low: dict, min_similarity: float = 0.6) -> Optional[dict]:
    """
    同为成功轨迹但质量不同时的软对比配对。
    chosen = 高质量成功，rejected = 低质量成功（步骤多/工具调用多）。
    """
    sim = tools_jaccard(high, low)
    if sim < min_similarity:
        return None

    h_score = high.get("value_score", {}).get("total", 0)
    l_score = low.get("value_score", {}).get("total",  0)
    if abs(h_score - l_score) < 0.05:
        return None   # 质量差距太小，没有配对意义

    h_text = steps_to_assistant_text(_get_steps(high)[:10])
    l_text = steps_to_assistant_text(_get_steps(low)[:10])

    if not h_text or not l_text or h_text == l_text:
        return None

    return {
        "pair_id":        str(uuid4()),
        "pair_type":      "quality_comparison",
        "pair_quality":   "medium",
        "task_type":      compute_task_fingerprint(high),
        "diverge_reason": "efficiency_difference",
        "error_type":     "none",
        "intent_only_failure": False,
        "tool_similarity": round(sim, 3),
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": _get_user_input(high)},
        ],
        "chosen":   h_text,
        "rejected": l_text,
        "metadata": {
            "high_session":  high.get("session_id"),
            "low_session":   low.get("session_id"),
            "score_gap":     round(h_score - l_score, 4),
            "high_steps":    high.get("stats", {}).get("steps"),
            "low_steps":     low.get("stats",  {}).get("steps"),
        },
    }


# ─────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────

def run_pairing(
    candidates: list[dict],
    min_similarity: float = 0.5,
    max_pairs_per_cluster: int = 10,
) -> list[dict]:
    """
    对所有 DPO 候选进行自动配对。
    """
    # 过滤掉无配对价值的垃圾候选（无工具调用、步骤太少）
    before_filter = len(candidates)
    candidates = [c for c in candidates if not _is_trivial_candidate(c)]
    filtered = before_filter - len(candidates)
    if filtered:
        print(f"过滤掉 {filtered} 个垃圾候选（无工具/步骤太少）", file=sys.stderr)

    # 按任务指纹分组
    clusters: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        fp = compute_task_fingerprint(c)
        clusters[fp].append(c)

    pairs: list[dict] = []

    for fp, cluster in clusters.items():
        # 干净的成功轨迹（无错误恢复）→ 作为 DPO 的 chosen 侧
        # 三池分离：干净成功 / 含错误恢复 / 真正失败
        # 1. clean_success_pool: 无错误恢复的成功轨迹（用于 quality_comparison 和作为配对上下文参考）
        clean_success_pool = [
            c for c in cluster
            if c.get("outcome") in ("success", "likely_success")
            and c.get("training_use") != "recovery_training"
        ]
        # 2. recovery_pool: 含错误恢复的轨迹（最终可能成功也可能失败）
        #    这些轨迹有 diverge point（错误点），是 DPO 的主要来源
        #    注意：它们不是"失败轨迹"，而是"含错误恢复轨迹"
        recovery_pool = [
            c for c in cluster
            if c.get("training_use") == "recovery_training"
        ]
        # 3. failure_pool: 真正失败的轨迹
        failure_pool = [c for c in cluster if c.get("outcome") in
                     ("failure", "likely_failure", "unknown")]
        # 合并 recovery + failure 作为 DPO error-state 配对的候选池
        error_state_pool = recovery_pool + failure_pool

        cluster_pairs: list[dict] = []

        # ── error-state 配对（最高价值） ──
        # recovery_pool 中的轨迹有 error + recovery，可以产生 success_vs_failure_at_error
        # clean_success_pool 仅用于 tool_similarity 匹配和 quality_comparison
        # 注意：chosen/rejected 都来自 recovery_pool 自身（same_session_recovery），
        #       不从 clean_success_pool 取行为
        successes_sorted = sorted(
            clean_success_pool,
            key=lambda x: x.get("value_score", {}).get("total", 0),
            reverse=True,
        )
        error_state_sorted = sorted(
            error_state_pool,
            key=lambda x: x.get("value_score", {}).get("total", 0),
        )

        # 每个 error_state 最多配 1 个 clean success（相似度最高的）
        # 避免 chosen/rejected 完全相同只是 success_session 不同的重复 pair
        for f in error_state_sorted[:10]:    # 最多10个恢复/失败轨迹
            best_pair = None
            best_sim = 0
            for s in successes_sorted[:5]:   # 最多5个成功轨迹
                pair = try_pair_success_failure(s, f, min_similarity)
                if pair and pair.get("tool_similarity", 0) > best_sim:
                    best_pair = pair
                    best_sim = pair.get("tool_similarity", 0)
            if best_pair:
                cluster_pairs.append(best_pair)
                if len(cluster_pairs) >= max_pairs_per_cluster:
                    break

        # ── 同为成功但质量不同的软对比 ──
        if len(clean_success_pool) >= 2 and len(cluster_pairs) < max_pairs_per_cluster:
            for h, l in combinations(successes_sorted, 2):
                if h.get("value_score", {}).get("total", 0) > \
                   l.get("value_score", {}).get("total", 0):
                    pair = try_pair_quality(h, l, min_similarity)
                    if pair:
                        cluster_pairs.append(pair)
                if len(cluster_pairs) >= max_pairs_per_cluster:
                    break

        pairs.extend(cluster_pairs)

    # ── 去重 ──
    # 1. 精确去重：prompt + chosen + rejected 完全相同的 pair 只保留第一条
    seen_signatures = set()
    unique_pairs = []
    for p in pairs:
        sig = (
            p.get("chosen", ""),
            p.get("rejected", ""),
            p.get("pair_quality", ""),
        )
        if sig not in seen_signatures:
            seen_signatures.add(sig)
            unique_pairs.append(p)
    pairs = unique_pairs

    # 2. 近重复组去重：同一个 near_duplicate_group_id 最多保留 1 条
    # 优先保留 preference_strength=strong 和 failed_item_eventually_resolved=True 的
    group_pairs = defaultdict(list)
    for p in pairs:
        gid = p.get("metadata", {}).get("near_duplicate_group_id", "")
        if gid:
            group_pairs[gid].append(p)
    
    deduped = []
    seen_groups = set()
    for p in pairs:
        gid = p.get("metadata", {}).get("near_duplicate_group_id", "")
        if gid:
            if gid in seen_groups:
                continue
            group = group_pairs[gid]
            group.sort(key=lambda x: (
                x.get("metadata", {}).get("preference_strength") == "strong",
                x.get("metadata", {}).get("failed_item_eventually_resolved", False),
            ), reverse=True)
            # high pair 每个 group 只保留 1 条；其他类型保留前 2 条
            max_per_group = 1 if group[0].get("pair_quality") == "high" else 2
            deduped.extend(group[:max_per_group])
            seen_groups.add(gid)
        else:
            deduped.append(p)
    
    return deduped


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def _load_jsonl(path: Path, label: str = "") -> list[dict]:
    """读取 JSONL 文件，返回 dict 列表"""
    result: list[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        result.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except OSError as e:
        print(f"[警告] 无法读取 {path}: {e}", file=sys.stderr)
    if label:
        print(f"  从 {label} 读入 {len(result)} 条", file=sys.stderr)
    return result


def _is_trivial_candidate(candidate: dict) -> bool:
    """判断是否是垃圾候选（无工具调用、步骤太少等），无配对价值"""
    steps = _get_steps(candidate)
    # 没有工具调用 → 无配对价值
    tool_calls = [s for s in steps if s.get("step_type") == "tool_call"]
    if not tool_calls:
        return True
    # 步骤太少
    if len(steps) <= 4:
        return True
    # 只有 output/reasoning 步骤，没有实际的工具交互
    unique_tools = list({s.get("tool_name", "") for s in tool_calls if s.get("tool_name")})
    if not unique_tools:
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(
        description="从 DPO 候选自动提取 DPO 配对",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input",  "-i", nargs="+", required=True,
                    help="一个或多个输入文件路径（支持同时传 dpo_candidate.jsonl 和 recovery_training.jsonl）")
    ap.add_argument("--output", "-o", required=True,
                    help="输出 dpo_pairs.jsonl 路径")
    ap.add_argument("--min-similarity", type=float, default=0.5,
                    help="工具集 Jaccard 相似度阈值（低于此值不配对）")
    ap.add_argument("--max-pairs-per-cluster", type=int, default=10,
                    help="每个任务类型最多输出多少配对")
    ap.add_argument("--high-value-only", action="store_true",
                    help="只输出 pair_quality=high 的配对（intent_only_failure）")
    args = ap.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 读入多个文件的所有候选
    candidates: list[dict] = []
    for input_path_str in args.input:
        input_path = Path(input_path_str)
        label = input_path.name
        if input_path.is_file():
            candidates.extend(_load_jsonl(input_path, label))
        else:
            print(f"[警告] 跳过不存在的文件: {input_path}", file=sys.stderr)

    print(f"读入总共 {len(candidates)} 个原始候选", file=sys.stderr)

    # 配对
    pairs = run_pairing(
        candidates,
        min_similarity        = args.min_similarity,
        max_pairs_per_cluster = args.max_pairs_per_cluster,
    )

    if args.high_value_only:
        pairs = [p for p in pairs if p["pair_quality"] == "high"]

    # 统计
    by_type    = defaultdict(int)
    by_quality = defaultdict(int)
    by_reason  = defaultdict(int)
    intent_only_count = 0
    for p in pairs:
        by_type[p["pair_type"]] += 1
        by_quality[p["pair_quality"]] += 1
        by_reason[p["diverge_reason"]] += 1
        if p.get("intent_only_failure"):
            intent_only_count += 1

    # 写出
    with open(output_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n配对完成: {len(pairs)} 对 → {output_path}", file=sys.stderr)
    print(f"\n按类型:", file=sys.stderr)
    for k, v in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {v:4d}  {k}", file=sys.stderr)
    print(f"\n按质量:", file=sys.stderr)
    for k, v in sorted(by_quality.items(), key=lambda x: -x[1]):
        print(f"  {v:4d}  {k}", file=sys.stderr)
    print(f"\n按分叉原因:", file=sys.stderr)
    for k, v in sorted(by_reason.items(), key=lambda x: -x[1]):
        print(f"  {v:4d}  {k}", file=sys.stderr)
    print(f"\n'说了不做'高价值对: {intent_only_count}", file=sys.stderr)


if __name__ == "__main__":
    main()
