#!/usr/bin/env python3
"""
将 OpenClaw trajectory.jsonl 事件流聚合成 extractor 可用的 session-level JSONL。

用法:
  python3 aggregate_sessions.py \
    --input ~/.openclaw/agents/main/sessions/ \
    --output ~/Downloads/session_aggregated.jsonl

每个输出行是一个完整的 session JSON，包含:
  - session_id, user_input, final_output
  - trajectory (步骤数组)
  - created_at, agent, model_version
  - metadata, session_outcome
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# ─── 工具 ────────────────────────────────────────────────────────

def parse_ts(ts_str: str | None) -> datetime | None:
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        return None


def split_thinking(text: str) -> tuple[str, str]:
    """
    把 assistantText 里的 <thinking>...</thinking> 块拆出来。
    返回 (thinking_content, output_content)
    thinking_content 可能为空字符串。
    """
    if not text:
        return "", ""
    thinking_parts = []
    output_parts = []
    last = 0
    for m in re.finditer(r"<thinking>(.*?)</thinking>", text, re.DOTALL):
        before = text[last:m.start()].strip()
        if before:
            output_parts.append(before)
        thinking_parts.append(m.group(1).strip())
        last = m.end()
    tail = text[last:].strip()
    if tail:
        output_parts.append(tail)
    return "\n\n".join(thinking_parts), "\n\n".join(output_parts)


def extract_user_task(prompt: str) -> str:
    """
    从 context.compiled / prompt.submitted 的 prompt 字段里提取真正的用户任务。
    OpenClaw 的 prompt 通常是完整的多轮对话文本，实际任务在最后一个 [Subagent Task] 或
    Human/User 标记之后。尽量提取最后一条用户消息，而不是整个 prompt。
    """
    # 策略1：找最后一个 [Subagent Task] 块
    m = re.search(r"\[Subagent Task\](.*?)(?:\[|$)", prompt, re.DOTALL)
    if m:
        return m.group(1).strip()[:5000]

    # 策略2：找最后一个 Human: / User: 标记
    parts = re.split(r"\n(?:Human|User):\s*", prompt)
    if len(parts) > 1:
        last_user = parts[-1].split("\nAssistant:")[0].strip()
        if last_user:
            return last_user[:5000]

    # 策略3：直接截取（兜底）
    return prompt[:5000]


# ─── 核心聚合逻辑 ─────────────────────────────────────────────────


# ─── 从原始 .jsonl session 文件补全 tool 数据 ──────────────────────

def _find_companion_jsonl(trajectory_path: Path) -> str | None:
    """
    找到与 .trajectory.jsonl 对应的原始 .jsonl session 文件。
    可能被重命名为 .jsonl.deleted.*，仍可读取。
    """
    session_dir = trajectory_path.parent
    # 从文件名提取 session ID（UUID 格式）
    fname = trajectory_path.name
    # .trajectory.jsonl → 去掉后缀得到 session_id
    base = fname.replace(".trajectory.jsonl", "")
    if len(base) < 8:
        return None
    # 查找同名 .jsonl 或 .jsonl.deleted.*
    for candidate in sorted(session_dir.iterdir()):
        cname = candidate.name
        if cname == f"{base}.jsonl":
            return str(candidate)
        if cname.startswith(base) and ".jsonl.deleted." in cname:
            return str(candidate)
    return None


def _parse_companion_jsonl(jsonl_path: str) -> dict:
    """
    解析原始 .jsonl session 文件，提取 tool 调用信息。
    返回:
      {
        "tool_uses": { callId: {"name": ..., "input": {...}} },
        "tool_results": { toolCallId: {"content": "...", "is_error": bool} },
        "reasonings": [(seq, text), ...],  # 按顺序的 reasoning 文本
      }
    """
    tool_uses: dict[str, dict] = {}
    tool_results: dict[str, dict] = {}
    reasonings: list[tuple[int, str]] = []
    try:
        with open(jsonl_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return {"tool_uses": {}, "tool_results": {}, "reasonings": []}

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if evt.get("type") != "message":
            continue
        msg = evt.get("message", {})
        role = msg.get("role", "")
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") in ("toolCall", "tool_use"):
                call_id = part.get("id") or part.get("callId", "")
                if call_id:
                    tool_uses[call_id] = {
                        "name": part.get("name", ""),
                        "input": part.get("input") or part.get("arguments", {}),
                    }
            elif part.get("type") == "thinking":
                text = part.get("thinking", "")
                if text:
                    reasonings.append((len(reasonings), text))
            elif part.get("type") == "text" and role == "assistant":
                text = part.get("text", "")
                if text.strip():
                    reasonings.append((len(reasonings), text))
        if role == "toolResult":
            tool_call_id = msg.get("toolCallId", "")
            if tool_call_id:
                texts = []
                for part in content:
                    if part.get("type") == "text":
                        texts.append(part.get("text", ""))
                    elif part.get("type") == "toolResult":
                        inner = part.get("content", [])
                        if isinstance(inner, list):
                            for ip in inner:
                                if isinstance(ip, dict):
                                    texts.append(ip.get("text", ""))
                full_text = "\n".join(texts)
                # 检测是否是错误
                is_error = _is_error_result(full_text)
                tool_results[tool_call_id] = {
                    "content": full_text,
                    "is_error": is_error,
                }
    return {
        "tool_uses": tool_uses,
        "tool_results": tool_results,
        "reasonings": reasonings,
    }


def _is_error_result(text: str) -> bool:
    """检测 tool result 是否包含错误信号。"""
    if not text or not text.strip():
        return False
    # JSON-level checks
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            if parsed.get("isError") is True or parsed.get("is_error") is True:
                return True
            if parsed.get("error"):
                return True
            if parsed.get("error_message"):
                return True
            if parsed.get("status") == "error":
                return True
            if parsed.get("success") or parsed.get("block_id") or parsed.get("blocks"):
                return False
    except (json.JSONDecodeError, TypeError):
        pass
    # String-level error patterns
    low = text.lower()
    error_pats = [
        # --- HTTP / API 错误 ---
        "429", "rate limit", "rate_limit", "too many requests",
        "timeout", "timed out", "connection timeout", "read timeout",
        "unauthorized", "forbidden", "permission denied",
        "connection refused",
        "internal server error", "bad gateway",
        # --- 通用技术错误 ---
        "exception", "traceback",
        "cannot read properties", "err_aborted",
        "not found", "does not exist", "no such file",
        # --- 中文错误关键词 ---
        "连接超时", "连接被拒绝", "连接失败",
        "请求超时", "请求被拒绝", "请求失败",
        "权限不足", "权限被拒绝", "未授权",
        "找不到", "不存在", "无此文件",
        "服务器错误", "服务不可用", "内部错误",
        "限流", "频率限制", "访问过快",
        "异常", "错误", "失败", "出错",
    ]
    for pat in error_pats:
        if pat in low:
            return True
    return False


def _build_backfill_maps(companion_data: dict, trajectory_tools: list[dict]) -> tuple[list, list]:
    """
    将 companion .jsonl 里的 tool call/result 与 trajectory 中的 tools 按顺序对齐。
    trajectory 里没有 callId 和 result_callId，只能按工具名 + 出现序号匹配。
    返回：两个清单，与 trajectory_tools 一一对应的 input_params 和 result_map。
    """
    tool_uses = companion_data.get("tool_uses", {})
    tool_results = companion_data.get("tool_results", {})
    if not tool_uses and not tool_results:
        return [], []

    # 构建 companion 中有序的 tool use 列表（按 callId 顺序不可靠，用 list）
    # 但实际上 JSON 里 callId 是 UUID，遍历 dict 即可
    use_items = list(tool_uses.values())
    result_items = list(tool_results.items())

    # 按工具名+callId 组织 companion 数据
    # tool_uses 按 callId 存储，但要按工具名+出现顺序对齐 trajectory
    # 将 tool_uses 的 dict items 转为按名称分组的列表
    grouped_calls: dict[str, list[dict]] = {}
    for cid, item in tool_uses.items():
        name = item.get("name", "")
        if name not in grouped_calls:
            grouped_calls[name] = []
        grouped_calls[name].append({**item, "_call_id": cid})

    call_counters: dict[str, int] = {}
    inputs = []
    results = []

    for traj_tool in trajectory_tools:
        tname = traj_tool.get("tool_name", "")
        if tname:
            idx = call_counters.get(tname, 0)
            call_counters[tname] = idx + 1
            calls = grouped_calls.get(tname, [])
            if idx < len(calls):
                matched = calls[idx]
                inputs.append(matched.get("input", {}))
                call_id = matched.get("_call_id", "")
                if call_id and call_id in tool_results:
                    rd = tool_results[call_id]
                    results.append({
                        "content": rd.get("content", ""),
                        "is_error": rd.get("is_error", False),
                    })
                else:
                    results.append({"content": None, "is_error": False})
            else:
                inputs.append({})
                results.append({"content": None, "is_error": False})
        else:
            inputs.append({})
            results.append({"content": None, "is_error": False})

    return inputs, results


def aggregate_trajectory_file(path: Path) -> list[dict]:
    """聚合一个 .trajectory.jsonl 文件为 session-level JSON 列表"""
    events: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []

    if not events:
        return []

    # ── 按 sessionId 分组 ──
    sessions: dict[str, list[dict]] = defaultdict(list)
    for ev in events:
        sid = ev.get("sessionId") or ev.get("traceId") or ""
        if sid:
            sessions[sid].append(ev)

    results = []
    for sid, evs in sessions.items():
        evs.sort(key=lambda e: e.get("seq", e.get("sourceSeq", 0)))

        first_ev = evs[0]
        last_ev  = evs[-1]

        agent         = first_ev.get("agentId") or "openclaw"
        model_version = first_ev.get("modelId") or last_ev.get("modelId")
        created_at    = parse_ts(first_ev.get("ts"))
        session_key   = first_ev.get("sessionKey", "")

        if ":" in session_key:
            parts = session_key.split(":")
            if len(parts) >= 2 and parts[0] == "agent":
                agent = parts[1]

        # ── 提取 user_input（第一轮的真实任务，不是完整 prompt） ──
        user_input: str | None = None
        for ev in evs:
            if ev.get("type") in ("context.compiled", "prompt.submitted"):
                raw_prompt = (ev.get("data") or {}).get("prompt") or ""
                if raw_prompt:
                    user_input = extract_user_task(raw_prompt)
                    break

        # ── 提取 final_output（最后一个 model.completed 的最后一条 assistantText） ──
        final_output: str | None = None
        for ev in reversed(evs):
            if ev.get("type") == "model.completed":
                at = (ev.get("data") or {}).get("assistantTexts") or []
                if at and isinstance(at[-1], str):
                    _, out = split_thinking(at[-1])
                    final_output = out[:10000] or at[-1][:10000]
                break

        # ── 收集所有轮次的 trace.artifacts（工具调用元数据） ──
        # 按 seq 排序，保留每轮的 toolMetas
        artifacts_per_round: list[dict] = []
        for ev in evs:
            if ev.get("type") == "trace.artifacts":
                artifacts_per_round.append(ev)

        # 建立 seq → toolMetas 的映射，方便与 model.completed 对应
        # OpenClaw 的 trace.artifacts 通常和对应的 model.completed 相邻
        artifacts_by_seq: dict[int, list[dict]] = {}
        for ev in artifacts_per_round:
            seq = ev.get("seq", 0)
            artifacts_by_seq[seq] = (ev.get("data") or {}).get("toolMetas") or []

        # ── 尝试加载 companion .jsonl 补全 tool 数据 ──
        companion_data = _parse_companion_jsonl(companion_path) if (
            companion_path := _find_companion_jsonl(path)
        ) else {"tool_uses": {}, "tool_results": {}, "reasonings": []}

        # ── 构建 trajectory 步骤 ──
        trajectory: list[dict] = []
        step_id    = 0
        round_num  = 0
        model_seqs: list[int] = []

        for ev in evs:
            ev_type = ev.get("type", "")
            data    = ev.get("data") or {}
            ts_str  = (parse_ts(ev.get("ts")) or datetime.min).isoformat()
            seq     = ev.get("seq", 0)

            # ── 用户输入轮次：只记录 round_num，不写入 trajectory ──
            # （避免把完整 prompt 重复进 assistant message）
            if ev_type in ("context.compiled", "prompt.submitted"):
                round_num += 1
                # 只在第一轮记录一个 reasoning 步骤（代表任务理解）
                if round_num == 1 and user_input:
                    step_id += 1
                    trajectory.append({
                        "step_id":   step_id,
                        "step_type": "reasoning",
                        "timestamp": ts_str,
                        "content":   f"[任务理解] {user_input[:500]}",
                    })

            # ── 模型输出：拆分 thinking 和 output ──
            elif ev_type == "model.completed":
                model_seqs.append(seq)
                at = data.get("assistantTexts") or []
                if not at:
                    continue
                last_text = at[-1] if isinstance(at[-1], str) else ""
                thinking, output = split_thinking(last_text)

                # reasoning 步骤（thinking 块）
                if thinking:
                    step_id += 1
                    trajectory.append({
                        "step_id":   step_id,
                        "step_type": "reasoning",
                        "timestamp": ts_str,
                        "content":   thinking[:8000],
                    })

                # 找本轮紧邻的 trace.artifacts（seq 最近的那个）
                nearby_seqs = sorted(
                    [s for s in artifacts_by_seq if abs(s - seq) <= 5],
                    key=lambda s: abs(s - seq)
                )
                if nearby_seqs:
                    tool_metas = artifacts_by_seq.pop(nearby_seqs[0])
                    for tm in tool_metas:
                        tool_name = tm.get("toolName") or tm.get("tool") or ""
                        if not tool_name:
                            continue
                        call_id = f"{sid}_{step_id + 1}"
                        # 从 tm 里提取 input_params
                        input_params = {}
                        raw_input = tm.get("input") or tm.get("args") or tm.get("params") or {}
                        if isinstance(raw_input, dict):
                            input_params = raw_input
                        elif isinstance(raw_input, str):
                            try:
                                input_params = json.loads(raw_input)
                            except Exception:
                                input_params = {"raw": raw_input[:500]}

                        step_id += 1
                        trajectory.append({
                            "step_id":      step_id,
                            "step_type":    "tool_call",
                            "timestamp":    ts_str,
                            "tool_name":    tool_name,
                            "skill_context": None,
                            "input_params": input_params,
                            "call_id":      call_id,
                            "metadata":     {"call_id": call_id},
                        })

                        # tool result
                        meta = tm.get("meta") or tm.get("output") or tm.get("result") or ""
                        result_status = "error" if (tm.get("isError") or tm.get("error") or _is_error_result(meta if isinstance(meta, str) else str(meta))) else "success"
                        step_id += 1
                        trajectory.append({
                            "step_id":        step_id,
                            "step_type":      "tool_result",
                            "timestamp":      ts_str,
                            "result_call_id": call_id,
                            "status":         result_status,
                            "output":         meta[:5000] if isinstance(meta, str) else str(meta)[:5000],
                            "error":          str(tm.get("error", ""))[:500] if tm.get("error") else None,
                        })

                # output 步骤（非工具调用的纯文字回复）
                if output:
                    step_id += 1
                    trajectory.append({
                        "step_id":   step_id,
                        "step_type": "output",
                        "timestamp": ts_str,
                        "content":   output[:8000],
                    })

        # ── 剩余未匹配的 trace.artifacts（兜底：没有对应 model.completed 的工具调用） ──
        for seq_key in sorted(artifacts_by_seq):
            for tm in artifacts_by_seq[seq_key]:
                tool_name = tm.get("toolName") or tm.get("tool") or ""
                if not tool_name:
                    continue
                call_id = f"{sid}_{step_id + 1}"
                step_id += 1
                trajectory.append({
                    "step_id":      step_id,
                    "step_type":    "tool_call",
                    "timestamp":    None,
                    "tool_name":    tool_name,
                    "skill_context": None,
                    "input_params": {},
                    "call_id":      call_id,
                    "metadata":     {"call_id": call_id},
                })
                meta = tm.get("meta") or tm.get("output") or ""
                result_status = "error" if (tm.get("isError") or tm.get("error") or _is_error_result(meta if isinstance(meta, str) else str(meta))) else "success"
                step_id += 1
                trajectory.append({
                    "step_id":        step_id,
                    "step_type":      "tool_result",
                    "timestamp":      None,
                    "result_call_id": call_id,
                    "status":         result_status,
                    "output":         meta[:5000] if isinstance(meta, str) else str(meta)[:5000],
                    "error":          str(tm.get("error", ""))[:500] if tm.get("error") else None,
                })

        # ── 从 companion .jsonl backfill 空的 tool params 和 results ──
        if companion_data["tool_uses"] or companion_data["tool_results"]:
            tool_steps = [s for s in trajectory if s["step_type"] == "tool_call"]
            backfilled_inputs, backfilled_results = _build_backfill_maps(companion_data, tool_steps)
            result_idx = 0
            for i, s in enumerate(trajectory):
                if s["step_type"] == "tool_call":
                    call_idx = len([t for t in trajectory[:i] if t["step_type"] == "tool_call"])
                    if call_idx < len(backfilled_inputs) and backfilled_inputs[call_idx]:
                        if not s.get("input_params") or not any(v for v in s["input_params"].values() if v):
                            s["input_params"] = backfilled_inputs[call_idx]
                elif s["step_type"] == "tool_result":
                    if result_idx < len(backfilled_results):
                        br = backfilled_results[result_idx]
                        if not s.get("output") or s.get("output") is None:
                            if br.get("content"):
                                s["output"] = br["content"]
                            if br.get("is_error"):
                                s["status"] = "error"
                        result_idx += 1

        # ── 重新检查 tool error（backfill 后可能有新 error） ──
        has_tool_error = any(
            s.get("step_type") == "tool_result" and s.get("status") == "error"
            for s in trajectory
        )

        if len(trajectory) < 2:
            continue

        # ── outcome 判断 ──
        final_status: str | None = None
        for ev in reversed(evs):
            if ev.get("type") in ("trace.artifacts", "session.ended"):
                d = ev.get("data") or {}
                final_status = d.get("finalStatus") or d.get("status")
                if final_status:
                    break

        if final_status == "success":
            outcome = "success"
        elif final_status in ("error", "failed"):
            outcome = "failure"
        elif has_tool_error:
            outcome = "likely_failure"
        else:
            # 有 final_output 且步骤正常结束视为 likely_success
            outcome = "likely_success" if final_output else "unknown"

        # ── 统计 ──
        tool_steps   = [s for s in trajectory if s["step_type"] == "tool_call"]
        unique_tools = list({s["tool_name"] for s in tool_steps if s.get("tool_name")})
        rounds       = round_num

        results.append({
            "session_id":   sid,
            "user_input":   user_input or "",
            "final_output": final_output,
            "trajectory":   trajectory,
            "created_at":   created_at.isoformat() if created_at else None,
            "agent":        agent,
            "model_version": model_version,
            "metadata": {
                "model_version": model_version,
                "session_key":   session_key,
                "provider":      first_ev.get("provider", ""),
                "tool_count":    len(tool_steps),
                "rounds":        rounds,
                "unique_tools":  unique_tools,
            },
            "session_outcome": outcome,
            "next_user_message": None,
        })

    return results


# ─── CLI ─────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="聚合 OpenClaw trajectory.jsonl 为 session-level JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input",  "-i", required=True, help="输入目录或单个 .trajectory.jsonl 文件")
    ap.add_argument("--output", "-o", required=True, help="输出 JSONL 文件路径")
    ap.add_argument("--min-steps", type=int, default=2, help="最少步骤数，不足则跳过")
    args = ap.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(set(input_path.rglob("*.trajectory.jsonl")))

    print(f"发现 {len(files)} 个 trajectory 文件", file=sys.stderr)

    total = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for f in files:
            for s in aggregate_trajectory_file(f):
                if len(s["trajectory"]) >= args.min_steps:
                    out.write(json.dumps(s, ensure_ascii=False) + "\n")
                    total += 1

    print(f"聚合完成: {total} 个 session → {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
