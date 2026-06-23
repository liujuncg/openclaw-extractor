"""
OpenClaw / Hermes 轨迹提取器 — 解析器 + 打分器
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Optional

from models import (
    SessionOutcome, StepStatus, StepType, Trajectory,
    TrajectoryStep, TrainingUse, TrainingValueScore,
)


# ─────────────────────────────────────────────────────────────────
# 日志解析器
# 支持两种输入格式：
#   A) 结构化 JSON（按之前设计的标准格式）
#   B) 非结构化文本日志（从 OpenClaw 现有日志行解析）
# ─────────────────────────────────────────────────────────────────

class TrajectoryParser:

    # ── 公共入口 ─────────────────────────────────────────────────

    def parse(self, raw: dict | str) -> Optional[Trajectory]:
        """
        自动检测格式并解析。
        raw 可以是：
          - dict: 标准结构化日志
          - str:  JSON 字符串 或 原始文本日志
        返回 None 表示日志无效，调用方应丢弃。
        """
        if isinstance(raw, str):
            raw = self._try_parse_json(raw)
            if raw is None:
                return None

        if not isinstance(raw, dict):
            return None

        # 检测格式
        if "trajectory" in raw:
            return self._parse_structured(raw)
        elif "log_lines" in raw or "text" in raw:
            return self._parse_unstructured(raw)
        else:
            # 尝试宽松解析
            return self._parse_lenient(raw)

    # ── 结构化格式解析 ────────────────────────────────────────────

    def _parse_structured(self, raw: dict) -> Optional[Trajectory]:
        warnings: list[str] = []

        # 必填字段检查
        for field in ("session_id", "trajectory", "user_input"):
            if field not in raw:
                return None   # 缺核心字段，直接丢弃

        traj = Trajectory(
            session_id   = raw["session_id"],
            user_input   = self._extract_user_input(raw["user_input"]),
            final_output = raw.get("final_output"),
            agent        = raw.get("agent", "hermes"),
            model_version= raw.get("metadata", {}).get("model_version"),
            created_at   = self._parse_ts(raw.get("created_at")),
        )

        # 解析步骤序列
        raw_steps: list[dict] = raw.get("trajectory", [])
        parsed_steps, step_warnings = self._parse_steps(raw_steps)
        traj.steps    = parsed_steps
        traj.warnings = step_warnings

        # 步骤完整性检查
        ids = [s.step_id for s in traj.steps]
        expected = list(range(1, len(ids) + 1))
        if ids != expected:
            warnings.append(f"step_id 不连续: {ids[:5]}...")

        # 检查 tool_call / tool_result 配对
        call_ids   = {s.call_id for s in traj.tool_calls if s.call_id}
        result_ids = {s.result_call_id for s in traj.tool_results if s.result_call_id}
        unmatched  = call_ids.symmetric_difference(result_ids)
        if unmatched:
            warnings.append(f"未配对的 tool call/result: {unmatched}")

        # 解析 session outcome
        outcome_raw = raw.get("session_outcome", {})
        if isinstance(outcome_raw, dict):
            traj.outcome = self._parse_outcome_field(outcome_raw)
        elif isinstance(outcome_raw, str):
            traj.outcome = self._str_to_outcome(outcome_raw)
        else:
            traj.outcome = SessionOutcome.UNKNOWN

        traj.warnings.extend(warnings)
        return traj

    def _parse_steps(
        self, raw_steps: list[dict]
    ) -> tuple[list[TrajectoryStep], list[str]]:
        steps: list[TrajectoryStep] = []
        warnings: list[str] = []

        for i, rs in enumerate(raw_steps):
            try:
                step = self._parse_one_step(rs, fallback_id=i + 1)
                steps.append(step)
            except Exception as e:
                warnings.append(f"step {i+1} 解析失败: {e}")

        return steps, warnings

    def _parse_one_step(self, rs: dict, fallback_id: int) -> TrajectoryStep:
        raw_type = rs.get("type", rs.get("step_type", "unknown"))
        step_type = self._normalize_step_type(raw_type)

        step = TrajectoryStep(
            step_id   = int(rs.get("step_id", fallback_id)),
            step_type = step_type,
            timestamp = self._parse_ts(rs.get("timestamp")),
            raw       = rs,
        )

        meta = rs.get("metadata", {})

        if step_type == StepType.REASONING:
            step.content = rs.get("content") or rs.get("text") or rs.get("thinking")

        elif step_type == StepType.SKILL_ROUTE:
            step.skill_selected       = rs.get("skill_selected") or rs.get("skill")
            step.skill_candidates     = rs.get("skill_candidates", [])
            step.routing_reason       = rs.get("routing_reason") or rs.get("reason")
            step.routing_confidence   = meta.get("confidence") or rs.get("confidence")

        elif step_type == StepType.TOOL_CALL:
            step.tool_name     = rs.get("tool_name") or rs.get("tool")
            step.skill_context = rs.get("skill_context") or rs.get("skill")
            step.input_params  = rs.get("input_params") or rs.get("params") or rs.get("arguments") or {}
            step.call_id       = meta.get("call_id") or rs.get("call_id") or rs.get("id")

        elif step_type == StepType.TOOL_RESULT:
            step.result_call_id = rs.get("call_id") or rs.get("result_call_id")
            parsed_status       = self._parse_status(rs.get("status", "success"))
            step.status         = parsed_status
            step.output         = rs.get("output") or rs.get("result") or rs.get("content")
            step.error          = rs.get("error") or rs.get("error_message")
            step.latency_ms     = rs.get("latency_ms") or meta.get("latency_ms")
            # 对无状态、PENDING 状态、或标为 success 但字段含隐式错误的情况，进行二次推断
            raw_status = rs.get("status")
            should_infer = (
                raw_status is None
                or parsed_status == StepStatus.PENDING
                or (parsed_status == StepStatus.SUCCESS and not step.error)
            )
            if should_infer:
                if self._infer_error_from_fields(rs):
                    step.status = StepStatus.ERROR
                    if not step.error:
                        step.error = rs.get("error") or rs.get("error_message") or "inferred_error"
                elif self._infer_error_from_text(str(step.output) if step.output else ""):
                    step.status = StepStatus.ERROR
                    if not step.error:
                        step.error = "inferred_error_from_text"

        elif step_type == StepType.OUTPUT:
            step.content = rs.get("content") or rs.get("text") or rs.get("output")

        return step

    # ── 非结构化文本日志解析 ─────────────────────────────────────

    def _parse_unstructured(self, raw: dict) -> Optional[Trajectory]:
        """
        处理类似 OpenClaw 早期日志的文本格式。
        期望输入包含 'text' 或 'log_lines' 字段。
        """
        text = raw.get("text", "")
        if isinstance(raw.get("log_lines"), list):
            text = "\n".join(raw["log_lines"])

        if not text.strip():
            return None

        session_id   = raw.get("session_id", self._extract_from_text(text, r"session[_-]?id[:\s]+(\S+)"))
        user_input   = raw.get("user_input", self._extract_from_text(text, r"user[_:](.+?)(?:\n|tool_call|$)"))

        if not user_input:
            return None

        traj = Trajectory(
            session_id = session_id or f"unstructured_{id(raw)}",
            user_input = user_input.strip(),
            agent      = raw.get("agent", "hermes"),
            created_at = self._parse_ts(raw.get("created_at")),
        )
        traj.warnings.append("来源为非结构化日志，步骤信息可能不完整")

        # 从文本里提取工具调用
        tool_pattern = re.compile(
            r"tool_call[:\s]+(\w+)\s*\(([^)]*)\)", re.IGNORECASE
        )
        for i, m in enumerate(tool_pattern.finditer(text), start=1):
            traj.steps.append(TrajectoryStep(
                step_id    = i,
                step_type  = StepType.TOOL_CALL,
                tool_name  = m.group(1),
                input_params = self._parse_kwargs(m.group(2)),
            ))

        # 提取最终输出
        output_match = re.search(r"final_output[:\s]+(.+?)(?:\n\n|$)", text, re.DOTALL)
        if output_match:
            traj.final_output = output_match.group(1).strip()

        traj.outcome = (
            SessionOutcome.LIKELY_SUCCESS
            if not re.search(r"error|failed|失败", text, re.IGNORECASE)
            else SessionOutcome.LIKELY_FAILURE
        )
        return traj

    # ── 宽松解析 ─────────────────────────────────────────────────

    def _parse_lenient(self, raw: dict) -> Optional[Trajectory]:
        """
        对格式不标准的日志做尽力解析。
        必须至少能找到 user_input。
        """
        user_input = (
            raw.get("user_input")
            or raw.get("query")
            or raw.get("prompt")
            or raw.get("input")
            or raw.get("message")
            or raw.get("content")
        )
        if not user_input:
            return None

        traj = Trajectory(
            session_id   = raw.get("session_id") or raw.get("id") or f"lenient_{id(raw)}",
            user_input   = str(user_input),
            final_output = raw.get("final_output") or raw.get("output") or raw.get("response"),
            agent        = raw.get("agent", "hermes"),
            created_at   = self._parse_ts(raw.get("created_at") or raw.get("timestamp")),
        )
        traj.warnings.append("宽松解析模式，格式不标准")

        # 尝试从 messages 字段重建步骤
        if "messages" in raw:
            traj.steps = self._steps_from_messages(raw["messages"])

        # outcome
        outcome_str = str(raw.get("outcome") or raw.get("status") or "")
        traj.outcome = self._str_to_outcome(outcome_str)

        return traj

    def _steps_from_messages(self, messages: list[dict]) -> list[TrajectoryStep]:
        steps = []
        for i, msg in enumerate(messages, start=1):
            role    = msg.get("role", "")
            content = msg.get("content", "")
            if role == "assistant":
                steps.append(TrajectoryStep(
                    step_id   = i,
                    step_type = StepType.REASONING,
                    content   = content,
                ))
            elif role == "tool":
                steps.append(TrajectoryStep(
                    step_id       = i,
                    step_type     = StepType.TOOL_RESULT,
                    result_call_id= msg.get("tool_call_id"),
                    output        = content,
                    status        = StepStatus.SUCCESS,
                ))
        return steps

    # ── 工具方法 ─────────────────────────────────────────────────

    @staticmethod
    def _try_parse_json(s: str) -> Optional[dict]:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _extract_user_input(raw_input: Any) -> str:
        if isinstance(raw_input, str):
            return raw_input
        if isinstance(raw_input, dict):
            return raw_input.get("raw") or raw_input.get("text") or str(raw_input)
        return str(raw_input)

    @staticmethod
    def _parse_ts(ts: Any) -> Optional[datetime]:
        if ts is None:
            return None
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts)
        if isinstance(ts, str):
            for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S",
                        "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
                try:
                    return datetime.strptime(ts, fmt)
                except ValueError:
                    continue
        return None

    @staticmethod
    def _normalize_step_type(raw: str) -> StepType:
        mapping = {
            "reasoning": StepType.REASONING, "think": StepType.REASONING,
            "thinking":  StepType.REASONING, "thought": StepType.REASONING,
            "skill_route": StepType.SKILL_ROUTE, "route": StepType.SKILL_ROUTE,
            "routing":     StepType.SKILL_ROUTE,
            "tool_call":   StepType.TOOL_CALL,  "tool": StepType.TOOL_CALL,
            "function_call": StepType.TOOL_CALL,
            "tool_result": StepType.TOOL_RESULT, "tool_response": StepType.TOOL_RESULT,
            "function_result": StepType.TOOL_RESULT,
            "output": StepType.OUTPUT, "response": StepType.OUTPUT,
            "final": StepType.OUTPUT,
        }
        return mapping.get(raw.lower().strip(), StepType.UNKNOWN)

    @staticmethod
    def _parse_status(s: Any) -> StepStatus:
        s = str(s).lower().strip()
        if not s or s == "none":
            return StepStatus.PENDING
        if s in ("success", "ok", "200", "true"):
            return StepStatus.SUCCESS
        if s in ("error", "failed", "fail", "false"):
            return StepStatus.ERROR
        if s in ("timeout", "timed_out"):
            return StepStatus.TIMEOUT
        # 对于未知状态，既不能盲目信任为 SUCCESS，也不能武断判为 ERROR。
        # 保守策略：返回 PENDING，让后续 error inference 进一步判断。
        return StepStatus.PENDING

    # error inference patterns
    _ERROR_TEXT_PATTERNS = [
        # --- HTTP / API 错误 ---
        "429", "rate limit", "rate_limit", "too many requests",
        "timeout", "timed out", "connection timeout", "read timeout",
        "unauthorized", "forbidden", "permission denied",
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

    @classmethod
    def _infer_error_from_fields(cls, rs):
        if rs.get("isError") is True or rs.get("is_error") is True:
            return True
        err = rs.get("error") or rs.get("error_message")
        if err and str(err).strip():
            return True
        return False

    @classmethod
    def _infer_error_from_text(cls, text):
        if not text or not text.strip():
            return False
        low = text.lower()
        for pat in cls._ERROR_TEXT_PATTERNS:
            if pat in low:
                return True
        return False

    @staticmethod
    def _parse_outcome_field(d: dict) -> SessionOutcome:
        status = str(d.get("status", "")).lower()
        mapping = {
            "success":  SessionOutcome.SUCCESS,
            "failure":  SessionOutcome.FAILURE,
            "partial":  SessionOutcome.PARTIAL,
            "abandoned":SessionOutcome.ABANDONED,
        }
        return mapping.get(status, SessionOutcome.UNKNOWN)

    @staticmethod
    def _str_to_outcome(s: str) -> SessionOutcome:
        s = s.lower().strip()
        if s in ("success", "ok", "completed", "done", "成功"):
            return SessionOutcome.SUCCESS
        if s in ("failure", "failed", "error", "失败"):
            return SessionOutcome.FAILURE
        if s in ("partial",):
            return SessionOutcome.PARTIAL
        if s in ("abandoned", "cancelled", "放弃"):
            return SessionOutcome.ABANDONED
        return SessionOutcome.UNKNOWN

    @staticmethod
    def _extract_from_text(text: str, pattern: str) -> Optional[str]:
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1).strip() if m else None

    @staticmethod
    def _parse_kwargs(s: str) -> dict:
        """简单解析 key=value, key='value' 形式的参数字符串"""
        result = {}
        for m in re.finditer(r'(\w+)\s*=\s*["\']?([^,\'"]+)["\']?', s):
            result[m.group(1)] = m.group(2).strip()
        return result


# ─────────────────────────────────────────────────────────────────
# 结果推断器
# ─────────────────────────────────────────────────────────────────

class OutcomeInferrer:
    """
    在 session_outcome 未明确标注时，从轨迹内容推断任务结果。
    """

    # 用户后续消息中表示失败/成功的关键词
    FAILURE_SIGNALS = {"不对", "重做", "错了", "不是", "再来", "wrong",
                       "redo", "incorrect", "try again", "not right"}
    SUCCESS_SIGNALS = {"谢谢", "好的", "完美", "正确", "不错", "done",
                       "thanks", "perfect", "great", "looks good"}

    def infer(self, traj: Trajectory, next_user_message: Optional[str] = None) -> SessionOutcome:
        # 已有明确结果则直接返回
        if traj.outcome not in (SessionOutcome.UNKNOWN,):
            return traj.outcome

        signals: dict[str, Optional[bool]] = {}

        # 信号1：最后一步是否是正常输出
        if traj.steps:
            last = traj.steps[-1]
            signals["clean_ending"] = (
                last.step_type == StepType.OUTPUT and
                not (last.content and re.search(r"error|失败|exception", last.content, re.IGNORECASE))
            )
        else:
            signals["clean_ending"] = None

        # 信号2：是否有未处理的工具错误
        signals["unhandled_errors"] = traj.has_unhandled_errors

        # 信号3：用户后续消息
        signals["user_negative"] = None
        signals["user_positive"] = None
        if next_user_message:
            nm = next_user_message.lower()
            signals["user_negative"] = any(s in nm for s in self.FAILURE_SIGNALS)
            signals["user_positive"] = any(s in nm for s in self.SUCCESS_SIGNALS)

        # 信号4：输出是否包含道歉/失败承认
        if traj.final_output:
            fo = traj.final_output.lower()
            signals["output_failure"] = bool(
                re.search(r"无法完成|任务失败|i(?: am| 'm)? unable|could not complete", fo)
            )
        else:
            signals["output_failure"] = None

        # ── 综合判断 ──
        if signals.get("user_negative"):
            return SessionOutcome.FAILURE
        if signals.get("user_positive"):
            return SessionOutcome.SUCCESS
        if signals.get("output_failure"):
            return SessionOutcome.FAILURE
        if signals.get("clean_ending") and not signals.get("unhandled_errors"):
            return SessionOutcome.LIKELY_SUCCESS
        if signals.get("unhandled_errors"):
            return SessionOutcome.LIKELY_FAILURE

        # 有最终输出但信号不足
        if traj.final_output:
            return SessionOutcome.LIKELY_SUCCESS

        return SessionOutcome.UNKNOWN


# ─────────────────────────────────────────────────────────────────
# 训练价值评分器
# ─────────────────────────────────────────────────────────────────

class TrainingValueScorer:

    def score(self, traj: Trajectory) -> TrainingValueScore:
        sc = TrainingValueScore()

        sc.complexity       = self._score_complexity(traj)
        sc.novelty          = 0.5   # 无向量库时默认中等新颖性
        sc.recovery_value   = self._score_recovery(traj)
        sc.planning_quality = self._score_planning(traj)
        sc.compliance       = 0.0 if traj.has_skill_violations else 1.0

        return sc

    @staticmethod
    def _score_complexity(traj: Trajectory) -> float:
        step_score  = min(1.0, len(traj.steps) / 15)
        tool_score  = min(1.0, len(traj.unique_tools) / 5)
        skill_score = min(1.0, len(traj.unique_skills) / 3)
        switch_score= min(1.0, traj.skill_switches / 3)
        return round(step_score * 0.4 + tool_score * 0.3 + skill_score * 0.2 + switch_score * 0.1, 4)

    @staticmethod
    def _score_recovery(traj: Trajectory) -> float:
        has_error    = len(traj.error_steps) > 0
        has_recovery = any(s.has_recovery_signal for s in traj.reasoning_steps)
        if has_error and has_recovery:
            return 1.0
        if has_error and not has_recovery:
            return 0.3   # 有错误但没有恢复，也有训练价值（作为负样本）
        return 0.0

    @staticmethod
    def _score_planning(traj: Trajectory) -> float:
        if not traj.reasoning_steps:
            return 0.0
        lengths = [len(s.content or "") for s in traj.reasoning_steps]
        avg_len = sum(lengths) / len(lengths)
        # 平均思维链长度 200 字符以上视为有价值
        return min(1.0, avg_len / 200)


# ─────────────────────────────────────────────────────────────────
# 训练用途分类器
# ─────────────────────────────────────────────────────────────────

class TrainingUseClassifier:

    # 阈值配置（可通过配置文件覆盖）
    SFT_MIN_SCORE      = 0.55
    DPO_MIN_NOVELTY    = 0.0   # 无向量库时关闭新颖性门控
    RECOVERY_MIN_SCORE = 0.3

    def classify(self, traj: Trajectory) -> TrainingUse:
        sc      = traj.value_score
        outcome = traj.outcome

        # 最高优先级：Skill 违规 → 负样本
        if traj.has_skill_violations:
            return TrainingUse.SKILL_VIOLATION

        # 主动中止轨迹 → 单独分桶（不丢弃）
        if outcome == SessionOutcome.ABANDONED and len(traj.steps) >= 3:
            return TrainingUse.GRACEFUL_ABORT

        # 有错误恢复轨迹
        if sc.recovery_value > self.RECOVERY_MIN_SCORE and len(traj.error_steps) > 0:
            return TrainingUse.RECOVERY_TRAINING

        # 高质量成功轨迹 → SFT 正样本
        if (
            outcome in (SessionOutcome.SUCCESS, SessionOutcome.LIKELY_SUCCESS) and
            sc.compliance == 1.0 and
            sc.total >= self.SFT_MIN_SCORE
        ):
            return TrainingUse.SFT_POSITIVE

        # 有成功/失败对比价值 → DPO 候选
        if outcome in (
            SessionOutcome.SUCCESS, SessionOutcome.LIKELY_SUCCESS,
            SessionOutcome.FAILURE, SessionOutcome.LIKELY_FAILURE
        ) and sc.total >= 0.35:
            return TrainingUse.DPO_CANDIDATE

        return TrainingUse.LOW_VALUE
