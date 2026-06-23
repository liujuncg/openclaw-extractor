"""
OpenClaw / Hermes 轨迹提取器 — 数据模型
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4


# ─────────────────────────────────────────────
# 枚举
# ─────────────────────────────────────────────

class StepType(str, Enum):
    REASONING    = "reasoning"
    SKILL_ROUTE  = "skill_route"
    TOOL_CALL    = "tool_call"
    TOOL_RESULT  = "tool_result"
    OUTPUT       = "output"
    UNKNOWN      = "unknown"


class StepStatus(str, Enum):
    SUCCESS  = "success"
    ERROR    = "error"
    TIMEOUT  = "timeout"
    PENDING  = "pending"


class SessionOutcome(str, Enum):
    SUCCESS        = "success"
    FAILURE        = "failure"
    LIKELY_SUCCESS = "likely_success"
    LIKELY_FAILURE = "likely_failure"
    PARTIAL        = "partial"
    ABANDONED      = "abandoned"
    UNKNOWN        = "unknown"


class TrainingUse(str, Enum):
    SFT_POSITIVE       = "sft_positive"
    RECOVERY_TRAINING  = "recovery_training"
    DPO_CANDIDATE      = "dpo_candidate"
    SKILL_VIOLATION    = "skill_violation_neg"
    GRACEFUL_ABORT     = "graceful_abort"
    LOW_VALUE          = "low_value"


# ─────────────────────────────────────────────
# 轨迹步骤
# ─────────────────────────────────────────────

@dataclass
class TrajectoryStep:
    step_id:    int
    step_type:  StepType
    timestamp:  Optional[datetime] = None

    # reasoning / output 内容
    content:    Optional[str] = None

    # tool_call 字段
    tool_name:      Optional[str] = None
    skill_context:  Optional[str] = None
    input_params:   dict[str, Any] = field(default_factory=dict)
    call_id:        Optional[str] = None

    # tool_result 字段
    result_call_id: Optional[str] = None
    status:         StepStatus = StepStatus.SUCCESS
    output:         Optional[Any] = None
    error:          Optional[str] = None
    latency_ms:     Optional[int] = None

    # skill_route 字段
    skill_selected:   Optional[str] = None
    skill_candidates: list[str] = field(default_factory=list)
    routing_reason:   Optional[str] = None
    routing_confidence: Optional[float] = None

    raw: dict[str, Any] = field(default_factory=dict)   # 原始字段备份

    # ── 便捷属性 ──

    @property
    def is_error(self) -> bool:
        return self.step_type == StepType.TOOL_RESULT and self.status == StepStatus.ERROR

    @property
    def is_reasoning(self) -> bool:
        return self.step_type == StepType.REASONING

    @property
    def has_recovery_signal(self) -> bool:
        text_to_check = self.content or ''
        if self.step_type in (StepType.OUTPUT, StepType.UNKNOWN) and self.content:
            text_to_check = self.content
        if not text_to_check and self.step_type == StepType.TOOL_RESULT:
            text_to_check = str(self.error or '') + str(self.output or '')
        if not text_to_check:
            return False
        keywords = [
            # 中文
            "重试", "错误", "失败", "回退", "纠正", "换一种", "重新",
            "尝试另", "改用", "换个", "出错了", "不对", "问题",
            # 英文（OpenClaw Agent 常用）
            "retry", "error", "failed", "fallback", "alternative",
            "try again", "try another", "instead", "wrong", "mistake",
            "unexpected", "exception", "traceback", "let me try",
            "doesn't work", "not working", "issue", "problem",
            "correct approach", "different approach", "switch to",
        ]
        return any(k in text_to_check.lower() for k in keywords)


# ─────────────────────────────────────────────
# 训练价值评分
# ─────────────────────────────────────────────

@dataclass
class TrainingValueScore:
    complexity:       float = 0.0   # 任务复杂度
    novelty:          float = 0.0   # 相对训练集的新颖性（无向量库时默认 0.5）
    recovery_value:   float = 0.0   # 包含错误恢复的价值
    planning_quality: float = 0.0   # 规划推理质量
    compliance:       float = 1.0   # Skill 合规性

    @property
    def total(self) -> float:
        weights = [0.20, 0.25, 0.25, 0.15, 0.15]
        values  = [self.complexity, self.novelty, self.recovery_value,
                   self.planning_quality, self.compliance]
        return round(sum(w * v for w, v in zip(weights, values)), 4)

    def to_dict(self) -> dict:
        return {
            "complexity":       round(self.complexity, 4),
            "novelty":          round(self.novelty, 4),
            "recovery_value":   round(self.recovery_value, 4),
            "planning_quality": round(self.planning_quality, 4),
            "compliance":       round(self.compliance, 4),
            "total":            self.total,
        }


# ─────────────────────────────────────────────
# 完整轨迹
# ─────────────────────────────────────────────

@dataclass
class Trajectory:
    session_id:   str
    user_input:   str
    steps:        list[TrajectoryStep] = field(default_factory=list)
    final_output: Optional[str] = None

    outcome:      SessionOutcome = SessionOutcome.UNKNOWN
    created_at:   Optional[datetime] = None
    agent:        str = "hermes"
    model_version: Optional[str] = None

    value_score:  TrainingValueScore = field(default_factory=TrainingValueScore)
    training_use: TrainingUse = TrainingUse.LOW_VALUE

    # 提取过程中发现的问题
    warnings:     list[str] = field(default_factory=list)

    # ── 计算属性 ──

    @property
    def tool_calls(self) -> list[TrajectoryStep]:
        return [s for s in self.steps if s.step_type == StepType.TOOL_CALL]

    @property
    def tool_results(self) -> list[TrajectoryStep]:
        return [s for s in self.steps if s.step_type == StepType.TOOL_RESULT]

    @property
    def reasoning_steps(self) -> list[TrajectoryStep]:
        return [s for s in self.steps if s.step_type == StepType.REASONING]

    @property
    def skill_routes(self) -> list[TrajectoryStep]:
        return [s for s in self.steps if s.step_type == StepType.SKILL_ROUTE]

    @property
    def error_steps(self) -> list[TrajectoryStep]:
        return [s for s in self.steps if s.is_error]

    @property
    def unique_tools(self) -> set[str]:
        return {s.tool_name for s in self.tool_calls if s.tool_name}

    @property
    def unique_skills(self) -> set[str]:
        skills = set()
        for s in self.steps:
            if s.skill_context:
                skills.add(s.skill_context)
            if s.skill_selected:
                skills.add(s.skill_selected)
        return skills

    @property
    def skill_switches(self) -> int:
        """统计 Skill 切换次数"""
        contexts = [s.skill_context for s in self.tool_calls if s.skill_context]
        return sum(1 for i in range(1, len(contexts)) if contexts[i] != contexts[i-1])

    @property
    def has_skill_violations(self) -> bool:
        """检测是否有 Skill 越界调用（工具所属 Skill 与当前 Skill 上下文不一致）"""
        for s in self.tool_calls:
            if s.skill_context and s.tool_name:
                # 简单规则：工具名前缀应该与 Skill 名匹配
                # 真实场景中替换为 OpenClaw 的 Skill 注册表查询
                if not _tool_belongs_to_skill(s.tool_name, s.skill_context):
                    return True
        return False

    @property
    def has_unhandled_errors(self) -> bool:
        error_count    = len(self.error_steps)
        recovery_count = sum(1 for s in self.reasoning_steps if s.has_recovery_signal)
        return error_count > recovery_count

    def to_sft_messages(self, system_prompt: str = "") -> list[dict]:
        """
        转换为 SFT 训练格式（OpenAI messages 格式）。

        格式：
          system  → system_prompt
          user    → user_input（只保留真实任务，不含完整 prompt）
          assistant → thinking + tool_call
          tool     → tool_result（作为独立的 tool message）
          assistant → 后续 thinking + tool_call + output

        多轮对话结构通过 tool_result 触发 assistant message 拆分来保留，
        而非把所有步骤拼成单条 message。
        """
        import json as _json

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": self.user_input})

        assistant_parts: list[str] = []

        def _flush_assistant():
            """把积累的 assistant_parts 写成一条 message"""
            if assistant_parts:
                messages.append({"role": "assistant", "content": "\n".join(assistant_parts)})
                assistant_parts.clear()

        for step in self.steps:

            # reasoning：跳过仅含任务理解前缀的步骤（那是用户输入的冗余复制）
            if step.step_type == StepType.REASONING and step.content:
                content = step.content
                if content.startswith("[任务理解]"):
                    continue   # aggregate_sessions 注入的辅助标记，不进 SFT
                assistant_parts.append(f"<thinking>\n{content}\n</thinking>")

            elif step.step_type == StepType.SKILL_ROUTE:
                assistant_parts.append(
                    f"<skill_route skill=\"{step.skill_selected or ''}\" "
                    f"confidence=\"{step.routing_confidence or ''}\">"
                    f"{step.routing_reason or ''}</skill_route>"
                )

            elif step.step_type == StepType.TOOL_CALL:
                assistant_parts.append(
                    f"<tool_call>\n"
                    f"{_json.dumps({'tool': step.tool_name, 'params': step.input_params}, ensure_ascii=False)}\n"
                    f"</tool_call>"
                )

            elif step.step_type == StepType.TOOL_RESULT:
                # tool_result 作为独立 message，触发 assistant message 拆分
                _flush_assistant()
                status = step.status.value if hasattr(step.status, 'value') else str(step.status)
                out    = step.output or step.error or ""
                if not isinstance(out, str):
                    out = _json.dumps(out, ensure_ascii=False)
                messages.append({
                    "role": "tool",
                    "content": f"<tool_result status=\"{status}\">\n{out[:3000]}\n</tool_result>",
                })

            elif step.step_type == StepType.OUTPUT and step.content:
                assistant_parts.append(step.content)

        _flush_assistant()
        if not messages or messages[-1]["role"] == "user":
            if self.final_output:
                messages.append({"role": "assistant", "content": self.final_output})

        return messages

    def to_dict(self) -> dict:
        """完整序列化（用于存储和 DPO 配对）"""
        return {
            "session_id":    self.session_id,
            "user_input":    self.user_input,
            "final_output":  self.final_output,
            "outcome":       self.outcome.value,
            "agent":         self.agent,
            "model_version": self.model_version,
            "created_at":    self.created_at.isoformat() if self.created_at else None,
            "value_score":   self.value_score.to_dict(),
            "training_use":  self.training_use.value,
            "stats": {
                "total_steps":    len(self.steps),
                "tool_calls":     len(self.tool_calls),
                "unique_tools":   list(self.unique_tools),
                "unique_skills":  list(self.unique_skills),
                "skill_switches": self.skill_switches,
                "error_steps":    len(self.error_steps),
                "has_recovery":   any(s.has_recovery_signal for s in self.reasoning_steps),
                "has_violations": self.has_skill_violations,
            },
            "warnings": self.warnings,
            # trajectory 用 list 格式，show_samples.py 的 traj[:10] 才能正常切片
            "trajectory": [_step_to_dict(s) for s in self.steps],
        }


# ─────────────────────────────────────────────
# 内部工具函数
# ─────────────────────────────────────────────

def _tool_belongs_to_skill(tool_name: str, skill_context: str) -> bool:
    """
    简化的工具归属检查。
    真实部署时替换为 OpenClaw Skill 注册表的查询逻辑。

    当前规则（收紧版，减少漏报）：
      1. skill 名（去掉 "skill" 后缀）出现在 tool 名里     e.g. DataSkill → data_query ✓
      2. tool 名前缀（至少 3 个字符）在 skill 核心名里     e.g. fs_list → FileSkill ✓
      3. 已知通用工具白名单 (read/write/exec/browser) → 不判违规
      4. 兜底：首字符相同 + 总体编辑距离 ≤2（保守）     e.g. email_send → EmailSkill ✓
    """
    # 白名单：这些是基础工具，任何 Skill 都可能合理调用
    UNIVERSAL_TOOLS = {"read", "write", "exec", "browser", "process", "shell"}
    tool_base = tool_name.lower().split("_")[0]
    if tool_base in UNIVERSAL_TOOLS:
        return True

    skill_core = skill_context.lower().replace("skill", "").replace("_", "").strip()
    tool_lower = tool_name.lower().replace("_", "")

    if not skill_core:
        return True   # skill 名为空时不做判断

    # 规则1：skill核心名出现在tool名中（子串包含）
    if skill_core in tool_lower:
        return True

    # 规则2：tool名前缀（至少 3 个字符）出现在 skill 核心名中
    prefix_len = max(3, min(len(tool_lower), len(skill_core)))
    if len(tool_lower) >= 3 and tool_lower[:prefix_len] in skill_core:
        return True

    # 规则3：编辑距离兜底（Levenshtein ≤2，非常保守）
    # 只有当两者长度相近时才触发，避免完全不相关字符串的 false positive
    if abs(len(tool_lower) - len(skill_core)) <= 3:
        if _levenshtein_distance(tool_lower, skill_core) <= 2:
            return True

    return False


def _levenshtein_distance(a: str, b: str) -> int:
    """计算两个字符串的 Levenshtein 编辑距离。"""
    m, n = len(a), len(b)
    # 极端情况优化
    if a == b:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    # 使用两行空间优化
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                curr[j - 1] + 1,      # 插入
                prev[j] + 1,          # 删除
                prev[j - 1] + cost,   # 替换
            )
        prev, curr = curr, prev
    return prev[n]


def _step_to_dict(s: TrajectoryStep) -> dict:
    return {
        "step_id":            s.step_id,
        "step_type":          s.step_type.value,
        "timestamp":          s.timestamp.isoformat() if s.timestamp else None,
        "content":            s.content,
        "tool_name":          s.tool_name,
        "skill_context":      s.skill_context,
        "input_params":       s.input_params,
        "call_id":            s.call_id,
        "result_call_id":     s.result_call_id,
        "status":             s.status.value,
        "output":             s.output,
        "error":              s.error,
        "latency_ms":         s.latency_ms,
        "skill_selected":     s.skill_selected,
        "skill_candidates":   s.skill_candidates,
        "routing_reason":     s.routing_reason,
        "routing_confidence": s.routing_confidence,
    }
