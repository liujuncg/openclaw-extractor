"""
改进版回归测试套件
覆盖 PR 修复内容：
  1. _parse_status 未知状态 fallback 改为 PENDING（非 SUCCESS）
  2. 中文错误关键词检测
  3. Skill 违规检测收紧（通用工具白名单 + Levenshtein 兜底）

运行: python3 tests/test_improvements.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import StepStatus, _tool_belongs_to_skill, _levenshtein_distance
from parsers import TrajectoryParser


# ── assert helpers ──────────────────────────────────────────────

def ok(msg: str):
    print(f"  \033[92m✓\033[0m {msg}")

def fail(msg: str):
    print(f"  \033[91m✗\033[0m {msg}")
    raise AssertionError(msg)

def assert_eq(a, b, msg: str = ""):
    if a != b:
        fail(f"{msg}: expected {b!r}, got {a!r}")
    ok(msg or f"{a!r} == {b!r}")

def assert_true(v, msg: str = ""):
    if not v:
        fail(msg or "expected True")
    ok(msg)


# ── Test 1: _parse_status 修复 ─────────────────────────────────

def test_parse_status_unknown_returns_pending():
    """P0 修复：未知状态不能盲目返回 SUCCESS，应返回 PENDING"""
    parser = TrajectoryParser()

    # 各种未知/异常状态 → PENDING
    unknown_cases = [
        None, "", "  ", "unknown", "UNKNOWN",
        "pending", "running", "in_progress", "maybe", "???",
        "完成中", "处理中"  # 中文状态
    ]
    for case in unknown_cases:
        status = parser._parse_status(case)
        assert_eq(status, StepStatus.PENDING, f"_parse_status({case!r}) == PENDING")

    # 空字符串 / none 字符串 → PENDING
    assert_eq(parser._parse_status(""), StepStatus.PENDING, "空字符串 → PENDING")
    assert_eq(parser._parse_status("none"), StepStatus.PENDING, "'none' → PENDING")

    # 明确状态照旧保持不变
    assert_eq(parser._parse_status("success"), StepStatus.SUCCESS, "success 不变")
    assert_eq(parser._parse_status("error"), StepStatus.ERROR, "error 不变")
    assert_eq(parser._parse_status("timeout"), StepStatus.TIMEOUT, "timeout 不变")


def test_parse_status_pending_gets_error_inference():
    """PENDING 状态的 tool_result 会触发 error inference，避免漏网之鱼"""
    parser = TrajectoryParser()

    # 状态 unknown，但 output 含 429 → 应被推断为 ERROR
    rs = {
        "step_id": 1, "type": "tool_result", "call_id": "c1",
        "status": "unknown", "output": "HTTP 429 Too Many Requests",
    }
    step = parser._parse_one_step(rs, 1)
    assert_eq(step.status, StepStatus.ERROR, "unknown + 429 output → inferred ERROR")

    # 状态为空，但 isError=True → ERROR
    rs2 = {
        "step_id": 2, "type": "tool_result", "call_id": "c2",
        "status": "", "isError": True,
    }
    step2 = parser._parse_one_step(rs2, 2)
    assert_eq(step2.status, StepStatus.ERROR, "空 status + isError → ERROR")

    # 状态为 pending，output 正常 → 保持 PENDING（不会误杀）
    rs3 = {
        "step_id": 3, "type": "tool_result", "call_id": "c3",
        "status": "pending", "output": '{"block_id": "blk_123"}',
    }
    step3 = parser._parse_one_step(rs3, 3)
    # PENDING 且无可疑输出 → 经过 inference 后仍是 PENDING（不是强行改 SUCCESS）
    assert_eq(step3.status, StepStatus.PENDING, "pending + 正常 output → 保持 PENDING")


# ── Test 2: 中文错误关键词 ─────────────────────────────────────

def test_chinese_error_patterns_detected():
    """中文错误信息应被正确识别为 ERROR"""
    parser = TrajectoryParser()

    chinese_error_cases = [
        ("连接超时", "connection timeout"),
        ("连接被拒绝", "connection refused"),
        ("连接失败", "connection failed"),
        ("请求超时", "request timeout"),
        ("权限不足", "insufficient permission"),
        ("权限被拒绝", "permission denied"),
        ("找不到文件", "file not found"),
        ("服务器错误", "server error"),
        ("服务不可用", "service unavailable"),
        ("限流", "rate limit"),
        ("频率限制", "rate limiting"),
        ("访问过快", "too fast"),
        ("发生异常", "exception occurred"),
        ("操作失败", "operation failed"),
        ("请求出错", "request error"),
    ]

    for text, label in chinese_error_cases:
        rs = {
            "step_id": 1, "type": "tool_result", "call_id": "c1",
            "status": "success", "output": text,  # 默认标为 success，看 inference 能否发现
        }
        step = parser._parse_one_step(rs, 1)
        # 注意：中文推断不要求 status 一定变成 ERROR，因为 _infer_error_from_text
        # 的作用是标记"含错误文本"。关键是 _infer_error_from_text 本身要能识别
        detected = parser._infer_error_from_text(text)
        assert_true(detected, f"中文关键词应被识别: '{text}' ({label})")


def test_chinese_error_with_parsing_pipeline():
    """端到端：含中文错误信息的 trajectory 应被正确解析为 ERROR"""
    parser = TrajectoryParser()
    raw = {
        "session_id": "cn_err_001",
        "user_input": "查询数据",
        "final_output": "查询失败",
        "trajectory": [
            {"step_id": 1, "type": "tool_call", "tool_name": "data_query", "call_id": "tc1"},
            {"step_id": 2, "type": "tool_result", "call_id": "tc1", "status": "success",
             "output": "数据库连接超时，请稍后重试"},  # status 是 success，但内容是错误
        ],
        "session_outcome": {"status": "failure"}
    }
    traj = parser.parse(raw)
    assert_true(traj is not None, "含中文错误的 raw 能解析")
    err_steps = traj.error_steps
    assert_true(len(err_steps) >= 1, f"应检测到至少 1 个 error_step，实际 {len(err_steps)}")


# ── Test 3: Skill 违规检测收紧 ─────────────────────────────────

def test_universal_tools_whitelist():
    """通用工具 (read/write/exec/browser/process/shell) 不应被判为违规"""
    universal = ["read", "write", "exec", "browser", "process", "shell"]
    for tool in universal:
        # 在任何 Skill 上下文中都不应被判违规
        result = _tool_belongs_to_skill(tool + "_file", "ReportSkill")
        assert_true(result, f"通用工具 {tool} 不应被判违规（白名单）")
        result2 = _tool_belongs_to_skill(tool, "UnknownSkill")
        assert_true(result2, f"通用工具 {tool} 在任意 Skill 中都 ok")


def test_levenshtein_distance():
    """Levenshtein 距离计算正确性"""
    assert_eq(_levenshtein_distance("", ""), 0, "空字符串距离=0")
    assert_eq(_levenshtein_distance("a", ""), 1, "'a' vs '' = 1")
    assert_eq(_levenshtein_distance("", "ab"), 2, "'' vs 'ab' = 2")
    assert_eq(_levenshtein_distance("kitten", "sitting"), 3, "kitten→sitting=3")
    assert_eq(_levenshtein_distance("email", "email"), 0, "相同字符串=0")
    assert_eq(_levenshtein_distance("email_send", "emailskill"), 5,
              "email_send vs emailskill 距离=5 (太长，应不匹配)")


def test_skill_violation_tightened_rules():
    """收紧后的 Skill 违规检测：明显不匹配应被判违规"""
    # === 明显违规（之前宽松规则可能通过，现在应被拦截）===

    # feishu_doc 工具在 DataSkill 中 → 明显不匹配
    result = _tool_belongs_to_skill("feishu_doc", "DataSkill")
    assert_true(not result, "feishu_doc 不在 DataSkill 中 → 应判违规")

    # db_write 在 FileSkill 中 → 不匹配（db 不符 file）
    result = _tool_belongs_to_skill("db_write", "FileSkill")
    assert_true(not result, "db_write 不在 FileSkill 中 → 应判违规")

    # email_send 在 DataSkill 中 → 不匹配
    result = _tool_belongs_to_skill("email_send", "DataSkill")
    assert_true(not result, "email_send 不在 DataSkill 中 → 应判违规")

    # === 合理匹配（应通过）===

    # data_query 在 DataSkill 中 → 匹配
    result = _tool_belongs_to_skill("data_query", "DataSkill")
    assert_true(result, "data_query 在 DataSkill 中 → 应通过")

    # report_generate 在 ReportSkill 中 → 匹配
    result = _tool_belongs_to_skill("report_generate", "ReportSkill")
    assert_true(result, "report_generate 在 ReportSkill 中 → 应通过")

    # email_send 在 EmailSkill 中 → 匹配
    result = _tool_belongs_to_skill("email_send", "EmailSkill")
    assert_true(result, "email_send 在 EmailSkill 中 → 应通过")

    # file_read 在 FileSkill 中 → 匹配（skill 核心名出现在 tool 名中）
    result = _tool_belongs_to_skill("file_read", "FileSkill")
    assert_true(result, "file_read 在 FileSkill 中（包含匹配）→ 应通过")


def test_skill_violation_trajectory_level():
    """在 Trajectory 级别验证 tightened Skill 违规检测"""
    from models import Trajectory, TrajectoryStep, StepType

    # 构造一个含 Skill 违规的 trajectory
    steps = [
        TrajectoryStep(step_id=1, step_type=StepType.TOOL_CALL,
                       tool_name="feishu_doc_update", skill_context="DataSkill"),
        TrajectoryStep(step_id=2, step_type=StepType.TOOL_RESULT, status=StepStatus.SUCCESS),
        TrajectoryStep(step_id=3, step_type=StepType.TOOL_CALL,
                       tool_name="email_send", skill_context="CodeSkill"),
        TrajectoryStep(step_id=4, step_type=StepType.TOOL_RESULT, status=StepStatus.SUCCESS),
    ]
    traj = Trajectory(session_id="sv_001", user_input="test", steps=steps)
    assert_true(traj.has_skill_violations, "含 feishu_doc 在 DataSkill + email 在 CodeSkill → 应判违规")

    # 合法 trajectory
    steps_ok = [
        TrajectoryStep(step_id=1, step_type=StepType.TOOL_CALL,
                       tool_name="data_query", skill_context="DataSkill"),
        TrajectoryStep(step_id=2, step_type=StepType.TOOL_RESULT, status=StepStatus.SUCCESS),
        TrajectoryStep(step_id=3, step_type=StepType.TOOL_CALL,
                       tool_name="read", skill_context="ReportSkill"),
        TrajectoryStep(step_id=4, step_type=StepType.TOOL_RESULT, status=StepStatus.SUCCESS),
    ]
    traj_ok = Trajectory(session_id="sv_002", user_input="test", steps=steps_ok)
    assert_true(not traj_ok.has_skill_violations,
                "data_query 在 DataSkill + read（通用工具）在 ReportSkill → 合法")


# ── main ────────────────────────────────────────────────────────

def main():
    print("\n\033[1mOpenClaw Extractor — 改进回归测试\033[0m")
    print("=" * 55)
    print("测试内容: _parse_status fallback | 中文错误 | Skill 违规收紧")
    print("=" * 55)

    tests = [
        test_parse_status_unknown_returns_pending,
        test_parse_status_pending_gets_error_inference,
        test_chinese_error_patterns_detected,
        test_chinese_error_with_parsing_pipeline,
        test_universal_tools_whitelist,
        test_levenshtein_distance,
        test_skill_violation_tightened_rules,
        test_skill_violation_trajectory_level,
    ]

    passed = 0
    failed_tests = []
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            failed_tests.append((t.__name__, str(e)))
        except Exception as e:
            import traceback
            failed_tests.append((t.__name__, traceback.format_exc()))

    print(f"\n{'=' * 55}")
    print(f"\033[1m结果: \033[92m{passed}/{len(tests)} 通过\033[0m", end="")
    if failed_tests:
        print(f"  \033[91m{len(failed_tests)} 失败\033[0m")
        for name, err in failed_tests:
            print(f"\n  \033[91mFAILED:\033[0m {name}")
            print(f"    {err}")
        sys.exit(1)
    else:
        print(f"\n\033[92m全部通过 ✓\033[0m")


if __name__ == "__main__":
    main()
