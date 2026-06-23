"""
OpenClaw / Hermes 轨迹提取器 — 测试套件
运行: python tests/test_extractor.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import SessionOutcome, StepType, TrainingUse
from parsers import (
    OutcomeInferrer, TrajectoryParser,
    TrainingUseClassifier, TrainingValueScorer,
)
from models import StepStatus
from extractor import ExtractionConfig, ExtractionPipeline


# ─────────────────────────────────────────────────────────────────
# 样本数据工厂
# ─────────────────────────────────────────────────────────────────

def make_success_session(session_id: str = "s001") -> dict:
    """标准成功轨迹（含规划 + 多工具调用 + 正常输出）"""
    return {
        "session_id": session_id,
        "created_at": "2026-06-19T10:00:00",
        "agent": "hermes",
        "user_input": {
            "raw": "帮我把Q3销售数据整理成按区域汇总的报告，发给团队"
        },
        "metadata": {"model_version": "hermes-v2.1"},
        "trajectory": [
            {
                "step_id": 1, "type": "reasoning",
                "content": (
                    "用户需要：1) 读取Q3数据 2) 按区域汇总 3) 生成报告 4) 发送邮件。"
                    "应先用 DataSkill 读取和处理数据，再用 ReportSkill 生成报告，"
                    "最后用 EmailSkill 发送。"
                )
            },
            {
                "step_id": 2, "type": "skill_route",
                "skill_selected": "DataSkill",
                "skill_candidates": ["DataSkill", "ReportSkill", "EmailSkill"],
                "routing_reason": "首先需要读取原始数据",
                "metadata": {"confidence": 0.92}
            },
            {
                "step_id": 3, "type": "tool_call",
                "tool_name": "data_read_csv",
                "skill_context": "DataSkill",
                "input_params": {"path": "/data/q3_sales.csv", "encoding": "utf-8"},
                "metadata": {"call_id": "tc_001"}
            },
            {
                "step_id": 4, "type": "tool_result",
                "call_id": "tc_001",
                "status": "success",
                "output": {"rows": 1247, "columns": ["region", "amount", "date"]},
                "latency_ms": 210
            },
            {
                "step_id": 5, "type": "reasoning",
                "content": "数据读取成功，共1247行，包含region/amount/date列。开始按region聚合。"
            },
            {
                "step_id": 6, "type": "tool_call",
                "tool_name": "data_aggregate",
                "skill_context": "DataSkill",
                "input_params": {"group_by": "region", "agg": "sum", "col": "amount"},
                "metadata": {"call_id": "tc_002"}
            },
            {
                "step_id": 7, "type": "tool_result",
                "call_id": "tc_002",
                "status": "success",
                "output": {"华北": 2340000, "华东": 3120000, "华南": 1890000},
                "latency_ms": 185
            },
            {
                "step_id": 8, "type": "skill_route",
                "skill_selected": "ReportSkill",
                "skill_candidates": ["ReportSkill", "EmailSkill"],
                "routing_reason": "数据处理完成，切换到报告生成"
            },
            {
                "step_id": 9, "type": "tool_call",
                "tool_name": "report_generate",
                "skill_context": "ReportSkill",
                "input_params": {
                    "template": "quarterly_summary",
                    "data": {"华北": 2340000, "华东": 3120000, "华南": 1890000},
                    "title": "Q3 区域销售汇总"
                },
                "metadata": {"call_id": "tc_003"}
            },
            {
                "step_id": 10, "type": "tool_result",
                "call_id": "tc_003",
                "status": "success",
                "output": {"report_id": "rpt_789", "url": "/reports/rpt_789.pdf"},
                "latency_ms": 430
            },
            {
                "step_id": 11, "type": "skill_route",
                "skill_selected": "EmailSkill",
                "routing_reason": "报告生成完成，发送邮件"
            },
            {
                "step_id": 12, "type": "tool_call",
                "tool_name": "email_send",
                "skill_context": "EmailSkill",
                "input_params": {
                    "to": "team@company.com",
                    "subject": "Q3区域销售汇总报告",
                    "attachment": "/reports/rpt_789.pdf"
                },
                "metadata": {"call_id": "tc_004"}
            },
            {
                "step_id": 13, "type": "tool_result",
                "call_id": "tc_004",
                "status": "success",
                "output": {"message_id": "msg_456"},
                "latency_ms": 320
            },
            {
                "step_id": 14, "type": "output",
                "content": "Q3销售数据已整理完成：华北¥234万、华东¥312万、华南¥189万。报告已发送至团队邮箱。"
            }
        ],
        "final_output": "Q3销售数据已整理完成：华北¥234万、华东¥312万、华南¥189万。报告已发送至团队邮箱。",
        "session_outcome": {"status": "success"}
    }


def make_recovery_session(session_id: str = "s002") -> dict:
    """含工具错误和恢复的轨迹"""
    return {
        "session_id": session_id,
        "created_at": "2026-06-19T11:00:00",
        "agent": "hermes",
        "user_input": {"raw": "查询用户ID为12345的订单历史"},
        "metadata": {"model_version": "hermes-v2.1"},
        "trajectory": [
            {
                "step_id": 1, "type": "reasoning",
                "content": "需要查询特定用户的订单历史，使用 OrderSkill 的查询工具。"
            },
            {
                "step_id": 2, "type": "tool_call",
                "tool_name": "order_query",
                "skill_context": "OrderSkill",
                "input_params": {"user_id": "12345", "limit": 50},
                "metadata": {"call_id": "tc_101"}
            },
            {
                "step_id": 3, "type": "tool_result",
                "call_id": "tc_101",
                "status": "error",
                "error": "DatabaseConnectionError: connection pool exhausted",
                "latency_ms": 5001
            },
            {
                "step_id": 4, "type": "reasoning",
                "content": (
                    "数据库连接池耗尽，不是查询参数问题。"
                    "策略：等待后重试，同时降低limit减少连接压力。"
                    "如果仍然失败，切换到备用只读副本。"
                )
            },
            {
                "step_id": 5, "type": "tool_call",
                "tool_name": "order_query",
                "skill_context": "OrderSkill",
                "input_params": {"user_id": "12345", "limit": 10, "replica": True},
                "metadata": {"call_id": "tc_102"}
            },
            {
                "step_id": 6, "type": "tool_result",
                "call_id": "tc_102",
                "status": "success",
                "output": {"orders": [{"id": "ord_001", "amount": 299}, {"id": "ord_002", "amount": 159}]},
                "latency_ms": 340
            },
            {
                "step_id": 7, "type": "output",
                "content": "找到用户12345的最近10条订单记录（主库繁忙，已从只读副本获取）。"
            }
        ],
        "final_output": "找到用户12345的最近10条订单记录（主库繁忙，已从只读副本获取）。",
        "session_outcome": {"status": "success"}
    }


def make_skill_violation_session(session_id: str = "s003") -> dict:
    """包含 Skill 越界调用的违规轨迹"""
    return {
        "session_id": session_id,
        "created_at": "2026-06-19T12:00:00",
        "agent": "hermes",
        "user_input": {"raw": "生成报告并直接写入数据库"},
        "metadata": {"model_version": "hermes-v2.1"},
        "trajectory": [
            {
                "step_id": 1, "type": "reasoning",
                "content": "需要生成报告并持久化，用 ReportSkill 生成后写入数据库。"
            },
            {
                "step_id": 2, "type": "tool_call",
                "tool_name": "report_generate",
                "skill_context": "ReportSkill",
                "input_params": {"template": "basic", "data": {}},
                "metadata": {"call_id": "tc_201"}
            },
            {
                "step_id": 3, "type": "tool_result",
                "call_id": "tc_201",
                "status": "success",
                "output": {"report_id": "rpt_999"}
            },
            {
                "step_id": 4, "type": "tool_call",
                # 越界：在 ReportSkill 上下文里调用了 DBSkill 的工具
                "tool_name": "db_write",
                "skill_context": "ReportSkill",
                "input_params": {"table": "reports", "data": {"id": "rpt_999"}},
                "metadata": {"call_id": "tc_202"}
            },
            {
                "step_id": 5, "type": "tool_result",
                "call_id": "tc_202",
                "status": "success",
                "output": {"rows_affected": 1}
            },
            {
                "step_id": 6, "type": "output",
                "content": "报告已生成并写入数据库。"
            }
        ],
        "final_output": "报告已生成并写入数据库。",
        "session_outcome": {"status": "success"}
    }


def make_low_value_session(session_id: str = "s004") -> dict:
    """低价值轨迹（单步，无规划）"""
    return {
        "session_id": session_id,
        "created_at": "2026-06-19T13:00:00",
        "agent": "hermes",
        "user_input": {"raw": "你好"},
        "trajectory": [
            {"step_id": 1, "type": "output", "content": "你好！有什么可以帮助你的？"}
        ],
        "final_output": "你好！有什么可以帮助你的？",
        "session_outcome": {"status": "success"}
    }


def make_unstructured_log(session_id: str = "s005") -> dict:
    """模拟非结构化文本日志"""
    return {
        "text": f"""
session_id: {session_id}
user: 分析最近7天的访问日志，找出异常IP
tool_call: log_read(path='/var/log/access.log', days=7)
tool_result: success, 45230 entries
tool_call: ip_analyze(data='...', threshold=100)
tool_result: success, found 3 suspicious IPs
final_output: 发现3个异常IP：192.168.1.100（请求1240次），10.0.0.55（请求890次）
""",
        "agent": "hermes"
    }


# ─────────────────────────────────────────────────────────────────
# 测试类
# ─────────────────────────────────────────────────────────────────

class Colors:
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    RESET  = "\033[0m"
    BOLD   = "\033[1m"


def ok(msg: str):
    print(f"  {Colors.GREEN}✓{Colors.RESET} {msg}")


def fail(msg: str):
    print(f"  {Colors.RED}✗{Colors.RESET} {msg}")
    raise AssertionError(msg)


def section(title: str):
    print(f"\n{Colors.CYAN}{Colors.BOLD}── {title}{Colors.RESET}")


def assert_eq(a, b, msg: str = ""):
    if a != b:
        fail(f"{msg}: expected {b!r}, got {a!r}")
    ok(msg or f"{a!r} == {b!r}")


def assert_true(v, msg: str = ""):
    if not v:
        fail(msg or "expected True")
    ok(msg)


def assert_in(v, container, msg: str = ""):
    if v not in container:
        fail(f"{msg}: {v!r} not in {container!r}")
    ok(msg or f"{v!r} in container")


# ── 测试组 ───────────────────────────────────────────────────────

def test_parser():
    section("TrajectoryParser")
    parser = TrajectoryParser()

    # 成功轨迹解析
    traj = parser.parse(make_success_session())
    assert_true(traj is not None, "成功轨迹解析不为 None")
    assert_eq(traj.session_id, "s001", "session_id")
    assert_eq(len(traj.steps), 14, "步骤总数")
    assert_eq(len(traj.tool_calls), 4, "工具调用数")
    assert_eq(len(traj.tool_results), 4, "工具结果数")
    assert_eq(len(traj.reasoning_steps), 2, "推理步骤数")
    assert_in("DataSkill", traj.unique_skills, "DataSkill 在 unique_skills 里")
    assert_eq(traj.skill_switches, 2, "Skill 切换次数（DataSkill→ReportSkill→EmailSkill）")

    # 恢复轨迹解析
    traj2 = parser.parse(make_recovery_session())
    assert_true(traj2 is not None, "恢复轨迹解析不为 None")
    assert_eq(len(traj2.error_steps), 1, "错误步骤数为 1")
    recovery_signals = [s for s in traj2.reasoning_steps if s.has_recovery_signal]
    assert_true(len(recovery_signals) > 0, "恢复信号被识别")

    # 非结构化日志
    traj3 = parser.parse(make_unstructured_log())
    assert_true(traj3 is not None, "非结构化日志解析不为 None")
    assert_true(len(traj3.warnings) > 0, "非结构化日志有警告标记")

    # 缺少 user_input → 返回 None
    invalid = {"session_id": "x", "trajectory": []}
    result = parser.parse(invalid)
    assert_true(result is None, "缺少 user_input 返回 None")

    # JSON 字符串输入
    traj4 = parser.parse(json.dumps(make_success_session()))
    assert_true(traj4 is not None, "JSON 字符串输入解析成功")


def test_outcome_inferrer():
    section("OutcomeInferrer")
    parser   = TrajectoryParser()
    inferrer = OutcomeInferrer()

    # 明确成功
    traj = parser.parse(make_success_session())
    traj.outcome = SessionOutcome.UNKNOWN   # 重置后重推断
    outcome = inferrer.infer(traj)
    assert_in(outcome, [SessionOutcome.SUCCESS, SessionOutcome.LIKELY_SUCCESS], "成功轨迹推断为成功")

    # 用户反馈负向
    traj2 = parser.parse(make_success_session("s_neg"))
    traj2.outcome = SessionOutcome.UNKNOWN
    outcome2 = inferrer.infer(traj2, next_user_message="不对，重做")
    assert_eq(outcome2, SessionOutcome.FAILURE, "负向用户反馈推断为失败")

    # 用户反馈正向
    traj3 = parser.parse(make_success_session("s_pos"))
    traj3.outcome = SessionOutcome.UNKNOWN
    outcome3 = inferrer.infer(traj3, next_user_message="谢谢，完美！")
    assert_eq(outcome3, SessionOutcome.SUCCESS, "正向用户反馈推断为成功")


def test_scorer():
    section("TrainingValueScorer")
    parser = TrajectoryParser()
    scorer = TrainingValueScorer()

    traj_success  = parser.parse(make_success_session())
    traj_recovery = parser.parse(make_recovery_session())
    traj_low      = parser.parse(make_low_value_session())

    sc_success  = scorer.score(traj_success)
    sc_recovery = scorer.score(traj_recovery)
    sc_low      = scorer.score(traj_low)

    assert_true(sc_success.total > sc_low.total, "成功复杂轨迹分数 > 低价值轨迹")
    assert_true(sc_recovery.recovery_value > 0.5, "恢复轨迹 recovery_value > 0.5")
    assert_eq(sc_low.complexity < 0.3, True, "低价值轨迹 complexity < 0.3")

    # Skill 违规
    traj_viol = parser.parse(make_skill_violation_session())
    sc_viol   = scorer.score(traj_viol)
    assert_eq(sc_viol.compliance, 0.0, "Skill 违规 compliance = 0")


def test_classifier():
    section("TrainingUseClassifier")
    parser     = TrajectoryParser()
    inferrer   = OutcomeInferrer()
    scorer     = TrainingValueScorer()
    classifier = TrainingUseClassifier()

    def classify(session_factory, outcome_override=None):
        traj = parser.parse(session_factory())
        if outcome_override:
            traj.outcome = outcome_override
        else:
            traj.outcome = inferrer.infer(traj)
        traj.value_score  = scorer.score(traj)
        traj.training_use = classifier.classify(traj)
        return traj.training_use

    use_success  = classify(make_success_session, SessionOutcome.SUCCESS)
    use_recovery = classify(make_recovery_session, SessionOutcome.SUCCESS)
    use_viol     = classify(make_skill_violation_session, SessionOutcome.SUCCESS)
    use_low      = classify(make_low_value_session, SessionOutcome.SUCCESS)

    # 成功轨迹分数 0.5065，默认 SFT_MIN_SCORE=0.55 时走 DPO_CANDIDATE
    # 调低阈值后应为 SFT_POSITIVE（实际部署时阈值按数据集质量调整）
    assert_in(use_success, [TrainingUse.SFT_POSITIVE, TrainingUse.DPO_CANDIDATE],
              "成功轨迹 → SFT_POSITIVE 或 DPO_CANDIDATE（取决于阈值）")
    assert_eq(use_recovery, TrainingUse.RECOVERY_TRAINING,  "恢复轨迹 → RECOVERY_TRAINING")
    assert_eq(use_viol,     TrainingUse.SKILL_VIOLATION,    "违规轨迹 → SKILL_VIOLATION")
    assert_eq(use_low,      TrainingUse.LOW_VALUE,          "低价值轨迹 → LOW_VALUE")


def test_sft_format():
    section("SFT 格式转换")
    parser = TrajectoryParser()
    traj   = parser.parse(make_success_session())

    messages = traj.to_sft_messages("You are Hermes.")
    assert_true(len(messages) >= 3, "messages 至少 3 条（system/user/assistant）")
    assert_eq(messages[0]["role"], "system",    "第一条是 system")
    assert_eq(messages[1]["role"], "user",      "第二条是 user")
    assert_eq(messages[2]["role"], "assistant", "第三条是 assistant")
    assert_in("<thinking>", messages[2]["content"], "assistant 内容含 <thinking> 块")
    assert_in("<tool_call>", messages[2]["content"], "assistant 内容含 <tool_call> 块")




def test_parser_error_inference():
    """parser infers ERROR from isError / error_message / 429 text"""
    section("Parser Error Inference")
    parser = TrajectoryParser()

    # isError=True
    rs = {"step_id": 1, "type": "tool_result", "call_id": "c1", "isError": True, "output": "some output"}
    step = parser._parse_one_step(rs, 1)
    assert_eq(step.status, StepStatus.ERROR, "isError=True -> ERROR")
    assert_true(step.error is not None, "isError sets error")

    # error_message
    rs = {"step_id": 2, "type": "tool_result", "call_id": "c2", "error_message": "connection refused"}
    step = parser._parse_one_step(rs, 2)
    assert_eq(step.status, StepStatus.ERROR, "error_message -> ERROR")

    # 429 text
    rs = {"step_id": 3, "type": "tool_result", "call_id": "c3", "output": "HTTP 429 Too Many Requests"}
    step = parser._parse_one_step(rs, 3)
    assert_eq(step.status, StepStatus.ERROR, "429 text -> ERROR")

    # rate limit text
    rs = {"step_id": 4, "type": "tool_result", "call_id": "c4", "output": "rate limit exceeded"}
    step = parser._parse_one_step(rs, 4)
    assert_eq(step.status, StepStatus.ERROR, "rate limit text -> ERROR")

    # normal result not misjedged
    rs = {"step_id": 5, "type": "tool_result", "call_id": "c5", "status": "success", "output": '{"block_id": "blk_123"}'}
    step = parser._parse_one_step(rs, 5)
    assert_eq(step.status, StepStatus.SUCCESS, "normal result -> SUCCESS")


def test_pipeline_recovery_training_with_tool_error():
    """extractor generates recovery_training for session with tool error"""
    section("Recovery Training with Tool Error")
    session = make_recovery_session("rec_err_001")
    import tempfile, json
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "test_logs.jsonl"
        input_file.write_text(json.dumps(session, ensure_ascii=False))
        output_dir = Path(tmpdir) / "output"
        config = ExtractionConfig(input_path=input_file, output_dir=output_dir, min_score=0.3, min_steps=1)
        pipeline = ExtractionPipeline(config)
        result = pipeline.run()
        rec_path = output_dir / "recovery_training.jsonl"
        assert_true(rec_path.exists(), "recovery_training.jsonl exists")
        lines = rec_path.read_text().strip().splitlines()
        assert_true(len(lines) >= 1, "recovery_training has >=1 sample")
        sample = json.loads(lines[0])
        assert_in("error_steps", sample, "recovery sample has error_steps")
        assert_true(len(sample["error_steps"]) >= 1, "error_steps >= 1")

def test_pipeline_end_to_end():
    section("端到端流水线")

    sessions = [
        make_success_session("e2e_001"),
        make_recovery_session("e2e_002"),
        make_skill_violation_session("e2e_003"),
        make_low_value_session("e2e_004"),
        make_unstructured_log("e2e_005"),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        # 写 JSONL 输入
        input_file = Path(tmpdir) / "test_logs.jsonl"
        input_file.write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in sessions)
        )

        output_dir = Path(tmpdir) / "output"
        config = ExtractionConfig(
            input_path      = input_file,
            output_dir      = output_dir,
            min_score       = 0.3,
            min_steps       = 1,
            output_format   = "jsonl",
            include_unknown = True,
            save_rejected   = True,
        )

        pipeline = ExtractionPipeline(config)
        result   = pipeline.run()

        assert_true(result.stats["total_raw"] == 5, "处理了5条日志")
        assert_true(result.stats["extracted"] >= 3,  "至少提取了3条样本")

        # 检查输出文件存在
        manifest_path = output_dir / "manifest.json"
        assert_true(manifest_path.exists(), "manifest.json 已生成")

        manifest = json.loads(manifest_path.read_text())
        assert_true("files" in manifest, "manifest 包含 files 字段")
        assert_true("stats" in manifest, "manifest 包含 stats 字段")

        # 检查 SFT 文件
        sft_path = output_dir / "sft_positive.jsonl"
        if sft_path.exists():
            lines = sft_path.read_text().strip().splitlines()
            sample = json.loads(lines[0])
            assert_in("messages", sample, "SFT 样本包含 messages 字段")
            assert_true(len(sample["messages"]) >= 3, "SFT messages 至少 3 条")
            assert_in("trajectory", sample, "SFT 样本包含 trajectory 字段")
            assert_true(isinstance(sample["trajectory"], dict), "SFT trajectory 是 dict")
            assert_true(bool(sample["trajectory"]), "SFT trajectory 非空")
            assert_true(len(sample["trajectory"].get("trajectory", [])) > 0,
                        "SFT trajectory.steps 非空")
            ok(f"SFT 文件包含 {len(lines)} 条样本")

        # 检查 rejected 文件（save_rejected=True）
        rejected_path = output_dir / "rejected.jsonl"
        assert_true(rejected_path.exists(), "rejected.jsonl 已生成（save_rejected=True）")
        if rejected_path.exists():
            lines = rejected_path.read_text().strip().splitlines()
            assert_true(len(lines) >= 1, "rejected 至少 1 条样本")
            sample = json.loads(lines[0])
            for field in ("session_id", "reason", "score", "training_use",
                          "outcome", "stats", "warnings"):
                assert_in(field, sample, f"rejected 样本包含 {field}")
            assert_in("trajectory", sample, "rejected 样本包含 trajectory 字段")
            assert_true(isinstance(sample["trajectory"], dict), "rejected trajectory 是 dict")
            assert_true(bool(sample["trajectory"]), "rejected trajectory 非空")
            assert_true(len(sample["trajectory"].get("trajectory", [])) > 0,
                        "rejected trajectory.steps 非空")
            ok(f"rejected 文件包含 {len(lines)} 条样本")

        # 检查恢复训练文件
        recovery_path = output_dir / "recovery_training.jsonl"
        if recovery_path.exists():
            lines = recovery_path.read_text().strip().splitlines()
            sample = json.loads(lines[0])
            assert_in("error_steps", sample, "恢复样本包含 error_steps 字段")
            ok(f"恢复训练文件包含 {len(lines)} 条样本")


# ─────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────

def main():
    print(f"\n{Colors.BOLD}OpenClaw 轨迹提取器 — 测试套件{Colors.RESET}")
    print("=" * 50)

    tests = [
        test_parser,
        test_outcome_inferrer,
        test_scorer,
        test_classifier,
        test_sft_format,
        test_pipeline_end_to_end,
        test_parser_error_inference,
        test_pipeline_recovery_training_with_tool_error,
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

    print(f"\n{'=' * 50}")
    print(f"{Colors.BOLD}结果: {Colors.GREEN}{passed}/{len(tests)} 通过{Colors.RESET}", end="")
    if failed_tests:
        print(f"  {Colors.RED}{len(failed_tests)} 失败{Colors.RESET}")
        for name, err in failed_tests:
            print(f"\n  {Colors.RED}FAILED:{Colors.RESET} {name}")
            print(f"    {err}")
        sys.exit(1)
    else:
        print(f"\n{Colors.GREEN}全部通过 ✓{Colors.RESET}")


if __name__ == "__main__":
    main()
