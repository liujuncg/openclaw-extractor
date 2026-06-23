"""
DPO 配对模块 — 测试套件
运行: python tests/test_dpo_pairing.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dpo_pairing import (_extract_recovery_actions, steps_to_assistant_text,
    try_pair_success_failure, run_pairing, _classify_retry, _params_changed,
    _get_steps, _is_trivial_candidate, _check_failed_item_resolved)


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


# ── data helpers ────────────────────────────────────────────────

def make_tool_call(tool_name: str, params: dict | None = None,
                   call_id: str = "c1") -> dict:
    return {"step_type": "tool_call", "tool_name": tool_name,
            "input_params": params or {}, "call_id": call_id}

def make_tool_result(status: str = "success", output: str = "",
                     error: str = "", call_id: str = "c1") -> dict:
    return {"step_type": "tool_result", "status": status,
            "output": output, "error": error, "result_call_id": call_id}

def make_reasoning(content: str) -> dict:
    return {"step_type": "reasoning", "content": content}

def make_output(content: str) -> dict:
    return {"step_type": "output", "content": content}


# ── test: chosen excludes tool_result ───────────────────────────

def test_chosen_excludes_tool_result():
    """_extract_recovery_actions 返回的步骤不应包含 tool_result"""
    steps = [
        make_tool_result(status="error", error="429 rate limit", call_id="c0"),
        make_reasoning("数据库连接池耗尽，需要等待后重试。"),
        make_tool_call("order_query", {"user_id": "12345", "limit": 10}, "c1"),
        make_tool_result(status="success", output='{"orders": []}', call_id="c1"),
        make_tool_call("email_send", {"to": "team@co.com"}, "c2"),
        make_tool_result(status="success", output='{"ok": true}', call_id="c2"),
    ]
    actions = _extract_recovery_actions(steps, error_idx=0, max_steps=8)
    types = [a["step_type"] for a in actions]
    assert_true("tool_call" in types, "recovery actions 含 tool_call")
    assert_true("tool_result" not in types, "recovery actions 不含 tool_result")
    assert_eq(len(actions), 2, "2 个 action（2 tool_call，reasoning 被跳过）")


# ── test: synthetic rejected metadata ──────────────────────────

def test_synthetic_rejected_metadata():
    """try_pair_success_failure 构造的 synthetic pair metadata 完整性"""
    fail_traj = [
        make_tool_call("feishu_doc", {"action": "update_block", "block_id": "blk_1", "content": "hello"}, "c0"),
        make_tool_result(status="error", error="429 rate limit", call_id="c0"),
        make_reasoning("遇到 rate limit，等待 2 秒后重试。"),
        make_tool_call("exec", {"command": "sleep 2"}, "c1"),
        make_tool_result(status="success", output="done", call_id="c1"),
        make_tool_call("feishu_doc", {"action": "update_block", "block_id": "blk_1", "content": "hello"}, "c2"),
        make_tool_result(status="success", output='{"block_id": "blk_1"}', call_id="c2"),
        make_output("已完成更新。"),
    ]
    failure = {
        "session_id": "fail_001", "user_input": "更新飞书文档",
        "outcome": "success", "training_use": "recovery_training",
        "value_score": {"total": 0.7},
        "stats": {"steps": 8, "tool_calls": 3, "unique_tools": ["feishu_doc", "exec"],
                  "unique_skills": [], "skill_switches": 0, "error_steps": 1, "has_recovery": True},
        "trajectory": fail_traj,
    }
    success = {
        "session_id": "ok_001", "user_input": "更新飞书文档",
        "outcome": "success", "training_use": "sft_positive",
        "value_score": {"total": 0.8},
        "stats": {"steps": 6, "tool_calls": 2, "unique_tools": ["feishu_doc", "exec"],
                  "unique_skills": [], "skill_switches": 0, "error_steps": 0, "has_recovery": False},
        "trajectory": [
            make_tool_call("feishu_doc", {"action": "update_block", "block_id": "blk_9", "content": "ok"}, "s0"),
            make_tool_result(status="success", output='{"block_id": "blk_9"}', call_id="s0"),
            make_output("完成"),
        ],
    }
    pair = try_pair_success_failure(success, failure, min_similarity=0.5)
    assert_true(pair is not None, "应返回非 None pair")
    md = pair["metadata"]
    assert_eq(md["rejected_source"], "synthetic_plain_retry", "rejected_source")
    assert_true(md["synthetic_rejected"] is True, "synthetic_rejected is True")
    assert_eq(md["pair_construction"], "same_session_recovery_vs_synthetic_plain_retry", "pair_construction")
    assert_eq(md["chosen_source"], "same_session_recovery", "chosen_source")
    assert_true("<tool_result" not in pair["chosen"], "chosen 不含 <tool_result")
    assert_true("feishu_doc" in pair["rejected"], "rejected 含 feishu_doc")
    assert_true("update_block" in pair["rejected"], "rejected 含 update_block")
    assert_true("blk_1" in pair["rejected"], "rejected 含 blk_1")


# ── test: rate_limit preference_strength ──────────────────────

def test_rate_limit_requires_failed_item_resolved_for_strong():
    """rate_limit 场景下，preference_strength 只有 failed_item_resolved=True 时才是 strong"""
    success = {
        "session_id": "ok_001", "user_input": "更新飞书文档",
        "outcome": "success", "training_use": "sft_positive",
        "value_score": {"total": 0.8},
        "stats": {"steps": 4, "tool_calls": 1,
                   "unique_tools": ["feishu_doc"],
                   "unique_skills": [], "skill_switches": 0,
                   "error_steps": 0, "has_recovery": False},
        "trajectory": [
            make_tool_call("feishu_doc", {"action": "update_block", "block_id": "blk_9", "content": "ok"}, "s0"),
            make_tool_result(status="success", output='{"block_id": "blk_9"}', call_id="s0"),
            make_output("完成"),
        ],
    }
    # ── resolved case: block_id="bad" 429 后先做 other 成功，再做 bad 成功 ──
    resolved_traj = [
        make_tool_call("feishu_doc", {"action": "update_block", "block_id": "bad", "content": "x"}, "r0"),
        make_tool_result(status="error", error="429 rate limit", call_id="r0"),
        make_reasoning("遇到 rate limit，先处理其他 block 再回头重试。"),
        make_tool_call("feishu_doc", {"action": "update_block", "block_id": "other", "content": "y"}, "r1"),
        make_tool_result(status="success", output='{"block_id": "other"}', call_id="r1"),
        make_tool_call("feishu_doc", {"action": "update_block", "block_id": "bad", "content": "x"}, "r2"),
        make_tool_result(status="success", output='{"block_id": "bad"}', call_id="r2"),
        make_output("全部完成。"),
    ]
    resolved_failure = {
        "session_id": "fail_r", "user_input": "更新飞书文档",
        "outcome": "success", "training_use": "recovery_training",
        "value_score": {"total": 0.7},
        "stats": {"steps": 8, "tool_calls": 3, "unique_tools": ["feishu_doc"],
                  "unique_skills": [], "skill_switches": 0, "error_steps": 1, "has_recovery": True},
        "trajectory": resolved_traj,
    }
    pair_r = try_pair_success_failure(success, resolved_failure, min_similarity=0.5)
    assert_true(pair_r is not None, "resolved case 应返回非 None pair")
    assert_eq(pair_r["pair_quality"], "high", "resolved: pair_quality == high")
    md_r = pair_r["metadata"]
    assert_true(md_r["failed_item_eventually_resolved"] is True, "resolved: failed_item_eventually_resolved is True")
    assert_eq(md_r["preference_strength"], "strong", "resolved: preference_strength == strong")
    # ── unresolved case: block_id="bad" 429 后只做 other，不再回到 bad ──
    unresolved_traj = [
        make_tool_call("feishu_doc", {"action": "update_block", "block_id": "bad", "content": "x"}, "u0"),
        make_tool_result(status="error", error="429 rate limit", call_id="u0"),
        make_reasoning("遇到 rate limit，先处理其他 block。"),
        make_tool_call("feishu_doc", {"action": "update_block", "block_id": "other", "content": "y"}, "u1"),
        make_tool_result(status="success", output='{"block_id": "other"}', call_id="u1"),
        make_output("部分完成。"),
    ]
    unresolved_failure = {
        "session_id": "fail_u", "user_input": "更新飞书文档",
        "outcome": "success", "training_use": "recovery_training",
        "value_score": {"total": 0.7},
        "stats": {"steps": 6, "tool_calls": 2, "unique_tools": ["feishu_doc"],
                  "unique_skills": [], "skill_switches": 0, "error_steps": 1, "has_recovery": True},
        "trajectory": unresolved_traj,
    }
    pair_u = try_pair_success_failure(success, unresolved_failure, min_similarity=0.5)
    assert_true(pair_u is not None, "unresolved case 应返回非 None pair")
    assert_eq(pair_u["pair_quality"], "medium", "unresolved: pair_quality == medium")
    md_u = pair_u["metadata"]
    assert_true(md_u["failed_item_eventually_resolved"] is False, "unresolved: failed_item_eventually_resolved is False")
    assert_eq(md_u["preference_strength"], "medium", "unresolved: preference_strength == medium")


# ── test: near-duplicate group dedup ──────────────────────────

def test_near_duplicate_group_dedup_limits_duplicates():
    """同一 near_duplicate_group_id 最多保留 2 对"""
    success = {
        "session_id": "ok_dedup", "user_input": "更新飞书文档",
        "outcome": "success", "training_use": "sft_positive",
        "value_score": {"total": 0.9},
        "stats": {"steps": 4, "tool_calls": 1,
                   "unique_tools": ["feishu_doc"],
                   "unique_skills": [], "skill_switches": 0,
                   "error_steps": 0, "has_recovery": False},
        "trajectory": [
            make_tool_call("feishu_doc", {"action": "update_block", "block_id": "z", "content": "ok"}, "z0"),
            make_tool_result(status="success", output='{"block_id":"z"}', call_id="z0"),
            make_output("完成"),
        ],
    }

    def _make_cand(sid: str, resolved: bool) -> dict:
        """构造一个 feishu_doc update_block 429 恢复候选"""
        traj = [
            make_tool_call("feishu_doc", {"action": "update_block", "block_id": "bad", "content": "x"}, "a0"),
            make_tool_result(status="error", error="429 rate limit", call_id="a0"),
            make_reasoning("遇到 rate limit，等待后重试。"),
            make_tool_call("feishu_doc", {"action": "update_block", "block_id": "other", "content": "y"}, "a1"),
            make_tool_result(status="success", output='{"block_id":"other"}', call_id="a1"),
        ]
        if resolved:
            traj.append(make_tool_call("feishu_doc", {"action": "update_block", "block_id": "bad", "content": "x"}, "a2"))
            traj.append(make_tool_result(status="success", output='{"block_id":"bad"}', call_id="a2"))
        traj.append(make_output("完成" if resolved else "部分完成。"))
        return {
            "session_id": sid, "user_input": "更新飞书文档",
            "outcome": "success", "training_use": "recovery_training",
            "value_score": {"total": 0.7},
            "stats": {"steps": len(traj), "tool_calls": 2 + int(resolved),
                       "unique_tools": ["feishu_doc"],
                       "unique_skills": [], "skill_switches": 0,
                       "error_steps": 1, "has_recovery": True},
            "trajectory": traj,
        }

    candidates = [success] + [_make_cand(f"dup_{chr(97+i)}", i < 2) for i in range(4)]
    pairs = run_pairing(candidates, min_similarity=0.5, max_pairs_per_cluster=20)
    group_ids = [p.get("metadata", {}).get("near_duplicate_group_id", "") for p in pairs]
    # 同一 group_id 最多出现 2 次
    from collections import Counter
    cnt = Counter(gid for gid in group_ids if gid)
    for gid, c in cnt.items():
        assert_true(c <= 2, f"group {gid[:30]} 出现 {c} 次，应 ≤2")


# ── test: targetId change counts as adaptive ──────────────────

def test_browser_target_id_change_counts_as_adaptive():
    """targetId 变化应识别为 adaptive_retry（browser targetId mismatch 的恢复行为）"""
    # targetId 变化 → True
    assert_true(_params_changed(
        {"url": "https://x.com", "targetId": "tab_1"},
        {"url": "https://x.com", "targetId": "tab_2"},
    ), "仅 targetId 变化 → True")
    # call_id / result_call_id 仍然被忽略
    assert_true(not _params_changed(
        {"url": "https://x.com", "call_id": "c1"},
        {"url": "https://x.com", "call_id": "c2"},
    ), "call_id 变化 → False")
    assert_true(not _params_changed(
        {"url": "https://x.com", "result_call_id": "r1"},
        {"url": "https://x.com", "result_call_id": "r2"},
    ), "result_call_id 变化 → False")
    steps = [
        make_tool_call("browser", {"url": "https://page", "targetId": "tab_1"}, "b0"),
        make_tool_result(status="error", error="targetId not found", call_id="b0"),
        make_reasoning("tab_1 找不到，需要换到 tab_2 重试同样的 url。"),
        make_tool_call("browser", {"url": "https://page", "targetId": "tab_2"}, "b1"),
        make_tool_result(status="success", output="loaded", call_id="b1"),
        make_output("加载成功。"),
    ]
    info = _classify_retry(steps, error_idx=1)
    assert_eq(info["recovery_kind"], "adaptive_retry",
              "targetId 变化 → recovery_kind=adaptive_retry")
    assert_eq(info["retry_type"], "param_change",
              "targetId 变化 → retry_type=param_change")
    assert_true(info["resolved"] is True, "重试成功 → resolved=True")


# ── test: no exact duplicate (prompt, chosen, rejected) ─────────

def test_run_pairing_does_not_emit_exact_duplicate_prompt_chosen_rejected():
    """run_pairing 不应输出 (prompt, chosen, rejected) 完全相同的 pair"""
    success = {
        "session_id": "ok_no_dup", "user_input": "更新飞书文档",
        "outcome": "success", "training_use": "sft_positive",
        "value_score": {"total": 0.9},
        "stats": {"steps": 4, "tool_calls": 1,
                   "unique_tools": ["feishu_doc"],
                   "unique_skills": [], "skill_switches": 0,
                   "error_steps": 0, "has_recovery": False},
        "trajectory": [
            make_tool_call("feishu_doc", {"action": "update_block", "block_id": "z", "content": "ok"}, "z0"),
            make_tool_result(status="success", output='{"block_id":"z"}', call_id="z0"),
            make_output("完成"),
        ],
    }
    def _make_cand(sid: str) -> dict:
        traj = [
            make_tool_call("feishu_doc", {"action": "update_block", "block_id": "bad", "content": "x"}, "a0"),
            make_tool_result(status="error", error="429 rate limit", call_id="a0"),
            make_reasoning("遇到 rate limit，等待后重试。"),
            make_tool_call("feishu_doc", {"action": "update_block", "block_id": "bad", "content": "x"}, "a1"),
            make_tool_result(status="success", output='{"block_id":"bad"}', call_id="a1"),
            make_output("完成"),
        ]
        return {
            "session_id": sid, "user_input": "更新飞书文档",
            "outcome": "success", "training_use": "recovery_training",
            "value_score": {"total": 0.7},
            "stats": {"steps": 6, "tool_calls": 2,
                       "unique_tools": ["feishu_doc"],
                       "unique_skills": [], "skill_switches": 0,
                       "error_steps": 1, "has_recovery": True},
            "trajectory": traj,
        }
    candidates = [success] + [_make_cand(f"dup_{chr(97+i)}") for i in range(3)]
    pairs = run_pairing(candidates, min_similarity=0.5, max_pairs_per_cluster=20)
    # 三元组去重
    seen = set()
    for p in pairs:
        key = (json.dumps(p["prompt"], ensure_ascii=False),
               p["chosen"], p["rejected"])
        assert_true(key not in seen, "不应有完全相同的 (prompt, chosen, rejected) 对")
        seen.add(key)


# ── test: high-quality pair has required metadata fields ────────

def test_high_pair_has_required_metadata_fields():
    """pair_quality=high 的 pair 应包含完整 metadata 字段"""
    # intent_only 场景：声明重试但没执行 → quality=high
    fail_traj = [
        make_tool_call("feishu_doc", {"action": "update_block", "block_id": "bad", "content": "x"}, "h0"),
        make_tool_result(status="error", error="429 rate limit", call_id="h0"),
        make_reasoning("遇到 rate limit，需要重试 rate limit 的 block。"),
        make_output("任务失败。"),
    ]
    failure = {
        "session_id": "intent_fail", "user_input": "更新飞书文档",
        "outcome": "success", "training_use": "recovery_training",
        "value_score": {"total": 0.7},
        "stats": {"steps": 4, "tool_calls": 1,
                   "unique_tools": ["feishu_doc"],
                   "unique_skills": [], "skill_switches": 0,
                   "error_steps": 1, "has_recovery": True},
        "trajectory": fail_traj,
    }
    success = {
        "session_id": "ok_intent", "user_input": "更新飞书文档",
        "outcome": "success", "training_use": "sft_positive",
        "value_score": {"total": 0.8},
        "stats": {"steps": 4, "tool_calls": 1,
                   "unique_tools": ["feishu_doc"],
                   "unique_skills": [], "skill_switches": 0,
                   "error_steps": 0, "has_recovery": False},
        "trajectory": [
            make_tool_call("feishu_doc", {"action": "update_block", "block_id": "z", "content": "ok"}, "z0"),
            make_tool_result(status="success", output='{"block_id":"z"}', call_id="z0"),
            make_output("完成"),
        ],
    }
    pair = try_pair_success_failure(success, failure, min_similarity=0.5)
    assert_true(pair is not None, "应返回非 None pair")
    assert_eq(pair["pair_quality"], "high", "pair_quality == high")
    md = pair["metadata"]
    required = [
        "success_session", "failure_session", "success_score",
        "failure_score", "success_steps", "failure_steps",
        "error_at_step", "recovery_kind", "preference_strength",
        "pair_construction", "chosen_source", "rejected_source",
        "synthetic_rejected", "near_duplicate_group_id",
    ]
    for field in required:
        assert_true(field in md, f"metadata 含 {field}")



# ── test: _get_steps 兼容多种 trajectory 格式 ─────────────────

def test_get_steps_supports_all_formats():
    """_get_steps 兼容 4 种格式：list, dict.trajectory, dict.steps, sample.steps"""
    steps_data = [
        {"step_type": "tool_call", "tool_name": "read"},
        {"step_type": "tool_result", "status": "success"},
    ]
    # 格式 1: trajectory 是 list
    s1 = {"trajectory": steps_data}
    assert_eq(_get_steps(s1), steps_data, "format 1: trajectory is list")
    # 格式 2: trajectory 是 dict，含 "trajectory" key
    s2 = {"trajectory": {"trajectory": steps_data, "user_input": "test"}}
    assert_eq(_get_steps(s2), steps_data, "format 2: trajectory.trajectory")
    # 格式 3: trajectory 是 dict，含 "steps" key
    s3 = {"trajectory": {"steps": steps_data, "user_input": "test"}}
    assert_eq(_get_steps(s3), steps_data, "format 3: trajectory.steps")
    # 格式 4: 顶层 "steps" key
    s4 = {"steps": steps_data}
    assert_eq(_get_steps(s4), steps_data, "format 4: sample.steps")
    # 格式 5: trajectory 缺失 → 空列表
    s5 = {"session_id": "x"}
    assert_eq(_get_steps(s5), [], "format 5: no trajectory → empty list")


# ── test: _is_trivial_candidate 兼容多种格式 ──────────────────

def test_is_trivial_candidate_supports_all_formats():
    """_is_trivial_candidate 通过 _get_steps 兼容所有格式"""
    good_steps = [
        {"step_type": "tool_call", "tool_name": "read"},
        {"step_type": "tool_result", "status": "success"},
        {"step_type": "tool_call", "tool_name": "write"},
        {"step_type": "tool_result", "status": "success"},
        {"step_type": "output", "content": "done"},
    ]
    bad_steps = [
        {"step_type": "reasoning", "content": "thinking"},
        {"step_type": "output", "content": "done"},
    ]
    # 格式 1: trajectory 是 list
    assert_true(not _is_trivial_candidate({"trajectory": good_steps}), "format 1 good: not trivial")
    assert_true(_is_trivial_candidate({"trajectory": bad_steps}), "format 1 bad: trivial (no tool_call)")
    # 格式 2: trajectory 是 dict，含 "trajectory" key
    assert_true(not _is_trivial_candidate({"trajectory": {"trajectory": good_steps}}), "format 2 good: not trivial")
    assert_true(_is_trivial_candidate({"trajectory": {"trajectory": bad_steps}}), "format 2 bad: trivial")
    # 格式 3: trajectory 是 dict，含 "steps" key
    assert_true(not _is_trivial_candidate({"trajectory": {"steps": good_steps}}), "format 3 good: not trivial")
    assert_true(_is_trivial_candidate({"trajectory": {"steps": bad_steps}}), "format 3 bad: trivial")
    # 格式 4: 顶层 "steps" key
    assert_true(not _is_trivial_candidate({"steps": good_steps}), "format 4 good: not trivial")
    assert_true(_is_trivial_candidate({"steps": bad_steps}), "format 4 bad: trivial")
    # 格式 5: 空 trajectory → trivial
    assert_true(_is_trivial_candidate({"trajectory": []}), "empty trajectory: trivial")


# ── test: backoff-only retry 不再被误判为 no_retry ──────────────

def test_backoff_only_retry_not_no_retry():
    """P0 修复回归：做了 exec sleep(backoff) 但没重试原工具 → 应为 backoff_retry 而非 no_retry"""
    steps = [
        make_tool_call("feishu_doc", {"action": "update_block", "block_id": "blk_1", "content": "x"}, "c0"),
        make_tool_result(status="error", error="429 rate limit", call_id="c0"),
        make_reasoning("遇到 rate limit，等待 2 秒后重试。"),
        make_tool_call("exec", {"command": "sleep 2"}, "c1"),
        make_tool_result(status="success", output="done", call_id="c1"),
        # 注意：没有重试 feishu_doc，只是做了 sleep
        make_output("已完成等待。"),
    ]
    info = _classify_retry(steps, error_idx=1)
    assert_eq(info["recovery_kind"], "backoff_retry",
              "backoff-only (no original tool retry) → backoff_retry (was no_retry before fix)")
    assert_eq(info["retry_type"], "backoff", "retry_type == backoff")


# ── test: 跨工具恢复匹配 ────────────────────────────────────────

def test_cross_tool_recovery_resolved():
    """P2 修复回归：feishu_doc 失败后用 browser 处理同一 block_id → resolved=True"""
    steps = [
        make_tool_call("feishu_doc", {"action": "update_block", "block_id": "blk_1", "content": "x"}, "c0"),
        make_tool_result(status="error", error="429 rate limit", call_id="c0"),
        make_reasoning("飞书 API 限流，改用浏览器直接操作同一 block。"),
        # 换工具但同一 block_id
        make_tool_call("browser", {"action": "navigate", "url": "https://docs.feishu.cn/blk_1", "block_id": "blk_1"}, "c1"),
        make_tool_result(status="success", output='{"block_id": "blk_1"}', call_id="c1"),
        make_output("已通过浏览器完成更新。"),
    ]
    result = _check_failed_item_resolved(steps, error_idx=1)
    assert_true(result is True, "跨工具恢复同一 block_id → resolved=True")


# ── main ────────────────────────────────────────────────────────

def main():
    print("\n\033[1mDPO 配对模块 — 测试套件\033[0m")
    print("=" * 50)
    tests = [test_chosen_excludes_tool_result, test_synthetic_rejected_metadata,
             test_rate_limit_requires_failed_item_resolved_for_strong,
             test_near_duplicate_group_dedup_limits_duplicates,
             test_browser_target_id_change_counts_as_adaptive,
             test_run_pairing_does_not_emit_exact_duplicate_prompt_chosen_rejected,
             test_high_pair_has_required_metadata_fields,
             test_get_steps_supports_all_formats,
             test_is_trivial_candidate_supports_all_formats,
             test_backoff_only_retry_not_no_retry,
             test_cross_tool_recovery_resolved]
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