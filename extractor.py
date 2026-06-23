"""
OpenClaw / Hermes 轨迹提取器 — 主流水线
用法:
  python extractor.py --input logs/ --output dataset/ --min-score 0.5
  python extractor.py --input session.jsonl --output dataset/ --format jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress,
    SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn,
)
from rich.table import Table

from models import SessionOutcome, StepType, Trajectory, TrainingUse
from parsers import (
    OutcomeInferrer, TrajectoryParser,
    TrainingUseClassifier, TrainingValueScorer,
)

console = Console()

HERMES_SYSTEM_PROMPT = (
    "You are Hermes, an intelligent agent within the OpenClaw framework. "
    "You have access to a set of Skills and tools. Always reason step-by-step, "
    "stay within your assigned Skill boundaries, and handle errors gracefully."
)


# ─────────────────────────────────────────────────────────────────
# 流水线
# ─────────────────────────────────────────────────────────────────

class ExtractionPipeline:

    def __init__(self, config: ExtractionConfig):
        self.config     = config
        self.parser     = TrajectoryParser()
        self.inferrer   = OutcomeInferrer()
        self.scorer     = TrainingValueScorer()
        self.classifier = TrainingUseClassifier()

        # 调整分类器阈值
        self.classifier.SFT_MIN_SCORE      = config.min_score
        self.classifier.RECOVERY_MIN_SCORE = config.min_score * 0.6

        # 统计
        self.stats: dict[str, int] = defaultdict(int)
        # 错误类型分布：{error_type: {"count": N, "recovered": M}}
        self.error_taxonomy: dict[str, dict] = defaultdict(lambda: {"count": 0, "recovered": 0})

    # ── 主入口 ────────────────────────────────────────────────────

    def run(self) -> ExtractionResult:
        t0 = time.time()
        console.rule("[bold cyan]OpenClaw 轨迹提取器")

        # 收集输入文件
        input_files = self._collect_files()
        if not input_files:
            console.print("[red]未找到输入文件，请检查 --input 路径")
            sys.exit(1)

        console.print(f"[cyan]发现 {len(input_files)} 个日志文件")

        # 准备输出目录
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # 输出分桶
        buckets: dict[TrainingUse, list[dict]] = defaultdict(list)
        rejected: list[dict] = []
        review_queue: list[dict] = []

        # 带进度条处理
        with Progress(
            SpinnerColumn(),
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("解析日志", total=None)

            for raw in self._iter_logs(input_files):
                self.stats["total_raw"] += 1
                progress.advance(task)

                traj, use, status = self._process_one(raw)
                if status == "parse_failed":
                    self.stats["parse_failed"] += 1
                    continue
                if status in ("filtered_min_steps", "filtered_max_steps"):
                    self.stats[f"filtered.{status}"] += 1
                    continue

                self.stats[f"use.{use.value}"] += 1

                if use == TrainingUse.LOW_VALUE:
                    rejected.append({
                        "session_id":    traj.session_id,
                        "reason":        "low_value",
                        "score":         traj.value_score.total,
                        "training_use":  traj.training_use.value,
                        "outcome":       traj.outcome.value,
                        "stats": {
                            "steps":          len(traj.steps),
                            "tool_calls":     len(traj.tool_calls),
                            "unique_tools":   list(traj.unique_tools),
                            "unique_skills":  list(traj.unique_skills),
                            "skill_switches": traj.skill_switches,
                            "error_steps":    len(traj.error_steps),
                            "has_recovery":   any(s.has_recovery_signal for s in traj.steps if s.is_reasoning or s.step_type == StepType.OUTPUT),
                        },
                        "warnings":   traj.warnings,
                        "trajectory": traj.to_dict(),
                    })
                    continue

                # UNKNOWN 结果：加入待审核队列，但不阻止有价值样本被写出
                if traj.outcome == SessionOutcome.UNKNOWN:
                    review_queue.append(traj.to_dict())
                    self.stats["needs_review"] += 1
                    if not self.config.include_unknown and use not in (
                        TrainingUse.DPO_CANDIDATE,
                        TrainingUse.RECOVERY_TRAINING,
                        TrainingUse.SKILL_VIOLATION,
                    ):
                        continue

                training_sample = self._to_training_sample(traj)
                buckets[use].append(training_sample)
                self.stats["extracted"] += 1

            progress.update(task, completed=self.stats["total_raw"],
                           total=self.stats["total_raw"])

        # 写输出
        written = self._write_outputs(buckets, rejected, review_queue)

        elapsed = time.time() - t0
        result  = ExtractionResult(stats=dict(self.stats), elapsed=elapsed, written=written)
        result.error_taxonomy = dict(self.error_taxonomy)
        self._print_summary(result)
        return result

    # ── 单条处理 ─────────────────────────────────────────────────

    def _process_one(self, raw: dict) -> Optional[tuple[Trajectory, TrainingUse, str]]:
        # 1. 解析
        traj = self.parser.parse(raw)
        if traj is None:
            return None, None, "parse_failed"

        # 2. 长度过滤
        if len(traj.steps) < self.config.min_steps:
            return None, None, "filtered_min_steps"
        if self.config.max_steps and len(traj.steps) > self.config.max_steps:
            return None, None, "filtered_max_steps"

        # 3. 推断结果
        next_msg = raw.get("metadata", {}).get("next_user_message")
        traj.outcome = self.inferrer.infer(traj, next_msg)

        # 4. 打分
        traj.value_score = self.scorer.score(traj)

        # 5. 分类
        traj.training_use = self.classifier.classify(traj)

        # 6. 收集错误类型分布（用于 FI 优先级报告）
        self._collect_error_taxonomy(traj)

        return traj, traj.training_use, "ok"

    def _collect_error_taxonomy(self, traj: Trajectory) -> None:
        """收集错误类型分布，用于 Fault Injection 优先级报告。"""
        ERROR_PATTERNS = {
            "rate_limit":   ["429", "rate limit", "rate_limit", "too many requests", "quota"],
            "timeout":      ["timeout", "timed out", "connection timeout", "read timeout"],
            "auth_error":   ["401", "403", "unauthorized", "forbidden", "permission denied"],
            "not_found":    ["404", "not found", "does not exist", "no such file"],
            "server_error": ["500", "502", "503", "internal server error", "bad gateway"],
            "parse_error":  ["json decode", "parse error", "invalid format", "syntax error"],
            "tool_missing": ["tool not found", "unknown tool", "no tool"],
        }
        recovery_step_ids = {
            s.step_id for s in traj.reasoning_steps if s.has_recovery_signal
        }
        for err_step in traj.error_steps:
            err_text = (err_step.error or "") + " " + str(err_step.output or "")
            err_low = err_text.lower()
            matched = "other_error"
            for etype, patterns in ERROR_PATTERNS.items():
                if any(p in err_low for p in patterns):
                    matched = etype
                    break
            self.error_taxonomy[matched]["count"] += 1
            # 检查错误后是否有恢复（3步内有 tool_call 或 recovery signal）
            err_idx = err_step.step_id
            for s in traj.steps:
                if s.step_id > err_idx and s.step_id <= err_idx + 4:
                    if s.step_type == StepType.TOOL_CALL or s.step_id in recovery_step_ids:
                        self.error_taxonomy[matched]["recovered"] += 1
                        break

    # ── 转换为训练格式 ────────────────────────────────────────────

    def _to_training_sample(self, traj: Trajectory) -> dict:
        base = {
            "session_id":    traj.session_id,
            "training_use":  traj.training_use.value,
            "outcome":       traj.outcome.value,
            "value_score":   traj.value_score.to_dict(),
            "agent":         traj.agent,
            "model_version": traj.model_version,
            "created_at":    traj.created_at.isoformat() if traj.created_at else None,
            "stats": {
                "steps":          len(traj.steps),
                "tool_calls":     len(traj.tool_calls),
                "unique_tools":   list(traj.unique_tools),
                "unique_skills":  list(traj.unique_skills),
                "skill_switches": traj.skill_switches,
                "error_steps":    len(traj.error_steps),
                "has_recovery":   any(s.has_recovery_signal for s in traj.steps if s.is_reasoning or s.step_type == StepType.OUTPUT),
            },
            "warnings": traj.warnings,
        }

        use = traj.training_use

        if use == TrainingUse.SFT_POSITIVE:
            base["messages"] = traj.to_sft_messages(HERMES_SYSTEM_PROMPT)
            base["trajectory"] = traj.to_dict()

        elif use == TrainingUse.RECOVERY_TRAINING:
            # 保留 messages（SFT 可用）+ 额外 trajectory（DPO 可用）
            base["messages"]        = traj.to_sft_messages(HERMES_SYSTEM_PROMPT)
            base["trajectory"]      = traj.to_dict()
            base["error_steps"]     = [s.step_id for s in traj.error_steps]
            recovery_steps = [
                s.step_id for s in traj.steps
                if (s.is_reasoning or s.step_type == StepType.OUTPUT)
                and s.has_recovery_signal
            ]
            base["recovery_steps"]  = recovery_steps
            base["has_recovery"]    = len(recovery_steps) > 0

        elif use == TrainingUse.DPO_CANDIDATE:
            base["trajectory"] = traj.to_dict()

        elif use == TrainingUse.SKILL_VIOLATION:
            base["trajectory"]      = traj.to_dict()
            base["violation_steps"] = [
                s.step_id for s in traj.tool_calls if traj.has_skill_violations
            ]

        elif use == TrainingUse.GRACEFUL_ABORT:
            base["messages"]   = traj.to_sft_messages(HERMES_SYSTEM_PROMPT)
            base["trajectory"] = traj.to_dict()
            base["abort_reason"] = (
                traj.final_output[:500] if traj.final_output else "unknown"
            )

        return base

    # ── 文件 I/O ──────────────────────────────────────────────────

    def _collect_files(self) -> list[Path]:
        src = self.config.input_path
        if src.is_file():
            return [src]
        if src.is_dir():
            files = []
            for ext in ("*.jsonl", "*.json", "*.log", "*.ndjson"):
                files.extend(src.rglob(ext))
            return sorted(files)
        return []

    def _iter_logs(self, files: list[Path]) -> Iterator[dict]:
        for f in files:
            yield from self._read_file(f)

    def _read_file(self, path: Path) -> Iterator[dict]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            console.print(f"[red]读取文件失败: {path} — {e}")
            return

        # JSONL（每行一个 JSON 对象）
        if path.suffix in (".jsonl", ".ndjson") or "\n{" in text[:200]:
            for i, line in enumerate(text.splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    self.stats["json_parse_error"] += 1

        # 单 JSON 文件（可能是 list 或 dict）
        elif path.suffix == ".json":
            try:
                obj = json.loads(text)
                if isinstance(obj, list):
                    yield from obj
                elif isinstance(obj, dict):
                    yield obj
            except json.JSONDecodeError:
                console.print(f"[yellow]JSON 解析失败: {path}")
                self.stats["json_parse_error"] += 1

        # 纯文本日志
        else:
            # 尝试按 session 分割
            sessions = re.split(r"(?=session[_-]?id[:\s])", text, flags=re.IGNORECASE)
            for s in sessions:
                if s.strip():
                    yield {"text": s, "source_file": str(path)}

    def _write_outputs(
        self,
        buckets: dict[TrainingUse, list[dict]],
        rejected: list[dict],
        review_queue: list[dict],
    ) -> dict[str, str]:
        written: dict[str, str] = {}
        out = self.config.output_dir
        fmt = self.config.output_format

        for use, samples in buckets.items():
            if not samples:
                continue
            fname = f"{use.value}.{fmt}"
            path  = out / fname
            self._write_samples(samples, path, fmt)
            written[use.value] = str(path)
            console.print(f"  [green]✓[/] {fname} — {len(samples)} 条")

        if rejected and self.config.save_rejected:
            path = out / f"rejected.{fmt}"
            self._write_samples(rejected, path, fmt)
            written["rejected"] = str(path)

        if review_queue:
            path = out / f"needs_review.{fmt}"
            self._write_samples(review_queue, path, fmt)
            written["needs_review"] = str(path)
            console.print(f"  [yellow]⚠[/] needs_review.{fmt} — {len(review_queue)} 条待人工标注")

        # manifest
        manifest = {
            "created_at": datetime.now(tz=__import__('datetime').timezone.utc).isoformat(),
            "config": {
                "input":     str(self.config.input_path),
                "min_score": self.config.min_score,
                "min_steps": self.config.min_steps,
            },
            "stats":   self.stats,
            "files":   written,
        }
        manifest_path = out / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        written["manifest"] = str(manifest_path)

        return written

    @staticmethod
    def _write_samples(samples: list[dict], path: Path, fmt: str) -> None:
        with path.open("w", encoding="utf-8") as f:
            if fmt == "jsonl":
                for s in samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            else:
                json.dump(samples, f, ensure_ascii=False, indent=2)

    # ── 汇总报告 ─────────────────────────────────────────────────

    @staticmethod
    def _print_summary(result: ExtractionResult) -> None:
        console.rule("[bold cyan]提取完成")

        table = Table(title="提取统计", show_header=True, header_style="bold magenta")
        table.add_column("指标", style="cyan", width=28)
        table.add_column("数量", justify="right", style="green")

        s = result.stats
        table.add_row("原始日志总条数",   str(s.get("total_raw", 0)))
        table.add_row("解析失败",          str(s.get("parse_failed", 0)))
        filtered = s.get("filtered.filtered_min_steps", 0) + s.get("filtered.filtered_max_steps", 0)
        if filtered:
            table.add_row("步骤数过滤",    str(filtered))
        table.add_row("成功提取",          str(s.get("extracted", 0)))
        table.add_row("── SFT 正样本",     str(s.get(f"use.{TrainingUse.SFT_POSITIVE.value}", 0)))
        table.add_row("── 纠错训练样本",   str(s.get(f"use.{TrainingUse.RECOVERY_TRAINING.value}", 0)))
        table.add_row("── DPO 候选",       str(s.get(f"use.{TrainingUse.DPO_CANDIDATE.value}", 0)))
        table.add_row("── Skill 违规样本", str(s.get(f"use.{TrainingUse.SKILL_VIOLATION.value}", 0)))
        table.add_row("── 主动中止样本",   str(s.get(f"use.{TrainingUse.GRACEFUL_ABORT.value}", 0)))
        table.add_row("低价值（丢弃）",    str(s.get(f"use.{TrainingUse.LOW_VALUE.value}", 0)))
        table.add_row("待人工审核",        str(s.get("needs_review", 0)))
        table.add_row("耗时 (s)",          f"{result.elapsed:.1f}")

        console.print(table)

        extraction_rate = (
            s.get("extracted", 0) / s.get("total_raw", 1) * 100
            if s.get("total_raw") else 0
        )
        console.print(f"\n[bold]有效提取率: [cyan]{extraction_rate:.1f}%")
        if s.get("needs_review", 0) > 0:
            console.print(
                f"[yellow]提示: {s['needs_review']} 条样本结果不明确，"
                f"请查看 needs_review 文件进行人工标注"
            )

        # 错误类型分布 + FI 优先级
        taxonomy = getattr(result, "error_taxonomy", {})
        if taxonomy:
            console.rule("[bold cyan]错误类型分布 (Fault Injection 优先级)")
            fi_table = Table(title="错误恢复率 → FI 优先级", show_header=True, header_style="bold magenta")
            fi_table.add_column("错误类型", style="cyan", width=18)
            fi_table.add_column("出现次数", justify="right", style="yellow")
            fi_table.add_column("恢复成功", justify="right", style="green")
            fi_table.add_column("恢复率", justify="right")
            fi_table.add_column("FI 优先级", style="bold")

            for etype, counts in sorted(taxonomy.items(), key=lambda x: -x[1]["count"]):
                total = counts["count"]
                recovered = counts["recovered"]
                rate = recovered / total * 100 if total else 0
                # FI 优先级 = 频率 × (1 - 恢复率)，越高越需要覆盖
                fi_score = total * (1 - rate / 100)
                if fi_score >= total * 0.7:
                    fi_level = "[red]critical"
                elif fi_score >= total * 0.4:
                    fi_level = "[yellow]high"
                elif fi_score >= total * 0.15:
                    fi_level = "medium"
                else:
                    fi_level = "[green]low"
                fi_table.add_row(
                    etype, str(total), str(recovered),
                    f"{rate:.0f}%", fi_level,
                )
            console.print(fi_table)


# ─────────────────────────────────────────────────────────────────
# 配置 & 结果
# ─────────────────────────────────────────────────────────────────


class ExtractionConfig:
    def __init__(
        self,
        input_path:      str | Path,
        output_dir:      str | Path,
        min_score:       float = 0.5,
        min_steps:       int   = 2,
        max_steps:       Optional[int] = None,
        output_format:   str   = "jsonl",
        include_unknown: bool  = False,
        save_rejected:   bool  = False,
    ):
        self.input_path      = Path(input_path)
        self.output_dir      = Path(output_dir)
        self.min_score       = min_score
        self.min_steps       = min_steps
        self.max_steps       = max_steps
        self.output_format   = output_format
        self.include_unknown = include_unknown
        self.save_rejected   = save_rejected


class ExtractionResult:
    def __init__(self, stats: dict, elapsed: float, written: dict[str, str]):
        self.stats   = stats
        self.elapsed = elapsed
        self.written = written
        self.error_taxonomy: dict = {}


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="OpenClaw / Hermes 日志轨迹提取器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input",   "-i", required=True, help="输入日志文件或目录")
    ap.add_argument("--output",  "-o", required=True, help="输出目录")
    ap.add_argument("--min-score",  type=float, default=0.5,   help="最低训练价值分数阈值")
    ap.add_argument("--min-steps",  type=int,   default=2,     help="最少步骤数过滤")
    ap.add_argument("--max-steps",  type=int,   default=None,  help="最多步骤数过滤（可选）")
    ap.add_argument("--format",  choices=["jsonl", "json"], default="jsonl", help="输出格式")
    ap.add_argument("--include-unknown", action="store_true", help="保留结果不明确的样本")
    ap.add_argument("--save-rejected",   action="store_true", help="保存被丢弃的样本")
    args = ap.parse_args()

    config = ExtractionConfig(
        input_path      = args.input,
        output_dir      = args.output,
        min_score       = args.min_score,
        min_steps       = args.min_steps,
        max_steps       = args.max_steps,
        output_format   = args.format,
        include_unknown = args.include_unknown,
        save_rejected   = args.save_rejected,
    )

    pipeline = ExtractionPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
