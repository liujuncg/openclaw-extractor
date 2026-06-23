#!/usr/bin/env python3
"""
最终导出过滤器 — 从 dpo_pairs.jsonl / sft_positive.jsonl 生成训练就绪文件。

输出:
  dpo_train_main.jsonl          high + same_session_recovery_vs_synthetic_plain_retry
  dpo_train_extended.jsonl      medium quality_comparison（人工抽样后可作扩展集）
  sft_positive_with_trajectory.jsonl   SFT 正样本（含 messages + trajectory）

用法:
  python3 export_dpo.py \
    --pairs  /path/to/dpo_pairs.jsonl \
    --sft    /path/to/sft_positive.jsonl \
    --output /path/to/output_dir/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def export(pairs_path: Path, sft_path: Path, output_dir: Path) -> None:
    pairs = load_jsonl(pairs_path)
    sft_samples = load_jsonl(sft_path)

    # ── dpo_train_main: high + same_session_recovery_vs_synthetic_plain_retry ──
    main_dpo = [
        p for p in pairs
        if p["pair_quality"] == "high"
        and p.get("metadata", {}).get("pair_construction")
            == "same_session_recovery_vs_synthetic_plain_retry"
    ]

    # 加 loss_weight：
    #   strong preference_strength → 1.0（最干净的偏好信号）
    #   synthetic_rejected         → 0.5（rejected 是合成的 plain retry）
    #   真实 rejected              → 1.0
    weighted_main = [
        {**p, "loss_weight": 1.0 if p.get("metadata", {}).get("preference_strength") == "strong"
                            else (0.5 if p.get("metadata", {}).get("synthetic_rejected") else 1.0)}
        for p in main_dpo
    ]

    strong = [p for p in weighted_main
              if p.get("metadata", {}).get("preference_strength") == "strong"]

    # ── dpo_train_extended: medium quality_comparison ──
    extended = [
        p for p in pairs
        if p["pair_quality"] == "medium"
        and p["pair_type"] == "quality_comparison"
    ]

    # ── sft_positive_with_trajectory ──
    sft_out = [
        s for s in sft_samples
        if s.get("messages") and len(s["messages"]) >= 3
        and s.get("trajectory")
    ]

    # ── 写出 ──
    write_jsonl(weighted_main, output_dir / "dpo_train_main.jsonl")
    write_jsonl(extended,     output_dir / "dpo_train_extended.jsonl")
    write_jsonl(sft_out,      output_dir / "sft_positive_with_trajectory.jsonl")

    # ── 汇报 ──
    print(f"dpo_train_main.jsonl:        {len(weighted_main):4d} 条"
          f"（strong={len(strong)}, medium_pref={len(weighted_main) - len(strong)}）")
    print(f"dpo_train_extended.jsonl:    {len(extended):4d} 条")
    print(f"sft_positive_with_trajectory.jsonl: {len(sft_out):4d} 条")

    if weighted_main:
        print(f"\n  loss_weight 分布:")
        from collections import Counter
        wc = Counter(p["loss_weight"] for p in weighted_main)
        for w, c in sorted(wc.items()):
            print(f"    {w}: {c}")

    if weighted_main:
        print(f"\n  diverge_reason 分布:")
        from collections import Counter
        rc = Counter(p["diverge_reason"] for p in weighted_main)
        for r, c in sorted(rc.items(), key=lambda x: -x[1]):
            print(f"    {r}: {c}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="最终导出过滤器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--pairs",  required=True, help="dpo_pairs.jsonl 路径")
    ap.add_argument("--sft",    required=True, help="sft_positive.jsonl 路径")
    ap.add_argument("--output", required=True, help="输出目录")
    args = ap.parse_args()

    export(Path(args.pairs), Path(args.sft), Path(args.output))


if __name__ == "__main__":
    main()
