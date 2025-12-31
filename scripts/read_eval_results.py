#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取 VERL 评测结果 Parquet 文件，输出聚合指标与若干样例，可选导出 CSV。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_INPUT_PATH = "/data/huaiwenzhang/projects/verl/test_results/grpo_loss_reweight_5_follow_cov_aime2024.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="读取 VERL 评测结果并打印摘要")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help=f"评测结果 parquet 路径，默认: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=30,
        help="展示的样例条数，默认为 3",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        help="可选：将结果另存为 CSV 的输出路径",
    )
    return parser.parse_args()


def safe_to_str(obj: Any) -> str:
    if isinstance(obj, (dict, list)):
        return json.dumps(obj, ensure_ascii=False)
    return str(obj)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    df = pd.read_parquet(input_path)
    total = len(df)

    lines = []
    lines.append(f"载入 {total} 条记录，字段: {list(df.columns)}")

    if "data_source" in df.columns:
        counts = df["data_source"].value_counts()
        lines.append("\n按 data_source 统计：")
        for name, cnt in counts.items():
            lines.append(f"  {name}: {cnt}")

    if args.sample_count > 0:
        lines.append(f"\n展示前 {min(args.sample_count, total)} 条样例：")
        sample_df = df.head(args.sample_count)
        for idx, row in sample_df.iterrows():
            gt = row.get("reward_model", {})
            responses = row.get("responses", [])
            first_resp = responses[0] if isinstance(responses, (list, tuple)) and responses else responses
            lines.append(f"- 样例 {idx}: data_source={row.get('data_source')}, ability={row.get('ability')}")
            lines.append(f"  ground_truth: {safe_to_str(gt)}")
            lines.append(f"  response: {safe_to_str(first_resp)}\n")

    if args.export_csv:
        output_csv = Path(args.export_csv)
        df.to_csv(output_csv, index=False)
        lines.append(f"已导出 CSV 到 {output_csv}")

    # 输出到同目录 txt 文件（覆盖写）
    out_txt = Path(args.input).with_suffix(".txt")
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"摘要已写入 {out_txt}")


if __name__ == "__main__":
    main()
