#!/usr/bin/env python3
"""
读取 rollout jsonl 末行的 token_pairs 记录（prob, adv_clipped），绘制散点图。
用法：
    python tools/plot_token_pairs.py \
        --input /data/huaiwenzhang/projects/verl/training_rollout_metrics/grpo_4x4090_advclip_sigmoid_p0.6_+5_-5_follow_cov/10.jsonl \
        --output /data/huaiwenzhang/projects/verl/training_rollout_metrics/grpo_4x4090_advclip_sigmoid_p0.6_+5_-5_follow_cov/10_cliped_token_pairs.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


def load_pairs(jsonl_path: Path):
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    try:
        last = json.loads(lines[-1])
    except json.JSONDecodeError:
        return []
    if not isinstance(last, dict) or last.get("type") != "token_pairs":
        return []
    pairs = last.get("pairs", [])
    out = []
    for item in pairs:
        if isinstance(item, dict) and "prob" in item and "adv_clipped" in item:
            out.append({"prob": float(item["prob"]), "adv_clipped": float(item["adv_clipped"])})
    return out


def main():
    parser = argparse.ArgumentParser(description="绘制 token_pairs 散点图")
    parser.add_argument("--input", type=Path, required=True, help="包含 token_pairs 的 jsonl 文件")
    parser.add_argument("--output", type=Path, required=True, help="输出图片路径")
    args = parser.parse_args()

    pairs = load_pairs(args.input)
    if not pairs:
        print(f"[skip] {args.input}: no token_pairs found")
        return

    x = [p["prob"] for p in pairs]
    y = [p["adv_clipped"] for p in pairs]

    # 尝试中文字体，缺失则回退
    try:
        simhei_path = font_manager.findfont("SimHei", fallback_to_default=False)
        font_prop = font_manager.FontProperties(fname=simhei_path)
    except Exception:
        font_prop = None

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=3, alpha=0.2, facecolors="none", edgecolors="tab:blue", linewidths=0.35)
    if font_prop:
        plt.xlabel("prob", fontproperties=font_prop)
        plt.ylabel("adv_clipped", fontproperties=font_prop)
    else:
        plt.xlabel("prob")
        plt.ylabel("adv_clipped")
    plt.title(args.input.stem)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output)
    plt.close()
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
