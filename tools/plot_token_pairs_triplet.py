#!/usr/bin/env python3
"""
读取 rollout jsonl 末行的 token_pairs 记录（prob, adv_raw, adv_clipped），在一张图中绘制两组散点：
- (prob, adv_raw)
- (prob, adv_clipped)
不同颜色区分，便于对比。

用法示例：
    python tools/plot_token_pairs_triplet.py \
        --input /data/huaiwenzhang/projects/verl/training_rollout_metrics/grpo_advclip_entropy_sigmoid_prob0.5_+5_follow_cov/210.jsonl \
        --output /data/huaiwenzhang/projects/verl/training_rollout_metrics/grpo_advclip_entropy_sigmoid_prob0.5_+5_follow_cov/210_triplet.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


def load_triplets(jsonl_path: Path):
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    try:
        last = json.loads(lines[-1])
    except json.JSONDecodeError:
        return []
    if not isinstance(last, dict) or last.get("type") != "token_pairs":
        return []
    triplets = last.get("pairs", [])
    out = []
    for item in triplets:
        if (
            isinstance(item, dict)
            and "prob" in item
            and "adv_raw" in item
            and "adv_clipped" in item
        ):
            out.append(
                {
                    "prob": float(item["prob"]),
                    "adv_raw": float(item["adv_raw"]),
                    "adv_clipped": float(item["adv_clipped"]),
                }
            )
    return out


def main():
    parser = argparse.ArgumentParser(description="绘制 token_pairs 三元组散点图（同图对比 raw/clipped）")
    parser.add_argument("--input", type=Path, required=True, help="包含 token_pairs 的 jsonl 文件")
    parser.add_argument("--output", type=Path, required=True, help="输出图片路径")
    args = parser.parse_args()

    triplets = load_triplets(args.input)
    if not triplets:
        print(f"[skip] {args.input}: no token_pairs with adv_raw/adv_clipped found")
        return

    probs = [p["prob"] for p in triplets]
    adv_raw = [p["adv_raw"] for p in triplets]
    adv_clipped = [p["adv_clipped"] for p in triplets]

    # 尝试中文字体，缺失则回退
    try:
        simhei_path = font_manager.findfont("SimHei", fallback_to_default=False)
        font_prop = font_manager.FontProperties(fname=simhei_path)
    except Exception:
        font_prop = None

    plt.figure(figsize=(6, 6))
    plt.scatter(
        probs,
        adv_raw,
        s=3,
        alpha=0.25,
        facecolors="none",
        edgecolors="tab:blue",
        linewidths=0.35,
        label="adv_raw",
    )
    plt.scatter(
        probs,
        adv_clipped,
        s=3,
        alpha=0.25,
        facecolors="none",
        edgecolors="tab:orange",
        linewidths=0.35,
        label="adv_clipped",
    )
    if font_prop:
        plt.xlabel("prob", fontproperties=font_prop)
        plt.ylabel("adv", fontproperties=font_prop)
    else:
        plt.xlabel("prob")
    plt.ylabel("adv")
    plt.title(args.input.stem)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output)
    plt.close()
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
