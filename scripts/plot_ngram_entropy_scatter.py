#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt

INPUT_PATH = Path("/data/huaiwenzhang/projects/verl/training_rollout_metrics/grpo_advclip_baseline_follow_cov_new/250.jsonl")
OUTPUT_PATH_TRUE = INPUT_PATH.with_name(
    f"{INPUT_PATH.stem}_ngram_entropy_scatter_acc_true.png"
)
OUTPUT_PATH_FALSE = INPUT_PATH.with_name(
    f"{INPUT_PATH.stem}_ngram_entropy_scatter_acc_false.png"
)
N_GRAM = 3
STRIP_TOKENS = ("<|im start|>user", "<|im end|>")


def clean_text(text: str) -> str:
    for tok in STRIP_TOKENS:
        text = text.replace(tok, " ")
    return " ".join(text.split())


def ngram_repetition(tokens, n: int) -> float:
    if n <= 0:
        raise ValueError("N_GRAM must be >= 1")
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    total = len(ngrams)
    unique = len(set(ngrams))
    return 1.0 - unique / total


def main() -> None:
    xs_true = []
    ys_true = []
    xs_false = []
    ys_false = []
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            entropy = obj.get("avg_token_entropy")
            output = obj.get("output", "")
            acc = obj.get("acc")
            if entropy is None:
                continue
            text = clean_text(output)
            tokens = text.split()
            rep = ngram_repetition(tokens, N_GRAM)
            if acc is True:
                xs_true.append(entropy)
                ys_true.append(rep)
            elif acc is False:
                xs_false.append(entropy)
                ys_false.append(rep)

    if not xs_true and not xs_false:
        raise SystemExit("No valid records found")

    OUTPUT_PATH_TRUE.parent.mkdir(parents=True, exist_ok=True)
    if xs_true:
        plt.figure(figsize=(7, 4.5))
        plt.scatter(xs_true, ys_true, s=14, alpha=0.6)
        plt.xlabel("avg_token_entropy")
        plt.ylabel(f"{N_GRAM}-gram repetition")
        plt.title("N-gram repetition vs avg_token_entropy (acc=true)")
        plt.tight_layout()
        plt.savefig(OUTPUT_PATH_TRUE, dpi=200)
        print(f"Saved: {OUTPUT_PATH_TRUE}")

    if xs_false:
        plt.figure(figsize=(7, 4.5))
        plt.scatter(xs_false, ys_false, s=14, alpha=0.6)
        plt.xlabel("avg_token_entropy")
        plt.ylabel(f"{N_GRAM}-gram repetition")
        plt.title("N-gram repetition vs avg_token_entropy (acc=false)")
        plt.tight_layout()
        plt.savefig(OUTPUT_PATH_FALSE, dpi=200)
        print(f"Saved: {OUTPUT_PATH_FALSE}")


if __name__ == "__main__":
    main()
