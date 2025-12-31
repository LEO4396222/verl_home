# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Apache 2.0

"""
Offline evaluate a parquet of generations with a reward model (+ optional sampling metrics).
- test_score/{data_source}: 平均 reward
- （可选）avg@K/{data_source}: K 次 pass@1 的平均
- （可选）cons@K/{data_source}: 对 K 个生成做“答案抽取→归一化→多数投票”，与 GT 比较
"""

from collections import defaultdict, Counter
import ast
import re
import random

import hydra
import numpy as np
import pandas as pd
import ray
from omegaconf import OmegaConf
from tqdm import tqdm

from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.fs import copy_to_local
from verl.utils.reward_score import default_compute_score


# --------- 简单答案抽取与归一化（按需要改进/替换） ---------
_PATTERNS = [
    r"Final Answer\s*[:：]\s*(.+)$",
    r"答案\s*[:：]\s*(.+)$",
    r"####\s*(.+)$",
    r"\\boxed\{([^}]*)\}",
]
def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).lower()

def extract_answer(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    for p in _PATTERNS:
        m = re.search(p, text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if m:
            return _norm(m.group(1))
    # 兜底：拿最后一个“像数值/短词”的片段
    m = re.findall(r"[-+]?\d+(?:\.\d+)?|[A-Za-z]+", text)
    return _norm(m[-1]) if m else ""


def is_correct_from_score(score: float, threshold: float | None, reward_range: str) -> bool:
    """
    根据 reward 范围/阈值判断是否正确：
    - 若给出 threshold，则用 (score > threshold)
    - 否则：reward_range="negpos" 或 "[-1,1]" → score > 0；否则按 [0,1] → score > 0.5
    """
    if threshold is not None:
        return score > threshold
    if reward_range in {"negpos", "[-1,1]"}:
        return score > 0
    return score > 0.5


def _resolve_reward_fn(config):
    """Resolve reward function, fallback to默认 reward."""
    reward_fn = get_custom_reward_fn(config)

    if reward_fn is None:
        def reward_fn(data_source, solution_str, ground_truth, extra_info=None):
            return default_compute_score(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
    return reward_fn


def _normalize_ground_truth(value):
    """Normalize GT so reward函数能解析 (支持 ['64'] 之类表示)."""
    if isinstance(value, (list, tuple, set)):
        if len(value) == 1:
            return _normalize_ground_truth(next(iter(value)))
        return ",".join(str(v) for v in value)
    if isinstance(value, str):
        stripped = value.strip()
        if (stripped.startswith("[") and stripped.endswith("]")) or (
            stripped.startswith("(") and stripped.endswith(")")
        ):
            try:
                parsed = ast.literal_eval(stripped)
                return _normalize_ground_truth(parsed)
            except Exception:
                pass
    return value


@ray.remote
def process_item(config, data_source, response_lst, reward_data):
    reward_fn = _resolve_reward_fn(config)

    # 可选：强制使用 OlympiadBench 同款（prime_math）评分逻辑，便于解析 “Answer:” 风格输出
    use_prime_math = bool(
        config.data.get("force_olympia_reward", False) or config.data.get("force_prime_math_reward", False)
    )
    data_source_for_reward = "olympiabench" if use_prime_math else data_source

    # ground truth 字段名可在 config 里改：data.ground_truth_key（默认 "ground_truth"）
    gt_key = config.data.get("ground_truth_key", "ground_truth")
    ground_truth = _normalize_ground_truth(reward_data[gt_key])
    extra_info = reward_data.get("extra_info") if isinstance(reward_data, dict) else None

    def score_response(response_text: str) -> float:
        score_kwargs = {
            "data_source": data_source_for_reward,
            "solution_str": response_text,
            "ground_truth": ground_truth,
        }
        if extra_info is not None:
            score_kwargs["extra_info"] = extra_info
        score = reward_fn(**score_kwargs)
        if isinstance(score, dict):
            score = score.get("score", 0.0)
        elif isinstance(score, (list, tuple)):
            score = score[0]
        return float(score)

    # 统一成 list
    if isinstance(response_lst, str):
        responses = [response_lst]
    else:
        responses = list(response_lst)

    # 基础分：用全部 responses 的平均 reward
    scores_all = [score_response(r) for r in responses] if responses else [0.0]
    avg_score = float(np.mean(scores_all))

    # 采样指标开关
    enable_sampling_metrics = bool(config.data.get("enable_sampling_metrics", False))
    if not enable_sampling_metrics:
        return data_source, avg_score

    # 采样数量 K 与策略
    K = int(config.data.get("sampling_count", 32))
    sample_random = bool(config.data.get("sampling_sample_random", False))
    # 正确与否的阈值/范围
    reward_range = str(config.data.get("reward_range", "negpos"))  # "negpos"/"[-1,1]" 或 "[0,1]"
    correct_threshold = config.data.get("correct_threshold", None)

    # 取样 K 个生成：默认前 K；可选随机
    if not responses:
        return data_source, avg_score, 0.0, 0.0, 0.0
    if len(responses) > K:
        sample = random.sample(responses, K) if sample_random else responses[:K]
    else:
        sample = responses  # 少于 K 就用实际数量

    # avg@K：对 K 个生成分别打分→转为 0/1 → 取平均
    scores = [score_response(r) for r in sample]
    flags = [1 if is_correct_from_score(s, correct_threshold, reward_range) else 0 for s in scores]
    avgK = float(np.mean(flags)) if flags else 0.0
    passK = 1.0 if any(flags) else 0.0

    # cons@K：对 K 个生成提取答案→多数投票→与 GT 比较
    answers = [extract_answer(r) for r in sample]
    answers = [a for a in answers if a != ""]
    if answers:
        majority_ans, _ = Counter(answers).most_common(1)[0]
        # 与 GT 的“规范化字符串”比较；若 GT 非字符串、比较失败，可退回 reward_fn 判别
        gt_norm = _norm(ground_truth)
        if gt_norm:
            consK = 1.0 if majority_ans == gt_norm else 0.0
        else:
            # Fallback：把多数答案包装成一个“回复”，交给 reward_fn 判定
            consK = 1.0 if is_correct_from_score(
                score_response(majority_ans), correct_threshold, reward_range
            ) else 0.0
    else:
        consK = 0.0

    return data_source, avg_score, avgK, consK, passK


@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    local_path = copy_to_local(config.data.path, use_shm=config.data.get("use_shm", False))
    dataset = pd.read_parquet(local_path)

    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    total = len(dataset)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(**OmegaConf.to_container(config.ray_kwargs.get("ray_init", {})))

    enable_sampling_metrics = bool(config.data.get("enable_sampling_metrics", False))
    K = int(config.data.get("sampling_count", 32))

    # 按 data_source 聚合
    data_source_reward = defaultdict(list)
    data_source_avgK = defaultdict(list)
    data_source_consK = defaultdict(list)
    data_source_passK = defaultdict(list)

    # Create remote tasks
    tasks = [
        process_item.remote(config, data_sources[i], responses[i], reward_model_data[i])
        for i in range(total)
    ]

    # Process results as they come in
    with tqdm(total=total) as pbar:
        while tasks:
            done_ids, tasks = ray.wait(tasks)
            for rid in done_ids:
                out = ray.get(rid)
                if enable_sampling_metrics and len(out) == 5:
                    data_source, score, avgK_val, consK_val, passK_val = out
                    data_source_avgK[data_source].append(avgK_val)
                    data_source_consK[data_source].append(consK_val)
                    data_source_passK[data_source].append(passK_val)
                else:
                    data_source, score = out
                data_source_reward[data_source].append(score)
                pbar.update(1)

    metric_dict = {}
    for ds, rewards in data_source_reward.items():
        metric_dict[f"test_score/{ds}"] = float(np.mean(rewards))

    if enable_sampling_metrics:
        for ds, vals in data_source_avgK.items():
            metric_dict[f"avg@{K}/{ds}"] = float(np.mean(vals))
        for ds, vals in data_source_consK.items():
            metric_dict[f"cons@{K}/{ds}"] = float(np.mean(vals))
        for ds, vals in data_source_passK.items():
            metric_dict[f"pass@{K}/{ds}"] = float(np.mean(vals))

    print(metric_dict)
    ray.shutdown()


if __name__ == "__main__":
    main()
