#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from dataclasses import dataclass

@dataclass
class AdvClipParams:
    n: float                 # rollout_n（组大小），需 > 1
    p_plus: float | None     # 正侧概率阈值 (0, 1]，可为 None（表示用系数）
    p_minus: float | None    # 负侧概率阈值 (0, 1]，可为 None（表示用系数）
    positive_coef: float | None   # 正侧系数，可为 None（表示用阈值）
    negative_coef: float | None   # 负侧系数（通常为负数），可为 None（表示用阈值）
    prob_epsilon: float = 1e-8    # 与代码一致的数值稳定项

def coef_from_thresholds(n: float,
                         p_plus: float | None,
                         p_minus: float | None,
                         prob_epsilon: float = 1e-8) -> tuple[float | None, float | None]:
    """
    由概率阈值反算系数：
      positive_coef = p_plus * ((n-1)/sqrt(n))
      negative_coef = ((1-n)/sqrt(n)) / max(p_minus, eps)
    """
    if n is None or n <= 1:
        raise ValueError("n 必须 > 1")
    span_pos = (n - 1.0) / math.sqrt(n)       # > 0
    span_neg = (1.0 - n) / math.sqrt(n)       # < 0

    pos_coef = None
    neg_coef = None

    if p_plus is not None:
        if not (0.0 < p_plus <= 1.0):
            raise ValueError("p_plus 必须在 (0, 1] 内")
        pos_coef = p_plus * span_pos

    if p_minus is not None:
        if not (0.0 < p_minus <= 1.0):
            raise ValueError("p_minus 必须在 (0, 1] 内")
        denom = max(p_minus, prob_epsilon)
        neg_coef = span_neg / denom

    return pos_coef, neg_coef

def thresholds_from_coefs(n: float,
                          positive_coef: float | None,
                          negative_coef: float | None,
                          prob_epsilon: float = 1e-8) -> tuple[float | None, float | None]:
    """
    由系数反算概率阈值：
      p_plus  = positive_coef / ((n-1)/sqrt(n))
      p_minus = ((1-n)/sqrt(n)) / negative_coef
    注意：p_minus>0 通常要求 negative_coef<0。
    """
    if n is None or n <= 1:
        raise ValueError("n 必须 > 1")
    span_pos = (n - 1.0) / math.sqrt(n)       # > 0
    span_neg = (1.0 - n) / math.sqrt(n)       # < 0

    p_plus = None
    p_minus = None

    if positive_coef is not None:
        if abs(span_pos) < 1e-12:
            raise ZeroDivisionError("span_pos 过小")
        p_plus = positive_coef / span_pos

    if negative_coef is not None:
        # 为了数值稳定，避免除 0
        denom = negative_coef if abs(negative_coef) > prob_epsilon else (-prob_epsilon if negative_coef < 0 else prob_epsilon)
        p_minus = span_neg / denom

    return p_plus, p_minus

def adv_bounds_for_token_prob(q: float,
                              positive_coef: float,
                              negative_coef: float,
                              prob_epsilon: float = 1e-8) -> tuple[float, float]:
    """
    给定 token 概率 q，计算优势裁剪上下界（与你贴的实现一致）：
      adv_max = positive_coef / (q + eps)
      adv_min = negative_coef * q
    """
    if not (0.0 <= q <= 1.0):
        raise ValueError("q 必须在 [0, 1] 内")
    q_eff = max(min(q, 1.0), prob_epsilon)
    adv_max = positive_coef / (q_eff + prob_epsilon)
    adv_min = negative_coef * q_eff
    return adv_min, adv_max

def main():
    # ===== 在这里设置你的输入 =====
    # 场景 A：给概率阈值 + n，算系数
    # params = AdvClipParams(
    #     n=8,
    #     p_plus=0.30,              # prob+
    #     p_minus=0.70,             # prob-
    #     positive_coef=None,       # 留空表示通过 p_plus 反推
    #     negative_coef=None,       # 留空表示通过 p_minus 反推
    #     prob_epsilon=1e-8,
    # )

    # 场景 B：如果你反过来只给系数，也可以这样：
    params = AdvClipParams(
        n=8,
        p_plus=0.6,
        p_minus=0.05,
        positive_coef=None,
        negative_coef=None,
    )

    # ===== 计算逻辑 =====
    pos_coef, neg_coef = params.positive_coef, params.negative_coef

    # 如果缺系数，就由阈值反推
    if pos_coef is None or neg_coef is None:
        c_pos, c_neg = coef_from_thresholds(params.n, params.p_plus, params.p_minus, params.prob_epsilon)
        if pos_coef is None:
            pos_coef = c_pos
        if neg_coef is None:
            neg_coef = c_neg

    # 再把得到的系数反推回阈值，做个自检
    p_plus_check, p_minus_check = thresholds_from_coefs(params.n, pos_coef, neg_coef, params.prob_epsilon)

    print("=== 结果 ===")
    print(f"n = {params.n}")
    print(f"positive_coef = {pos_coef:.6f}" if pos_coef is not None else "positive_coef = None")
    print(f"negative_coef = {neg_coef:.6f}" if neg_coef is not None else "negative_coef = None")
    print(f"prob+ (from coef)  = {p_plus_check:.6f}" if p_plus_check is not None else "p_plus (from coef) = None")
    print(f"prob-(from coef)  = {p_minus_check:.6f}" if p_minus_check is not None else "p_minus(from coef) = None")

    

if __name__ == "__main__":
    main()
