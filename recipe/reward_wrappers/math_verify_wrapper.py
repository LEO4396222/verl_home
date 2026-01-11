from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score import math_verify


_MATH_VERIFY_SOURCES = {
    "olympiabench",
    "openai/gsm8k",
    "aime_2024",
    "aime_2025",
    "amc23",
    "huggingfaceh4/math-500",
    "hothan/olympiadbench/oe_mm_maths_en_comp",
}


def _normalize_source(data_source):
    if not isinstance(data_source, str):
        return ""
    return data_source.strip().lower()

def _to_float(value):
    if isinstance(value, dict):
        if "score" in value:
            value = value["score"]
        elif "acc" in value:
            value = value["acc"]
        else:
            return 0.0
    if isinstance(value, (list, tuple)):
        value = value[0] if value else 0.0
    return float(value)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    # 指定数据源使用 Math-Verify，失败则回退默认评分
    source_norm = _normalize_source(data_source)
    if source_norm in _MATH_VERIFY_SOURCES:
        try:
            return _to_float(math_verify.compute_score(solution_str, ground_truth))
        except Exception:
            return _to_float(default_compute_score(data_source, solution_str, ground_truth, extra_info=extra_info))
    return _to_float(default_compute_score(data_source, solution_str, ground_truth, extra_info=extra_info))
