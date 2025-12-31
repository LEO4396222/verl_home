#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较两个 HuggingFace 模型在同一提示下的生成结果（单轮 Chat 模式）。

示例：
  python scripts/compare_two_models.py \
    --model-a Qwen/Qwen2.5-1.5B-Instruct \
    --model-b checkpoints/verl_grpo_qwen2_5_1_5b_gsm8k_math/grpo_4x4090_advclip_prob_+0.3_-0.7/global_step_4200/hf_merged \
    --prompt "1+1等于几？请给出最终答案。"
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str, trust_remote_code: bool = True):
    """加载模型与分词器，自动设置 pad_token，左侧对齐以便自回归生成。"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else None
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=dtype,
    )
    if device == "cpu":
        model.to(device)
    model.eval()
    return tokenizer, model


def generate_once(tokenizer, model, prompt: str, max_new_tokens: int, temperature: float, top_p: float, do_sample: bool):
    """单轮生成，使用 chat template 包装 user 提示。"""
    messages = [{"role": "user", "content": prompt}]
    chat_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(
        chat_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
    gen_ids = outputs[0, inputs["input_ids"].size(1) :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def parse_args():
    parser = argparse.ArgumentParser(description="比较两个 HF 模型的生成结果")
    parser.add_argument("--model-a", required=True, help="模型 A 路径或 HF 名称")
    parser.add_argument("--model-b", required=True, help="模型 B 路径或 HF 名称")
    parser.add_argument("--prompt", required=True, help="待生成的用户提示")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="生成长度")
    parser.add_argument("--temperature", type=float, default=0.0, help="温度，0 表示贪心")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p 采样阈值")
    parser.add_argument("--do-sample", action="store_true", help="是否采样（默认贪心）")
    parser.add_argument("--trust-remote-code", action="store_true", default=True, help="信任远程代码")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"加载模型 A: {args.model_a}")
    tok_a, mod_a = load_model(args.model_a, trust_remote_code=args.trust_remote_code)
    print(f"加载模型 B: {args.model_b}")
    tok_b, mod_b = load_model(args.model_b, trust_remote_code=args.trust_remote_code)

    print("\n=== 模型 A 输出 ===")
    out_a = generate_once(tok_a, mod_a, args.prompt, args.max_new_tokens, args.temperature, args.top_p, args.do_sample)
    print(out_a.strip())

    print("\n=== 模型 B 输出 ===")
    out_b = generate_once(tok_b, mod_b, args.prompt, args.max_new_tokens, args.temperature, args.top_p, args.do_sample)
    print(out_b.strip())


if __name__ == "__main__":
    main()
