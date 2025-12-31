import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="合并后的 HF 模型目录")
    parser.add_argument(
        "--data-path",
        required=True,
        help="输入 parquet，需包含 prompt 列（Qwen chat 格式）和 data_source/reward_model 等评估字段",
    )
    parser.add_argument("--output-path", required=True, help="输出 parquet 路径")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    df = pd.read_parquet(args.data_path)
    prompts = df["prompt"].tolist()

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    outputs = []
    for i in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[i : i + args.batch_size]
        # 先生成字符串，再统一分词，显式提供 attention_mask，避免 pad_token=eos 引发的不确定行为
        chat_texts = tok.apply_chat_template(
            [p.tolist() if isinstance(p, np.ndarray) else p for p in batch_prompts],
            add_generation_prompt=True,
            tokenize=False,
        )
        enc = tok(
            chat_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
            return_attention_mask=True,
        )
        input_ids = enc.input_ids.to(model.device)
        attention_mask = enc.attention_mask.to(model.device)
        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.temperature > 0,
            )
        for j in range(gen.size(0)):
            output_tokens = gen[j, input_ids.size(1) :].tolist()
            text = tok.decode(output_tokens, skip_special_tokens=True)
            outputs.append(text)

    df = df.copy()
    df["responses"] = pd.Series([np.array([t], dtype=object) for t in outputs])
    Path(os.path.dirname(args.output_path)).mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output_path)
    print(f"wrote {len(df)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
