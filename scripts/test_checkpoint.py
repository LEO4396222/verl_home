#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试训练好的检查点模型脚本

使用VERL框架的main_generation和main_eval模块来测试训练好的检查点模型。

CUDA_VISIBLE_DEVICES=0,1,2,3 python test_checkpoint.py
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='测试VERL训练好的检查点模型')
    parser.add_argument('--checkpoint-path', 
                       type=str, 
                       default='/data/huaiwenzhang/projects/verl/checkpoints/verl_grpo_qwen2_5_1_5b_gsm8k_math/grpo_4x4090_advclip_prob_+0.3_-0.7/global_step_4200',
                       help='训练好的检查点路径')
    parser.add_argument('--test-data', 
                       type=str, 
                       default='/data/huaiwenzhang/Datasets/gsm8k/test.parquet',
                       help='测试数据集的Parquet文件路径 (具体名称：dapo_math/test.parquet；aime_2024/aime_2024_eval.parquet;gsm8k/test.parquet;aime_2025/aime_2025_eval.parquet;amc23/amc23_eval.parquet;MATH_500/test.parquet;olympiaBench/olympiadbench_OE_MM_maths_en_COMP_text_eval.parquet)')
    parser.add_argument('--dataset', 
                       type=str, 
                       choices=['gsm8k', 'math', 'MATH_500', 'aime_2024', 'aime_2025', 'amc23', 'dapo_math','dapo_math_small', 'olympiaBench'],
                       help='快速选择预定义的数据集 (会覆盖--test-data参数)')
    parser.add_argument('--output-dir', 
                       type=str, 
                       default='./test_results',
                       help='输出结果目录')
    parser.add_argument('--nnodes', 
                       type=int, 
                       default=1,
                       help='使用的节点数')
    parser.add_argument('--n-gpus-per-node', 
                       type=int, 
                       default=1,
                       help='每个节点使用的GPU数量')
    parser.add_argument('--batch-size', 
                       type=int, 
                       default=8,
                       help='批处理大小')
    parser.add_argument('--prompt-length', 
                       type=int, 
                       default=1024,
                       help='提示长度限制')
    parser.add_argument('--response-length', 
                       type=int, 
                       default=4096,
                       help='生成回复的最大长度')
    parser.add_argument('--gpu-memory-utilization', 
                       type=float, 
                       default=0.85,
                       help='推理阶段 target GPU 显存利用率 (0-1)')
    parser.add_argument('--temperature', 
                       type=float, 
                       default=.0,
                       help='生成温度，0.0表示确定性生成')
    parser.add_argument('--top-p', 
                       type=float, 
                       default=1.0,
                       help='Top-p采样参数，0.0-1.0，默认1.0表示不限制词汇范围')
    parser.add_argument('--rollout-name',
                       type=str,
                       default='vllm',
                       choices=['vllm', 'hf'],
                       help='生成后端，vllm 或 hf')
    parser.add_argument('--data-use-shm',
                       type=str,
                       default=None,
                       choices=['true','false'],
                       help='data.use_shm 覆盖，可选 true/false')
    parser.add_argument('--use-hf-generation',
                       action='store_true',
                       default=False,
                       help='使用本地 HF 推理（单卡）生成，绕过 vLLM/Ray')
    parser.add_argument('--n-samples', 
                       type=int, 
                       default=1,
                       help='每个提示生成的样本数')
    parser.add_argument('--trust-remote-code', 
                       action='store_true',
                       default=True,
                       help='信任远程代码')
    parser.add_argument('--enable-sampling-metrics', 
                       action='store_true',
                       default=False,
                       help='启用avg@32和cons@32指标计算')
    parser.add_argument('--sampling-count', 
                       type=int, 
                       default=32,
                       help='采样次数，默认为32')
    parser.add_argument('--use-qwen-math-prompt', 
                       action='store_true',
                       default=False,
                       help='在 prompt 中插入 Qwen-Math system 指令')
    parser.add_argument(
        '--base-model-path',
        type=str,
        default='Qwen/Qwen2.5-1.5B-Instruct',  # 或者你本地的模型目录
        help='基座模型（HF 名称或本地目录），用于加载 tokenizer 和 config')

    args, unknown = parser.parse_known_args()
    # 支持透传 Hydra 覆盖参数，如 rollout.name=hf data.use_shm=False
    args.hydra_overrides = unknown
    return args


def run_hf_generation(args, tmp_output_path):
    """使用 HF pipeline 直接单卡生成，绕过 vLLM/Ray。"""
    print(f"[HF] 加载数据: {args.test_data}")
    df = pd.read_parquet(args.test_data)
    prompts = df["prompt"].tolist()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=args.trust_remote_code)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.bfloat16 if device == "cuda" else None,
        trust_remote_code=args.trust_remote_code,
    )
    if device == "cpu":
        model.to(device)
    model.eval()

    outputs = []
    batch_size = args.batch_size
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        chat_texts = tokenizer.apply_chat_template(
            [p.tolist() if isinstance(p, np.ndarray) else p for p in batch_prompts],
            add_generation_prompt=True,
            tokenize=False,
        )
        enc = tokenizer(
            chat_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.prompt_length,
            return_attention_mask=True,
        )
        input_ids = enc.input_ids.to(model.device)
        attention_mask = enc.attention_mask.to(model.device)
        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.response_length,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.temperature > 0,
            )
        for j in range(gen.size(0)):
            text = tokenizer.decode(gen[j, input_ids.size(1) :], skip_special_tokens=True)
            outputs.append(np.array([text], dtype=object))
        print(f"[HF] 已生成 {min(i + batch_size, len(prompts))}/{len(prompts)}")

    df = df.copy()
    df["responses"] = pd.Series(outputs)
    df.to_parquet(tmp_output_path)
    print(f"[HF] 生成完成，写入 {tmp_output_path}")


def run_generation(args, tmp_output_path):
    """运行生成过程"""
    print(f"开始使用检查点生成回复...")
    print(f"检查点路径: {args.checkpoint_path}")
    print(f"测试数据: {args.test_data}")
    print(f"输出路径: {tmp_output_path}")
    
    # 确定实际的采样次数
    n_samples = args.sampling_count if args.enable_sampling_metrics else args.n_samples
    print(f"采样次数: {n_samples}")
    
    # 构建命令
    cmd = [
    'python', '-m', 'verl.trainer.main_generation',
    f'trainer.nnodes={args.nnodes}',
    f'trainer.n_gpus_per_node={args.n_gpus_per_node}',
    f'data.path={args.test_data}',
    f'data.prompt_key=prompt',
    f'data.batch_size={args.batch_size}',
    f'data.n_samples={n_samples}',
    f'data.output_path={tmp_output_path}',
    f'model.path={args.base_model_path}',
    f'+trainer.checkpoint.load_path={args.checkpoint_path}',
    f'rollout.name={args.rollout_name}',
    f'rollout.temperature={args.temperature}',
    f'rollout.top_p={args.top_p}',
    f'rollout.prompt_length={args.prompt_length}',
    f'rollout.response_length={args.response_length}',
    f'rollout.tensor_model_parallel_size={args.n_gpus_per_node}',
    f'+rollout.pipeline_model_parallel_size={args.n_gpus_per_node}',  # 这里用 + 创建字段
    f'rollout.gpu_memory_utilization={args.gpu_memory_utilization}',
]
    if args.data_use_shm is not None:
        cmd.append(f'+data.use_shm={args.data_use_shm}')
    if getattr(args, "hydra_overrides", None):
        cmd.extend(args.hydra_overrides)
    # 若需要显式禁用 checkpoint 加载，可在 hydra_overrides 传入 trainer.checkpoint.load_path=null；
    # 为避免重复 key 失败，若用户已透传覆盖，则移除默认的 load_path 再追加透传值
    load_override = [ov for ov in cmd if ov.startswith("+trainer.checkpoint.load_path=") or ov.startswith("trainer.checkpoint.load_path=")]
    if load_override and getattr(args, "hydra_overrides", None):
        # 保留用户透传的，移除默认的
        cmd = [ov for ov in cmd if ov != load_override[0]]
    
    # 执行命令
    cmd_str = ' \
    '.join(cmd)
    print(f"执行命令:\n{cmd_str}")
    exit_code = os.system(' '.join(cmd))
    
    if exit_code != 0:
        print(f"生成过程失败，退出码: {exit_code}")
        sys.exit(1)
    
    print("生成过程完成！")


def run_evaluation(tmp_output_path, enable_sampling_metrics=False, sampling_count=32):
    """运行评估过程"""
    print("开始评估生成结果...")
    
    # 构建基础命令
    cmd = [
        'python', '-m', 'verl.trainer.main_eval',
        f'data.path={tmp_output_path}',
        'data.response_key=responses',
        'data.data_source_key=data_source',
        'data.reward_model_key=reward_model'
    ]
    
    # 如果启用采样指标，添加相关参数
    if enable_sampling_metrics:
        cmd.extend([
            f'+data.enable_sampling_metrics={enable_sampling_metrics}',
            f'+data.sampling_count={sampling_count}',
            # 设置奖励范围为[-1,1]，与VERL默认的奖励函数一致
            '+data.reward_range=negpos'
        ])
        print(f"启用采样指标计算: avg@{sampling_count} 和 cons@{sampling_count}，采样次数: {sampling_count}")
    
    # 执行命令
    cmd_str = ' \
    '.join(cmd)
    print(f"执行命令:\n{cmd_str}")
    exit_code = os.system(' '.join(cmd))
    
    if exit_code != 0:
        print(f"评估过程失败，退出码: {exit_code}")
        # 评估失败不应该导致整个测试失败
        return False
    
    print("评估过程完成！")
    return True


def main():
    """主函数"""
    args = parse_args()
    
    # 数据集路径映射
    dataset_paths = {
        'dapo_math': '/data/huaiwenzhang/Datasets/dapo_math/test.parquet',
        'dapo_math_train': '/data/huaiwenzhang/Datasets/dapo_math/train.parquet',
        'dapo_math_small': '/data/huaiwenzhang/Datasets/dapo_math/test_small.parquet',
        'aime_2024': '/data/huaiwenzhang/Datasets/aime_2024/aime_2024_eval.parquet',
        'gsm8k': '/data/huaiwenzhang/Datasets/gsm8k/test.parquet',
        'aime_2025': '/data/huaiwenzhang/Datasets/aime_2025/aime_2025_eval.parquet',
        'amc23': '/data/huaiwenzhang/Datasets/amc23/amc23_eval.parquet',
        'MATH_500': '/data/huaiwenzhang/Datasets/MATH_500/test.parquet',
        'olympiaBench': '/data/huaiwenzhang/Datasets/olympiaBench/olympiadbench_OE_MM_maths_en_COMP_text_eval.parquet',
    }
    
    # 如果指定了数据集，使用预定义的路径
    if args.dataset:
        args.test_data = dataset_paths[args.dataset]
        print(f"使用预定义数据集: {args.dataset} -> {args.test_data}")
    
    # 验证测试数据文件是否存在
    if not os.path.exists(args.test_data):
        print(f"错误: 测试数据文件不存在: {args.test_data}")
        print("可用的数据集路径:")
        for dataset, path in dataset_paths.items():
            exists = "✓ 存在" if os.path.exists(path) else "✗ 不存在"
            print(f"  --dataset={dataset}  -> {path} [{exists}]")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建临时输出路径
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    tmp_output_path = f'./tmp_generation_output_{timestamp}.parquet'
    final_output_path = output_dir / f'{timestamp}_generation_output.parquet'

    # 若启用 Qwen-Math prompt，将数据集写到临时路径
    qwen_prompt_path = None
    if args.use_qwen_math_prompt:
        print("启用 Qwen-Math prompt 注入...")
        df = pd.read_parquet(args.test_data)
        system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

        def inject_prompt(record):
            if isinstance(record, np.ndarray):
                prompt_list = record.tolist()
            elif isinstance(record, list):
                prompt_list = record
            else:
                prompt_list = [{"role": "user", "content": str(record)}]
            prompt_list = [p for p in prompt_list if p.get("role") != "system"]
            new_prompt = [{"role": "system", "content": system_prompt}] + prompt_list
            return np.array(new_prompt, dtype=object)

        df["prompt"] = df["prompt"].apply(inject_prompt)
        qwen_prompt_path = output_dir / f'{timestamp}_qwen_prompt.parquet'
        df.to_parquet(qwen_prompt_path)
        args.test_data = str(qwen_prompt_path)
        print(f"Qwen-Math prompt 数据已写入: {qwen_prompt_path}")

    try:
        start_time = time.time()
        if args.use_hf_generation:
            run_hf_generation(args, tmp_output_path)
        else:
            run_generation(args, tmp_output_path)
        generation_time = time.time() - start_time
        print(f"生成耗时: {generation_time:.2f} 秒")

        # 复制结果到最终目录
        os.rename(tmp_output_path, final_output_path)
        print(f"生成完成，结果保存在: {final_output_path}")
        print(f"总耗时: {generation_time:.2f} 秒")
        
    except Exception as e:
        print(f"测试过程中出错: {e}")
        sys.exit(1)
    finally:
        # 确保清理临时文件
        if os.path.exists(tmp_output_path):
            try:
                os.remove(tmp_output_path)
                print(f"已清理临时文件: {tmp_output_path}")
            except:
                pass
        if qwen_prompt_path and os.path.exists(qwen_prompt_path):
            try:
                os.remove(qwen_prompt_path)
                print(f"已清理临时 Qwen prompt 文件: {qwen_prompt_path}")
            except:
                pass


if __name__ == "__main__":
    main()
