#!/usr/bin/env python3
"""将 FSDP 分片的 actor 权重合并为 HuggingFace 单文件权重."""

from pathlib import Path

from verl.model_merger.base_model_merger import ModelMergerConfig
from verl.model_merger.fsdp_model_merger import FSDPModelMerger


def main():
    # 源分片所在目录（指向 actor 子目录）
    ckpt_dir = Path(
        "/data/huaiwenzhang/projects/verl/checkpoints/verl_grpo_qwen2_5_1_5b_gsm8k_math/grpo_advclip_entropy_sigmoid_prob0.5_+5_follow_cov/global_step_200/actor"
    )
    # 合并后模型输出目录
    target_dir = Path(
        "/data/huaiwenzhang/projects/verl/checkpoints/verl_grpo_qwen2_5_1_5b_gsm8k_math/grpo_advclip_entropy_sigmoid_prob0.5_+5_follow_cov/merged_global_step_200"
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    # 手动构造 merger 配置，避免命令行依赖
    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        target_dir=str(target_dir),
        hf_upload_path=None,
        private=False,
        test_hf_dir=None,
        tie_word_embedding=False,
        trust_remote_code=True,
        is_value_model=False,
        local_dir=str(ckpt_dir),
        hf_model_config_path=str(ckpt_dir / "huggingface"),
        use_cpu_initialization=False,
    )

    merger = FSDPModelMerger(config)
    merger.merge_and_save()
    merger.cleanup()


if __name__ == "__main__":
    main()
