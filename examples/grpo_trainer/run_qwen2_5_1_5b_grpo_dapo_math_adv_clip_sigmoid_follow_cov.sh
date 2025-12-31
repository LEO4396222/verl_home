#!/usr/bin/env bash
set -x

export CUDA_VISIBLE_DEVICES=4,5,6,7


MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct
DATA_PATH=/data/huaiwenzhang/Datasets


dapo_math_train_path=$DATA_PATH/dapo_math/train.parquet
dapo_math_test_path=$DATA_PATH/dapo_math/test.parquet

train_files="['$dapo_math_train_path']"
test_files="['$dapo_math_test_path']"

# 默认启用 Qwen-Math Prompt 模版（包含 \boxed{} 约束，利于奖励解析）。
USE_QWEN_MATH_TEMPLATE=${USE_QWEN_MATH_TEMPLATE:-1}
QWEN_MATH_TEMPLATE_PATH=${QWEN_MATH_TEMPLATE_PATH:-"$(dirname "$0")/qwen_math_chat_template.jinja"}

# 可选：过滤当批全对/全错的 prompts（仅跳过当前 batch，后续仍可采样）
FILTER_ALL_CORRECT_WRONG=${FILTER_ALL_CORRECT_WRONG:-true}

# 可选：剥离 dapo_math user 首段 prompt 前缀（-0表示不执行，-1表示执行）
STRIP_DAPO_MATH_USER_PREFIX=${STRIP_DAPO_MATH_USER_PREFIX:-1}
DAPO_MATH_USER_PREFIX='Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n'

# 可选：剥离 dapo_math user 首段 prompt 前缀（-0表示不执行，-1表示执行）
STRIP_DAPO_MATH_USER_PREFIX=${STRIP_DAPO_MATH_USER_PREFIX:-1}
DAPO_MATH_USER_PREFIX='Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n'

extra_args=()
if [[ "$USE_QWEN_MATH_TEMPLATE" == "1" ]]; then
    if [[ ! -f "$QWEN_MATH_TEMPLATE_PATH" ]]; then
        echo "Qwen math chat template not found: $QWEN_MATH_TEMPLATE_PATH" >&2
        exit 1
    fi
    # 压平模版便于 CLI 传参，模板为 Jinja 格式。
    qwen_math_template_compact=$(tr '\n' ' ' < "$QWEN_MATH_TEMPLATE_PATH")
    # 用单引号包裹，避免 Hydra 解析空格和特殊符号。
    extra_args+=("+data.apply_chat_template_kwargs.chat_template='${qwen_math_template_compact}'")
    extra_args+=("+data.apply_chat_template_kwargs.chat_template_format=jinja")
fi
if [[ "$STRIP_DAPO_MATH_USER_PREFIX" == "1" ]]; then
    extra_args+=("data.strip_user_prompt_prefix=true")
    extra_args+=("+data.user_prompt_prefix='${DAPO_MATH_USER_PREFIX}'")
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=False \
    data.truncation=left \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.policy_loss.advantage_clip.enable=True \
    actor_rollout_ref.actor.policy_loss.advantage_clip.mode=sigmoid \
    actor_rollout_ref.actor.policy_loss.advantage_clip.sigmoid_p0_prob=0.5 \
    actor_rollout_ref.actor.policy_loss.advantage_clip.sigmoid_p0_quantile=0.5 \
    actor_rollout_ref.actor.policy_loss.advantage_clip.sigmoid_alpha_pos=5.0 \
    actor_rollout_ref.actor.policy_loss.advantage_clip.sigmoid_alpha_neg=null \
    trainer.cliped_token_pair_enable=True \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    trainer.resume_mode=disable \
    trainer.logger='["console","swanlab"]' \
    trainer.rollout_data_dir=/data/huaiwenzhang/projects/verl/training_rollout_metrics/grpo_advclip_entropy_sigmoid_prob0.5_+5_follow_cov \
    trainer.project_name='verl_grpo_qwen2_5_1_5b_gsm8k_math' \
    trainer.experiment_name='grpo_advclip_entropy_sigmoid_prob0.5_+5_follow_cov' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.total_epochs=15 \
    trainer.filter_all_correct_wrong=true \
    "${extra_args[@]}" \
    "$@"
