#!/usr/bin/env bash
set -x

export WANDB_API_KEY=YOUR_WANDB_API_KEY
# export VLLM_USE_V1=1

# NCCL (align with Baselines/DAPO baseline format)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=LOC
export NCCL_CUMEM_HOST_ENABLE=0



MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct
DATA_PATH=/data/huaiwenzhang/Datasets

dapo_math_train_path=$DATA_PATH/dapo_math/train.parquet
dapo_math_test_path=$DATA_PATH/dapo_math/test.parquet

train_files="['$dapo_math_train_path']"
test_files="['$dapo_math_test_path']"

# Default enable Qwen-Math prompt template (contains \boxed{} constraint for reward parsing)
USE_QWEN_MATH_TEMPLATE=${USE_QWEN_MATH_TEMPLATE:-1}
QWEN_MATH_TEMPLATE_PATH=${QWEN_MATH_TEMPLATE_PATH:-"$(dirname "$0")/../../../../examples/grpo_trainer/qwen_math_chat_template.jinja"}

# Strip the first user-prefix for dapo_math
STRIP_DAPO_MATH_USER_PREFIX=${STRIP_DAPO_MATH_USER_PREFIX:-1}
DAPO_MATH_USER_PREFIX='Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n'

extra_args=()
if [[ "$USE_QWEN_MATH_TEMPLATE" == "1" ]]; then
    if [[ ! -f "$QWEN_MATH_TEMPLATE_PATH" ]]; then
        echo "Qwen math chat template not found: $QWEN_MATH_TEMPLATE_PATH" >&2
        exit 1
    fi
    qwen_math_template_compact=$(tr '\n' ' ' < "$QWEN_MATH_TEMPLATE_PATH")
    extra_args+=("+data.apply_chat_template_kwargs.chat_template='${qwen_math_template_compact}'")
    extra_args+=("+data.apply_chat_template_kwargs.chat_template_format=jinja")
fi
if [[ "$STRIP_DAPO_MATH_USER_PREFIX" == "1" ]]; then
    extra_args+=("data.strip_user_prompt_prefix=true")
    extra_args+=("+data.user_prompt_prefix='${DAPO_MATH_USER_PREFIX}'")
fi

python3 -m recipe.dapo.main_dapo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric=acc \
    algorithm.filter_groups.max_num_gen_batches=10 \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.prompt_key=prompt \
    data.truncation=left \
    data.filter_overlong_prompts=False \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.gen_batch_size=128 \
    data.train_batch_size=128 \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=10240 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=null \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=10240 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=null \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=1 \
    actor_rollout_ref.actor.clip_ratio_high=1 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_mode=clip_cov \
    actor_rollout_ref.actor.k_percent=0.2 \
    actor_rollout_ref.actor.ppo_kl_coef=1 \
    actor_rollout_ref.actor.clip_cov_ratio=0.0002 \
    actor_rollout_ref.actor.clip_cov_lb=1.0 \
    actor_rollout_ref.actor.clip_cov_ub=5.0 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.weight_decay=0 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=null \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.actor.policy_loss.advantage_clip.enable=False \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=False \
    reward_model.overlong_buffer.len=2048 \
    reward_model.overlong_buffer.penalty_factor=1.0 \
    trainer.cliped_token_pair_enable=False \
    trainer.logger='["console","swanlab"]' \
    trainer.rollout_data_dir=/data/huaiwenzhang/projects/verl/training_rollout_metrics/encov_advclip_baseline_follow_cov \
    trainer.project_name='verl_grpo_qwen2_5_1_5b_gsm8k_math' \
    trainer.experiment_name='encov_advclip_baseline_follow_cov' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=25 \
    trainer.save_freq=50 \
    trainer.total_epochs=1000 \
    trainer.default_local_dir="/YOUR_CKPTS_PATH" \
    trainer.resume_mode=disable \
    "${extra_args[@]}" \
    "$@"
