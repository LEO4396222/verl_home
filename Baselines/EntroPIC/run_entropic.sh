# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_REPO_DIR="${HF_CACHE_DIR}/hub/models--Qwen--Qwen2.5-Math-1.5B"
MODEL_SNAPSHOT_FILE="${MODEL_REPO_DIR}/refs/main"
if [[ ! -f "$MODEL_SNAPSHOT_FILE" ]]; then
    echo "Model cache not found: $MODEL_REPO_DIR" >&2
    exit 1
fi
MODEL_PATH="${MODEL_REPO_DIR}/snapshots/$(cat "$MODEL_SNAPSHOT_FILE")"

DATA_PATH=/data/huaiwenzhang/Datasets
dapo_math_train_path=$DATA_PATH/dapo_math/train.parquet
math500_test_path=$DATA_PATH/_normalized_for_concat_v2/math500.parquet
amc23_test_path=$DATA_PATH/_normalized_for_concat_v2/amc23.parquet
aime_2024_test_path=$DATA_PATH/_normalized_for_concat_v2/aime24.parquet
aime_2025_test_path=$DATA_PATH/_normalized_for_concat_v2/aime25.parquet
minerva_test_path=$DATA_PATH/_normalized_for_concat_v2/minerva_math.parquet
olympiabench_test_path=$DATA_PATH/_normalized_for_concat_v2/olympiadbench.parquet

train_files="['$dapo_math_train_path']"
test_files="['$math500_test_path', '$amc23_test_path', '$aime_2024_test_path', '$aime_2025_test_path', '$minerva_test_path', '$olympiabench_test_path']"

USE_QWEN_MATH_TEMPLATE=${USE_QWEN_MATH_TEMPLATE:-1}
QWEN_MATH_TEMPLATE_PATH=${QWEN_MATH_TEMPLATE_PATH:-"$(dirname "$0")/../../examples/grpo_trainer/qwen_math_chat_template.jinja"}

STRIP_DAPO_MATH_USER_PREFIX=${STRIP_DAPO_MATH_USER_PREFIX:-1}
DAPO_MATH_USER_PREFIX='Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n'

STRIP_DAPO_MATH_USER_SUFFIX=${STRIP_DAPO_MATH_USER_SUFFIX:-1}
DAPO_MATH_USER_SUFFIX='Remember to put your answer on its own line after "Answer:".'

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
if [[ "$STRIP_DAPO_MATH_USER_SUFFIX" == "1" ]]; then
    extra_args+=("data.strip_user_prompt_suffix=true")
    extra_args+=("+data.user_prompt_suffix='${DAPO_MATH_USER_SUFFIX}'")
fi

python3 -m main_entropic \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.policy_loss.loss_mode='entropic' \
    actor_rollout_ref.actor.target_entropy=0.1 \
    actor_rollout_ref.actor.entropy_control_p=1 \
    actor_rollout_ref.actor.entropy_control_i=0.01 \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=False \
    data.truncation=left \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name='verl_grpo_qwen2_5_1_5b_gsm8k_math' \
    trainer.experiment_name='EntroPIC_Qwen2.5-MATH-1.5B' \
    trainer.rollout_data_dir=/data/huaiwenzhang/projects/verl/training_rollout_metrics/EntroPIC_Qwen2.5-MATH-1.5B \
    trainer.rollout_rank_log_freq=0 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=10 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    trainer.validation_n_list=[1,32] \
    enable_pg_entropy_diff_metrics=true \
    trainer.total_epochs=15 \
    data.shuffle=True \
    data.seed=42 \
    "${extra_args[@]}" \
    "$@"
