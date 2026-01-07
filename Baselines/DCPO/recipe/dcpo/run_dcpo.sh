set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3

export N_GPUS_PER_NODE=4

pip install func_timeout
pip install latex2sympy2_extended
pip install math-verify
pip install tensorboard

export WORLD_SIZE=1
export RANK=0
export CHECKPOINT_SAVE=${CHECKPOINT_SAVE:-"/tmp/"}
export CHECKPOINT_LOAD=${CHECKPOINT_LOAD:-/checkpoint_load}
ray stop --force
ps -ef | grep "python" | awk '{print $2}' | xargs kill -9
# nccl=eth0
for ((x = 0; x < $WORLD_SIZE; x++)); do
    rm -rf ${CHECKPOINT_SAVE}/_worker_${x}_ready
done

#wandb
wandb_key=""
export WANDB_API_KEY=$wandb_key

# tensorboard
export TENSORBOARD_DIR=${CHECKPOINT_SAVE}/runs

# pre-train model path
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-Math-1.5B"}



##qwen
data_root=/data/huaiwenzhang/Datasets/dapo_math
train_path=$data_root/train.parquet
test_path=$data_root/test.parquet

train_files="['$train_path']"
val_files="['$test_path']"


# Ray
export NNODES=${WORLD_SIZE:-1}

export TRAIN_PROMPT_BSZ=512
export GEN_PROMPT_BSZ=512
# 训练前先验证
export VAL_BEFORE_TRAIN=True


export RESUME_MODE="auto"

# wandb
export PROJECT_NAME=qwen3-8b
export EXP_NAME=dcpo_8b_17k3v5

# token 1024 + 3072
export MAX_PROMPT_LENGTH=1024
export MAX_RESPONSE_LENGTH=$((4096 - $MAX_PROMPT_LENGTH))
export OVERLONG_BUFFER_LEN=512
export ACTOR_PPO_MAX_TOKEN_LEN=4096
export INFER_PPO_MAX_TOKEN_LEN=4096
export SP_SIZE=1
if [ ! $ENABLE_FILTER_GROUPS ]; then
    export GEN_PROMPT_BSZ=$TRAIN_PROMPT_BSZ
fi

# batch_size 
export TRAIN_PROMPT_MINI_BSZ=$(($WORLD_SIZE * 8))


# #### dcpo 

#是否过滤std=0的数据
export ENABLE_FILTER_GROUPS=False
# advantage
export ADV_ESTIMATOR="dcpo"
export NORM_ADV_BY_STD_IN_GRPO=True

# 使用kl loss
export USE_KL_LOSS_ENABLE=False

# dual clip
export CLIP_RATIO_C=10.0
export CLIP_RATIO_HIGH=0.28
export CLIP_TYPE="dynamic" # dynamic dual

# loss type
export LOSS_AGG_MODE="only-token-mean"
# function 
export LOSS_MODE="dcpo"
# 长度惩罚的版本及系数
# 是否使用长度惩罚
export ENABLE_OVERLONG_BUFFER=False
export OVERLONG_BUFFER_VERSION="dcpo"
export OVERLONG_BUFFER_ERROR_PENALTY=1.0
# ######## dapo end


#基本
export CKPTS_DIR=$CHECKPOINT_SAVE
export TRAIN_FILE=$train_files
export TEST_FILE=$val_files
export TOTAL_EPOCHS=100
export SAVE_FREQ=50
export TEST_FREQ=5
export MAX_NUM_GEN_BATCHES=$(($TRAIN_PROMPT_BSZ / $GEN_PROMPT_BSZ * 100))

pwd
while true; do

    while true; do
        touch ${CHECKPOINT_SAVE}/_worker_${RANK}_ready
        count=0
        # 遍历所有worker编号
        for ((x = 0; x < $WORLD_SIZE; x++)); do
            # 检测对应worker的就绪文件
            if [[ -f "${CHECKPOINT_SAVE}/_worker_${x}_ready" ]]; then
                echo $count
                count=$((count + 1)) # 存在则计数器+1
            fi
        done
        echo "Progress: ${count}/${WORLD_SIZE} workers ready"

        # 判断是否全部就绪
        if [[ $count -eq ${WORLD_SIZE} ]]; then
            echo "全部就绪"
            break # 全部就绪继续执行
        else
            sleep 2s # 等待5秒后再次检查
        fi
    done
    bash ./recipe/dcpo/run_base.sh
done
