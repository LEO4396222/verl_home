# VERL 检查点测试指南

本文档提供了如何使用VERL框架测试训练好的检查点模型的详细指南。

## 测试方法概述

测试过程分为两个主要步骤：

1. **生成（Generation）**：使用`verl.trainer.main_generation`模块，让模型对测试数据集生成回复
2. **评估（Evaluation）**：使用`verl.trainer.main_eval`模块，评估生成回复的质量和准确性

## 准备工作

### 1. 测试数据集格式

测试数据集需要是**Parquet格式**，并且包含以下字段：

- `prompt`：模型输入的提示（必需）
- `data_source`：数据来源标识（评估时需要）
- `reward_model`：包含ground truth信息的字段（评估时需要）

提示数据应该已经是**chat template格式**，例如：

```python
# 示例提示格式（列表中的字典）
chat_example = [
    {"role": "system", "content": "你是一个数学解题助手"},
    {"role": "user", "content": "计算：123 + 456 = ?"}
]
```

### 2. 检查点模型准备

确保您的检查点路径正确，例如：
`/home/huaiwenzhang/data/huaiwenzhang/projects/verl/checkpoints/verl_grpo_qwen2_5_1_5b_gsm8k_math/grpo_4x4090_adv_clip/global_step_200`

## 使用方法

### 方法一：使用Python脚本（推荐）

Python脚本提供了更多的配置选项和更好的错误处理：

```bash
chmod +x test_checkpoint.py
# 使用默认数据集（gsm8k）
./test_checkpoint.py

# 使用特定数据集
./test_checkpoint.py --dataset gsm8k

# 使用自定义测试数据路径
./test_checkpoint.py --test-data /path/to/your/test_data.parquet
```

**主要参数说明：**

- `--dataset`：快速选择预定义的数据集（如 gsm8k, math, MATH_500, aime_2024等）
- `--test-data`：测试数据集路径（默认：/home/huaiwenzhang/data/gsm8k/test.parquet）
- `--checkpoint-path`：检查点路径（默认为您提供的路径）
- `--output-dir`：结果输出目录
- `--n-gpus-per-node`：使用的GPU数量
- `--batch-size`：批处理大小
- `--temperature`：生成温度（0.0表示确定性生成）

**完整参数列表：**

```bash
./test_checkpoint.py --help
```

### 方法二：使用Bash脚本

```bash
chmod +x test_checkpoint.sh

# 使用默认数据集（gsm8k）
./test_checkpoint.sh

# 使用特定数据集
./test_checkpoint.sh --dataset gsm8k

# 使用自定义测试数据路径
./test_checkpoint.sh --test-data /path/to/your/test_data.parquet
```

### 支持的数据集

脚本现在支持以下预定义数据集：

- `gsm8k`：GSM8K数学问题集
- `math`：MATH数学竞赛数据集
- `MATH_500`：MATH数据集的500题子集
- `aime_2024`：AIME 2024数学竞赛题
- `aime_2025`：AIME 2025数学竞赛题
- `amc23`：AMC 2023数学竞赛题
- `dapo_math`：DAPO数学数据集
- `olympiaBench`：Olympia数学基准测试集

脚本会自动验证数据集文件是否存在，并在不存在时提供错误提示。

## 测试流程详解

### 1. 生成阶段

`main_generation.py`模块会：
- 加载训练好的检查点模型
- 读取测试数据集
- 对每个提示生成指定数量的回复
- 将生成结果保存为新的Parquet文件

### 2. 评估阶段

`main_eval.py`模块会：
- 读取生成的结果文件
- 对每个生成的回复计算奖励分数
- 按数据源统计平均分数
- 输出评估结果

## 常见问题解答

### 1. 测试数据格式不正确怎么办？

确保您的测试数据符合要求的格式：
- 使用Parquet格式存储
- 包含必要的字段（prompt, data_source, reward_model）
- prompt字段已经是chat template格式

### 2. 如何处理大模型的内存问题？

- 调整`--batch-size`参数减小批量大小
- 调整`--prompt-length`和`--response-length`限制输入输出长度
- 确保`rollout.gpu_memory_utilization`设置合理（默认0.8）

### 3. 评估过程失败怎么办？

如果评估失败，生成的结果文件仍然会被保存，您可以：
- 检查评估配置是否正确
- 手动分析生成的结果文件
- 调整reward model相关参数

## 输出结果解释

测试完成后，您将获得：

1. 生成的回复文件（Parquet格式）
2. 评估分数统计（按数据源分类）

评估分数越高，表示模型在相应任务上的表现越好。

## 自定义测试配置

如果需要更详细的配置，您可以参考以下文件修改默认配置：

- `verl/trainer/config/generation.yaml`：生成配置
- `verl/trainer/config/evaluation.yaml`：评估配置

## 示例测试命令

```bash
# 使用Python脚本测试GSM8K数据集（简单方式）
./test_checkpoint.py --dataset gsm8k --n-gpus-per-node 4 --batch-size 4

# 使用Python脚本测试MATH数据集
./test_checkpoint.py \
    --dataset math \
    --checkpoint-path /home/huaiwenzhang/data/huaiwenzhang/projects/verl/checkpoints/verl_grpo_qwen2_5_1_5b_gsm8k_math/grpo_4x4090_adv_clip/global_step_200 \
    --n-gpus-per-node 4 \
    --batch-size 4 \
    --temperature 0.0 \
    --output-dir ./math_test_results

# 使用Bash脚本测试特定数据集
./test_checkpoint.sh --dataset aime_2024
```

## 注意事项

1. 确保您的环境已正确安装VERL框架及其依赖
2. 测试数据的质量会直接影响测试结果的准确性
3. 对于大型模型，建议使用足够的GPU资源以获得更好的性能
4. 根据具体任务调整生成参数（如temperature、response_length等）

## 技术支持

如果遇到问题，请参考VERL框架的官方文档或提交issue到项目仓库。