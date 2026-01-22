# Training Examples Guide

This guide provides comprehensive examples for training OpenManus-RL models across different environments and algorithms.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Training Algorithms](#training-algorithms)
  - [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
  - [PPO Training](#ppo-training)
  - [GRPO Training](#grpo-training)
- [Environment-Specific Examples](#environment-specific-examples)
  - [ALFWorld](#alfworld)
  - [WebShop](#webshop)
  - [GAIA](#gaia)
- [Advanced Topics](#advanced-topics)
- [Common Issues](#common-issues)

---

## Environment Setup

### Prerequisites

```bash
# Create conda environment
conda create -n openmanus-rl python=3.10 -y
conda activate openmanus-rl

# Clone repository with submodules
git clone --recursive https://github.com/OpenManus/OpenManus-RL.git
cd OpenManus-RL

# Install dependencies
pip3 install torch torchvision
pip install -e .[vllm]
pip3 install flash-attn --no-build-isolation
pip install wandb
```

### Environment Variables

Create a `.env` file in the repository root:

```bash
# OpenAI API (if using GPT models)
OPENAI_API_KEY=your_openai_key

# WandB logging
WANDB_API_KEY=your_wandb_key
WANDB_BASE_URL=https://api.wandb.ai

# Google API for GAIA tools (optional)
GOOGLE_API_KEY=your_google_key
GOOGLE_CX=your_custom_search_engine_id

# CUDA devices
CUDA_VISIBLE_DEVICES=0,1,2,3
```

---

## Data Preparation

### Download Datasets

```bash
# Download OpenManus-RL dataset from HuggingFace
# Visit: https://huggingface.co/datasets/CharlieDreemur/OpenManus-RL

# Or use huggingface-cli
huggingface-cli download CharlieDreemur/OpenManus-RL --repo-type dataset --local-dir ./data/openmanus-rl
```

### Data Format

Training data should be in Parquet format with the following structure:

```python
{
    "id": "task_001",
    "conversations": [
        {"role": "user", "content": "Task description"},
        {"role": "assistant", "content": "Think: ...\nAct: ..."},
        {"role": "user", "content": "Observation: ..."},
        {"role": "assistant", "content": "Think: ...\nAct: answer(result)"}
    ],
    "reward_model": "environment_specific_reward"
}
```

---

## Training Algorithms

### Supervised Fine-Tuning (SFT)

SFT is the first step to teach the model the ReAct format and basic agent behaviors.

#### Basic SFT Training

```bash
./scripts/run_sft.sh 4 /path/to/base/model \
    data.truncation=right \
    trainer.total_training_steps=1000 \
    trainer.logger="['console','wandb']" \
    trainer.project_name="openmanus-rl-sft" \
    trainer.experiment_name="qwen-sft-baseline"
```

#### SFT with Custom Data

```bash
./scripts/run_sft.sh 4 Qwen/Qwen2.5-1.5B-Instruct \
    data.train_files=./data/custom/train.parquet \
    data.val_files=./data/custom/val.parquet \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    trainer.total_training_steps=2000 \
    trainer.save_freq=500 \
    trainer.test_freq=100
```

#### Key SFT Parameters

```bash
# Learning rate
trainer.lr=1e-5

# Batch sizes
data.train_batch_size=32
data.micro_batch_size=4

# Gradient accumulation
trainer.gradient_accumulation_steps=8

# Mixed precision
trainer.fp16=True  # or bf16=True

# Checkpointing
trainer.save_freq=500  # Save every 500 steps
trainer.checkpoint_dir=./checkpoints/sft
```

---

### PPO Training

PPO uses a critic network for value estimation and is suitable for complex tasks.

#### ALFWorld PPO

```bash
bash scripts/ppo_train/train_alfworld.sh vllm \
    actor_rollout_ref.model.path=./checkpoints/sft/global_step_1000 \
    data.train_files=./data/alfworld/train.parquet \
    data.val_files=./data/alfworld/val.parquet \
    trainer.total_epochs=150 \
    trainer.experiment_name="alfworld_ppo_v1"
```

#### WebShop PPO

```bash
bash scripts/ppo_train/train_webshop.sh vllm \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    data.train_batch_size=128 \
    trainer.total_epochs=200
```

#### Key PPO Parameters

```bash
# Advantage estimation
algorithm.adv_estimator=gae
algorithm.gamma=0.99
algorithm.lam=0.95

# PPO specific
actor_rollout_ref.actor.clip_eps=0.2
actor_rollout_ref.actor.entropy_coeff=0.01

# Critic network
critic.optim.lr=1e-5
critic.ppo_mini_batch_size=128

# KL divergence
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.01
```

---

### GRPO Training

GRPO (Group Relative Policy Optimization) is memory-efficient and doesn't require a critic network.

#### ALFWorld GRPO

```bash
bash scripts/ppo_train/train_alfworld_grpo.sh vllm \
    actor_rollout_ref.model.path=./checkpoints/sft/global_step_1000 \
    actor_rollout_ref.rollout.n=8 \
    data.train_batch_size=128 \
    trainer.experiment_name="alfworld_grpo_v1"
```

#### WebShop GRPO

```bash
bash scripts/ppo_train/train_webshop_grpo.sh vllm \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.rollout.n=16 \
    trainer.total_epochs=150
```

#### Key GRPO Parameters

```bash
# Advantage estimation
algorithm.adv_estimator=grpo

# Sampling
actor_rollout_ref.rollout.n=8  # Number of samples per prompt
actor_rollout_ref.rollout.val_kwargs.temperature=0.4

# No KL loss (typically)
actor_rollout_ref.actor.use_kl_loss=False
actor_rollout_ref.actor.kl_loss_coef=0.0
```

For more details, see the [GRPO Training Guide](GRPO_TRAINING_GUIDE.md).

---

## Environment-Specific Examples

### ALFWorld

ALFWorld is a text-based household task environment.

#### Environment Setup

```bash
# Install ALFWorld
pip install alfworld

# Download game data
alfworld-download -f
```

#### Quick Start Training

```bash
# SFT first
./scripts/run_sft.sh 4 Qwen/Qwen2.5-1.5B-Instruct \
    data.train_files=./data/alfworld/sft_train.parquet \
    trainer.total_training_steps=1000 \
    trainer.experiment_name="alfworld_sft"

# Then GRPO
bash scripts/ppo_train/train_alfworld_grpo.sh vllm \
    actor_rollout_ref.model.path=./checkpoints/alfworld_sft/global_step_1000 \
    trainer.experiment_name="alfworld_grpo_from_sft"
```

#### Environment-Specific Settings

```bash
env.env_name=alfworld/AlfredTWEnv
env.max_steps=50  # Maximum steps per episode
env.seed=0

# Reward settings
actor_rollout_ref.actor.use_invalid_action_penalty=True
actor_rollout_ref.actor.invalid_action_penalty_coef=0.1
```

#### Evaluation

```bash
# Run rollout evaluation
python scripts/rollout/openmanus_rollout.py \
    --env alfworld \
    --model ./checkpoints/alfworld_grpo/final \
    --base_url http://127.0.0.1:8000/v1 \
    --batch_size 4 \
    --total_envs 100 \
    --max_steps 50 \
    --dump_path logs/alfworld/eval_results.jsonl
```

---

### WebShop

WebShop is an e-commerce environment for product search and selection.

#### Environment Setup

```bash
# Setup WebShop
cd openmanus_rl/environments/env_package/webshop/webshop/
conda create -n agentenv_webshop python==3.10 -y
conda activate agentenv_webshop
bash ./setup.sh -d all
```

#### Quick Start Training

```bash
# Activate main environment
conda activate openmanus-rl

# GRPO training (recommended)
bash scripts/ppo_train/train_webshop_grpo.sh vllm \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    data.train_batch_size=128 \
    trainer.total_epochs=150
```

#### Environment-Specific Settings

```bash
env.env_name=Webshop
env.max_steps=15  # Shorter episodes than ALFWorld
env.seed=0

# Memory settings (WebShop needs more memory)
actor_rollout_ref.rollout.gpu_memory_utilization=0.6
data.max_prompt_length=4096
```

---

### GAIA

GAIA is a general AI assistant benchmark with diverse real-world tasks.

#### Environment Setup

```bash
# Download GAIA validation set
mkdir -p data/gaia
# Download from https://huggingface.co/datasets/gaia-benchmark/GAIA

# Set up Google API (for search tools)
export GOOGLE_API_KEY=your_key
export GOOGLE_CX=your_cx
```

#### Rollout Evaluation

```bash
# With OpenAI models
python scripts/rollout/openmanus_rollout.py \
    --env gaia \
    --model gpt-4o \
    --gaia_tools python_code_generator \
    --batch_size 2 \
    --total_envs 10 \
    --max_steps 30 \
    --dump_path logs/gaia/gpt4o_results.jsonl

# With local vLLM models
python scripts/rollout/openmanus_rollout.py \
    --env gaia \
    --model qwen2.5-7b-instruct \
    --base_url http://127.0.0.1:8000/v1 \
    --gaia_tools python_code_generator \
    --batch_size 2 \
    --total_envs 10 \
    --max_steps 30 \
    --dump_path logs/gaia/local_results.jsonl
```

#### Calculate Scores

```bash
python scripts/gaia_calculate_score.py \
    --input_file logs/gaia/gpt4o_results.jsonl \
    --output_file logs/gaia/gpt4o_scores.json
```

---

## Advanced Topics

### Multi-Node Training

```bash
# Node 1 (master)
bash scripts/ppo_train/train_alfworld_grpo.sh vllm \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=2 \
    trainer.node_rank=0 \
    trainer.master_addr=node1_ip \
    trainer.master_port=29500

# Node 2 (worker)
bash scripts/ppo_train/train_alfworld_grpo.sh vllm \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=2 \
    trainer.node_rank=1 \
    trainer.master_addr=node1_ip \
    trainer.master_port=29500
```

### Mixed Precision Training

```bash
# Use BF16 (recommended for newer GPUs)
actor_rollout_ref.model.dtype=bfloat16
critic.model.dtype=bfloat16

# Or FP16
actor_rollout_ref.model.dtype=float16
critic.model.dtype=float16
```

### Custom Reward Functions

Create a custom reward function:

```python
# custom_reward.py
def compute_custom_reward(trajectory, **kwargs):
    """
    Compute custom reward based on trajectory.
    
    Args:
        trajectory: List of conversation turns
        **kwargs: Additional arguments
    
    Returns:
        float: Reward value
    """
    # Your custom logic here
    reward = 0.0
    
    # Example: Reward for correct format
    if "<think>" in trajectory[-1]["content"]:
        reward += 0.1
    
    # Example: Reward for success
    if "answer(" in trajectory[-1]["content"]:
        reward += 1.0
    
    return reward
```

Use it in training:

```bash
bash scripts/ppo_train/train_alfworld_grpo.sh vllm \
    custom_reward_function.path=./custom_reward.py \
    custom_reward_function.name=compute_custom_reward
```

### Resume from Checkpoint

```bash
bash scripts/ppo_train/train_alfworld_grpo.sh vllm \
    actor_rollout_ref.model.path=./checkpoints/grpo/epoch_50 \
    trainer.resume_from_checkpoint=./checkpoints/grpo/epoch_50/trainer_state.pt
```

---

## Common Issues

### Out of Memory

**Symptoms**: CUDA OOM errors during training

**Solutions**:
```bash
# Reduce batch size
data.train_batch_size=64

# Enable parameter offloading
actor_rollout_ref.actor.fsdp_config.param_offload=True

# Reduce GPU memory utilization for vLLM
actor_rollout_ref.rollout.gpu_memory_utilization=0.4

# Use gradient checkpointing
actor_rollout_ref.model.enable_gradient_checkpointing=True
```

### Training Divergence

**Symptoms**: Reward/loss values explode or become NaN

**Solutions**:
```bash
# Lower learning rate
actor_rollout_ref.actor.optim.lr=5e-7

# Use gradient clipping
actor_rollout_ref.actor.grad_clip=1.0

# Increase KL penalty
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.05
```

### Slow Training

**Symptoms**: Training takes too long per epoch

**Solutions**:
```bash
# Use vLLM engine (fastest)
ENGINE=vllm

# Increase tensor parallelism
actor_rollout_ref.rollout.tensor_model_parallel_size=2

# Enable chunked prefill
actor_rollout_ref.rollout.enable_chunked_prefill=True

# Reduce validation frequency
trainer.test_freq=10
```

### Environment Connection Issues

**Symptoms**: Cannot connect to environment server

**Solutions**:
```bash
# Check environment server is running
# For WebShop:
cd openmanus_rl/environments/env_package/webshop/webshop/
conda activate agentenv_webshop
webshop --host 0.0.0.0 --port 36001

# Check port is not in use
lsof -i :36001

# Update port in training script if needed
env.port=36001
```

---

## Best Practices

1. **Start with SFT**: Always begin with supervised fine-tuning to teach basic agent behaviors
2. **Use GRPO for faster iteration**: GRPO is more memory-efficient and faster than PPO
3. **Monitor WandB logs**: Keep track of reward trends, policy loss, and environment-specific metrics
4. **Validate frequently**: Use `trainer.test_freq` to catch issues early
5. **Save checkpoints**: Use `trainer.save_freq` to avoid losing progress
6. **Tune hyperparameters gradually**: Change one parameter at a time to understand its effect

---

## Additional Resources

- [GRPO Training Guide](GRPO_TRAINING_GUIDE.md)
- [Development Guide](DEVELOPMENT_GUIDE_EN.md)
- [Rollout Guide](ROLLOUT_GUIDE.md)
- [Evaluation Guide](EVALUATION_GUIDE_EN.md)
- [Training Process Overview](README.md)

---

## Getting Help

If you encounter issues:
1. Check this guide and other documentation
2. Review WandB logs for training metrics
3. Consult the troubleshooting section
4. Search existing GitHub issues
5. Open a new issue with:
   - Error messages
   - Training configuration
   - Environment details
   - Steps to reproduce

---

**Last Updated**: January 2026
