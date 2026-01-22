# GRPO Training Guide

## Overview

GRPO (Group Relative Policy Optimization) is an efficient RL algorithm that normalizes rewards within groups of trajectories, making it particularly effective for agent training. Unlike PPO which uses a critic network, GRPO directly computes advantages by comparing rewards across multiple samples from the same prompt.

## Key Differences: PPO vs GRPO

### PPO (Proximal Policy Optimization)
- Uses a critic network to estimate value functions
- Requires GAE (Generalized Advantage Estimation)
- More computationally expensive (needs critic network training)
- Better for complex value estimation tasks

### GRPO (Group Relative Policy Optimization)
- **No critic network required** - more memory efficient
- Normalizes rewards across multiple samples (group normalization)
- Simpler to implement and tune
- **Faster training** due to reduced overhead
- Better for tasks with sparse rewards

## Configuration

The key difference in configuration is setting:
```bash
algorithm.adv_estimator=grpo  # Instead of 'gae' for PPO
```

Additional GRPO-specific settings:
```bash
actor_rollout_ref.rollout.n=8                # Number of samples per prompt
actor_rollout_ref.actor.use_kl_loss=False    # Typically disabled for GRPO
actor_rollout_ref.actor.kl_loss_coef=0.0     # Set to 0
```

## Training Scripts

We provide GRPO training scripts for both ALFWorld and WebShop environments:

### ALFWorld GRPO Training

```bash
# Basic usage
bash scripts/ppo_train/train_alfworld_grpo.sh

# With custom engine
bash scripts/ppo_train/train_alfworld_grpo.sh vllm

# With additional parameters
bash scripts/ppo_train/train_alfworld_grpo.sh vllm \
    trainer.total_epochs=200 \
    actor_rollout_ref.actor.optim.lr=5e-7
```

**Configuration highlights:**
- Model: Qwen3-4B (default, customize with `actor_rollout_ref.model.path`)
- Training data size: 128 samples per batch
- Max steps per episode: 50
- Samples per prompt: 8
- GPU: 4 GPUs (configurable)

### WebShop GRPO Training

```bash
# Basic usage
bash scripts/ppo_train/train_webshop_grpo.sh

# With custom configuration
bash scripts/ppo_train/train_webshop_grpo.sh vllm \
    data.train_batch_size=256 \
    actor_rollout_ref.rollout.n=16
```

**Configuration highlights:**
- Model: Qwen2.5-1.5B-Instruct (default)
- Max steps per episode: 15
- Samples per prompt: 8
- GPU: 2 GPUs with tensor parallelism

## Data Preparation

### Required Data Format

Training data should be in Parquet format with the following structure:
```python
{
    'prompt': str,           # Initial task description
    'messages': List[dict],  # Conversation history (optional)
    'reward_model': str,     # Reward model identifier
}
```

### Setting Data Paths

Update the following parameters in the training scripts:
```bash
data.train_files=/path/to/train.parquet
data.val_files=/path/to/val.parquet
```

## Hyperparameter Tuning

### Key Hyperparameters

1. **Number of samples (n)**
   ```bash
   actor_rollout_ref.rollout.n=8  # More samples = better normalization
   ```
   - Recommended: 4-16
   - Higher values improve stability but increase computation

2. **Learning rate**
   ```bash
   actor_rollout_ref.actor.optim.lr=1e-6
   ```
   - Start with 1e-6, adjust based on training stability
   - GRPO typically works well with lower learning rates

3. **Batch size**
   ```bash
   data.train_batch_size=128
   ```
   - Balance between GPU memory and training stability
   - Larger batches improve normalization quality

4. **Temperature**
   ```bash
   actor_rollout_ref.rollout.val_kwargs.temperature=0.4
   ```
   - Controls sampling diversity
   - Lower values (0.2-0.5) for exploitation
   - Higher values (0.6-1.0) for exploration

## Monitoring Training

### WandB Integration

The scripts are configured with WandB logging:
```bash
trainer.logger=['console','wandb']
trainer.project_name='openmanus-rl_grpo_alfworld'
trainer.experiment_name='grpo_qwen3_4b'
```

Set your WandB API key:
```bash
export WANDB_API_KEY=your_key_here
```

### Key Metrics to Monitor

1. **Episode Reward**: Average reward per episode
2. **Policy Loss**: Should decrease over time
3. **Reward Standard Deviation**: Indicates exploration diversity
4. **Success Rate**: Task completion rate (environment-specific)

## Troubleshooting

### Out of Memory (OOM)

Solutions:
```bash
# Reduce batch size
data.train_batch_size=64

# Enable parameter offloading
actor_rollout_ref.actor.fsdp_config.param_offload=True

# Reduce samples per prompt
actor_rollout_ref.rollout.n=4

# Use gradient checkpointing (already enabled by default)
actor_rollout_ref.model.enable_gradient_checkpointing=True
```

### Training Instability

Solutions:
```bash
# Lower learning rate
actor_rollout_ref.actor.optim.lr=5e-7

# Increase samples for better normalization
actor_rollout_ref.rollout.n=16

# Adjust batch size
data.train_batch_size=256
```

### Slow Training

Solutions:
```bash
# Use vLLM engine (default)
bash scripts/ppo_train/train_alfworld_grpo.sh vllm

# Increase GPU memory utilization
actor_rollout_ref.rollout.gpu_memory_utilization=0.7

# Adjust tensor parallelism
actor_rollout_ref.rollout.tensor_model_parallel_size=2
```

## Advanced Usage

### Custom Reward Functions

For custom reward computation, refer to the verl documentation and the char_count example:
```bash
custom_reward_function.path=path/to/reward_function.py
custom_reward_function.name=your_reward_function
```

### Multi-Node Training

```bash
trainer.n_gpus_per_node=4
trainer.nnodes=2  # Number of nodes
```

### Checkpointing

```bash
# Save checkpoints every N epochs
trainer.save_freq=10

# Test every N epochs
trainer.test_freq=5
```

## Comparison with PPO

When to use GRPO:
- ✅ Limited GPU memory (no critic network)
- ✅ Sparse reward tasks
- ✅ Need faster training iterations
- ✅ Good exploration is already achieved through sampling

When to use PPO:
- ✅ Complex value estimation needed
- ✅ Dense reward signals
- ✅ Need precise value function
- ✅ Critic warmup can stabilize training

## Example Workflows

### Quick Experimentation

```bash
# Fast iteration with small model and batch size
bash scripts/ppo_train/train_alfworld_grpo.sh vllm \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    data.train_batch_size=32 \
    trainer.total_epochs=50 \
    trainer.test_freq=5
```

### Production Training

```bash
# Full-scale training with larger model
bash scripts/ppo_train/train_alfworld_grpo.sh vllm \
    actor_rollout_ref.model.path=/path/to/Qwen/Qwen3-14B \
    data.train_batch_size=256 \
    actor_rollout_ref.rollout.n=16 \
    trainer.total_epochs=300 \
    trainer.save_freq=25
```

## References

- [Verl Framework Documentation](https://github.com/volcengine/verl)
- [GRPO Paper (if available)]
- [OpenManus-RL Documentation](../README.md)

## Getting Help

For issues or questions:
1. Check the troubleshooting section above
2. Review WandB logs for training metrics
3. Consult the [Development Guide](DEVELOPMENT_GUIDE_EN.md)
4. Open an issue on GitHub

---

**Note**: Update the data file paths in the training scripts before running:
- `data.train_files=/path/to/your/train.parquet`
- `data.val_files=/path/to/your/val.parquet`
