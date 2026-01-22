# DPO/DAPO Training Guide for OpenManus-RL

## Overview

This guide covers **DPO (Direct Preference Optimization)** and **DAPO (Direct Alignment from Preferences Optimization)** training for agent models in OpenManus-RL. These methods provide an alternative to PPO/GRPO by directly optimizing from preference data without requiring a separate reward model.

## Table of Contents

- [What is DPO/DAPO?](#what-is-dpo-dapo)
- [When to Use DPO vs PPO/GRPO](#when-to-use-dpo-vs-ppo-grpo)
- [Data Preparation](#data-preparation)
- [Training with DAPO](#training-with-dapo)
- [Evaluation](#evaluation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Troubleshooting](#troubleshooting)

---

## What is DPO/DAPO?

### DPO (Direct Preference Optimization)

DPO is an offline preference learning algorithm that:
- **Eliminates the need for a separate reward model**
- **Trains directly from preference pairs** (preferred vs dispreferred responses)
- **More stable than RLHF** - no online RL required
- **Simpler to implement** - single-stage training

### DAPO (Direct Alignment from Preferences Optimization)

DAPO is Verl's implementation that extends DPO with:
- **Better handling of long sequences**
- **Integration with vLLM for efficient training**
- **Support for distributed training**
- **Compatible with agent trajectories**

---

## When to Use DPO vs PPO/GRPO

### Use DPO/DAPO when:
- ✅ You have preference data (preferred vs dispreferred trajectories)
- ✅ You want simpler, more stable training
- ✅ Reward modeling is difficult for your task
- ✅ You have offline data and don't need online exploration

### Use PPO/GRPO when:
- ✅ You have access to environment for online rollouts
- ✅ You need active exploration
- ✅ Reward signal is well-defined
- ✅ You want to learn from environment interaction

### Comparison Table

| Aspect | DPO/DAPO | PPO/GRPO |
|--------|----------|----------|
| Data | Offline preferences | Online rollouts |
| Stability | Very stable | Less stable |
| Complexity | Simple | More complex |
| Reward Model | Not needed | PPO needs critic |
| Exploration | No active exploration | Active exploration |
| Training Speed | Faster | Slower |
| Memory | Less memory | More memory (GRPO < PPO) |

---

## Data Preparation

### Preference Data Format

DPO requires pairs of trajectories where one is preferred over the other:

```json
{
  "prompt": "Navigate to the coffee machine",
  "chosen": [
    {"role": "user", "content": "Task: Navigate to coffee machine"},
    {"role": "assistant", "content": "Think: I need to find the kitchen\nAct: go_to(kitchen)"},
    {"role": "user", "content": "Observation: You are in the kitchen"},
    {"role": "assistant", "content": "Think: Now I can see the coffee machine\nAct: answer(success)"}
  ],
  "rejected": [
    {"role": "user", "content": "Task: Navigate to coffee machine"},
    {"role": "assistant", "content": "Think: Let me wander around\nAct: go_to(bedroom)"},
    {"role": "user", "content": "Observation: You are in the bedroom"},
    {"role": "assistant", "content": "Think: Wrong place\nAct: answer(failure)"}
  ]
}
```

### Creating Preference Data

You can create preference data in several ways:

#### 1. From Rollout Results

Collect multiple rollouts for the same task and rank by success:

```python
import pandas as pd

def create_preference_pairs(rollouts_df):
    """Create preference pairs from rollout results."""
    pairs = []
    
    # Group by task
    for task_id, group in rollouts_df.groupby('task_id'):
        trajectories = group.to_dict('records')
        
        # Sort by reward
        trajectories.sort(key=lambda x: x['total_reward'], reverse=True)
        
        # Create pairs: better vs worse
        for i in range(len(trajectories) - 1):
            if trajectories[i]['total_reward'] > trajectories[i+1]['total_reward']:
                pairs.append({
                    'prompt': trajectories[i]['prompt'],
                    'chosen': trajectories[i]['trajectory'],
                    'rejected': trajectories[i+1]['trajectory'],
                })
    
    return pairs
```

#### 2. From Human Annotations

Collect human preferences on agent trajectories:

```python
def annotate_preferences(trajectories):
    """Interactive annotation tool."""
    pairs = []
    
    for task_id, candidates in trajectories.items():
        if len(candidates) < 2:
            continue
        
        print(f"\nTask: {task_id}")
        print(f"\nCandidate A:\n{format_trajectory(candidates[0])}")
        print(f"\nCandidate B:\n{format_trajectory(candidates[1])}")
        
        choice = input("Which is better? (A/B/Equal): ")
        
        if choice == 'A':
            pairs.append({
                'prompt': task_id,
                'chosen': candidates[0],
                'rejected': candidates[1],
            })
        elif choice == 'B':
            pairs.append({
                'prompt': task_id,
                'chosen': candidates[1],
                'rejected': candidates[0],
            })
    
    return pairs
```

#### 3. Using AI Feedback

Use a strong model (like GPT-4) to rank trajectories:

```python
from openai import OpenAI

def rank_with_ai(trajectories, prompt):
    """Rank trajectories using AI feedback."""
    client = OpenAI()
    
    ranking_prompt = f"""
    Task: {prompt}
    
    Evaluate these two agent trajectories and determine which one is better:
    
    Trajectory A:
    {format_trajectory(trajectories[0])}
    
    Trajectory B:
    {format_trajectory(trajectories[1])}
    
    Respond with just 'A' or 'B' for which is better.
    """
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": ranking_prompt}]
    )
    
    return response.choices[0].message.content.strip()
```

### Converting to Parquet Format

```python
import pandas as pd

# Prepare data
preference_data = []
for pair in pairs:
    preference_data.append({
        'prompt': pair['prompt'],
        'chosen': pair['chosen'],
        'rejected': pair['rejected'],
    })

# Convert to DataFrame
df = pd.DataFrame(preference_data)

# Save as parquet
df.to_parquet('data/dpo/train.parquet', index=False)
```

---

## Training with DAPO

### Basic Training Script

Create `scripts/dpo_train/train_alfworld_dpo.sh`:

```bash
#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES="0,1,2,3"

python3 -m verl.recipe.dapo.main_dapo \
    data.train_files=./data/alfworld_dpo/train.parquet \
    data.val_files=./data/alfworld_dpo/val.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    actor.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor.optim.lr=5e-7 \
    actor.ppo_mini_batch_size=32 \
    actor.ppo_micro_batch_size_per_gpu=4 \
    actor.model.enable_gradient_checkpointing=True \
    ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    ref.model.load_type=megatron \
    algorithm.beta=0.1 \
    algorithm.label_smoothing=0.0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='openmanus-rl_dapo_alfworld' \
    trainer.experiment_name='dapo_qwen2.5_1.5b' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=100
```

### Key DAPO Parameters

```bash
# Beta parameter (KL penalty strength)
algorithm.beta=0.1  # Higher = stay closer to reference model

# Label smoothing
algorithm.label_smoothing=0.0  # Add label smoothing if overfitting

# Learning rate (typically lower than SFT)
actor.optim.lr=5e-7

# Reference model
ref.model.path=path/to/reference/model  # Usually the SFT model
```

---

## Evaluation

### Option 1: Using Rollout Scripts

After DPO training, evaluate with rollouts:

```bash
python scripts/rollout/openmanus_rollout.py \
    --env alfworld \
    --model ./checkpoints/dapo/epoch_100 \
    --base_url http://127.0.0.1:8000/v1 \
    --batch_size 4 \
    --total_envs 100 \
    --max_steps 50 \
    --dump_path logs/dapo_eval.jsonl
```

### Option 2: Compare with Baseline

```python
import pandas as pd

def compare_models(baseline_results, dpo_results):
    """Compare baseline vs DPO model."""
    baseline_df = pd.read_json(baseline_results, lines=True)
    dpo_df = pd.read_json(dpo_results, lines=True)
    
    print(f"Baseline Success Rate: {baseline_df['success'].mean():.2%}")
    print(f"DPO Success Rate: {dpo_df['success'].mean():.2%}")
    
    print(f"\nBaseline Avg Reward: {baseline_df['total_reward'].mean():.2f}")
    print(f"DPO Avg Reward: {dpo_df['total_reward'].mean():.2f}")
    
    print(f"\nBaseline Avg Steps: {baseline_df['num_steps'].mean():.1f}")
    print(f"DPO Avg Steps: {dpo_df['num_steps'].mean():.1f}")
```

---

## Hyperparameter Tuning

### Beta (KL Penalty)

Controls how much the model can deviate from the reference:

```bash
# Conservative (stays close to reference)
algorithm.beta=0.5

# Moderate (balanced)
algorithm.beta=0.1

# Aggressive (more freedom)
algorithm.beta=0.01
```

**Guidelines:**
- Start with `beta=0.1`
- If model deviates too much → increase beta
- If model doesn't improve → decrease beta

### Learning Rate

```bash
# Conservative
actor.optim.lr=1e-7

# Moderate
actor.optim.lr=5e-7

# Aggressive
actor.optim.lr=1e-6
```

### Batch Size

```bash
# Small (memory constrained)
data.train_batch_size=32

# Medium
data.train_batch_size=64

# Large (best results)
data.train_batch_size=128
```

---

## Troubleshooting

### Issue: Model Not Improving

**Symptoms**: Validation loss not decreasing, performance not better than baseline

**Solutions**:
1. Check preference data quality:
   ```python
   # Verify chosen are actually better than rejected
   df = pd.read_parquet('train.parquet')
   print(df[['chosen_reward', 'rejected_reward']].head())
   ```

2. Decrease beta:
   ```bash
   algorithm.beta=0.05  # Allow more deviation
   ```

3. Increase learning rate:
   ```bash
   actor.optim.lr=1e-6
   ```

### Issue: Training Unstable

**Symptoms**: Loss spikes, NaN values

**Solutions**:
1. Lower learning rate:
   ```bash
   actor.optim.lr=1e-7
   ```

2. Enable gradient clipping:
   ```bash
   actor.grad_clip=1.0
   ```

3. Increase beta:
   ```bash
   algorithm.beta=0.2  # More conservative
   ```

### Issue: Model Deviates Too Much from Reference

**Symptoms**: Model produces invalid outputs or ignores formatting

**Solutions**:
1. Increase KL penalty:
   ```bash
   algorithm.beta=0.5
   ```

2. Add format rewards during data collection

3. Use reference model from later SFT checkpoint

---

## Best Practices

1. **Start with good SFT model**: DPO works best when starting from a strong supervised fine-tuned model

2. **High-quality preferences**: Ensure your preference pairs are clear and consistent

3. **Balance dataset**: Have roughly equal numbers of successful and failed trajectories

4. **Monitor KL divergence**: Watch KL divergence from reference model during training

5. **Multiple samples**: Generate multiple trajectories per prompt for better preference pairs

6. **Combine with PPO/GRPO**: Use DPO for initial alignment, then PPO for fine-tuning

---

## Example Workflow

### Complete End-to-End Pipeline

```bash
# 1. SFT training
./scripts/run_sft.sh 4 Qwen/Qwen2.5-1.5B-Instruct \
    trainer.total_training_steps=1000

# 2. Collect rollout data
python scripts/rollout/openmanus_rollout.py \
    --env alfworld \
    --model ./checkpoints/sft/global_step_1000 \
    --batch_size 8 \
    --total_envs 500 \
    --dump_path data/rollouts.jsonl

# 3. Create preference pairs
python scripts/create_preference_data.py \
    --input data/rollouts.jsonl \
    --output data/dpo/train.parquet

# 4. DPO training
bash scripts/dpo_train/train_alfworld_dpo.sh

# 5. Evaluate
python scripts/rollout/openmanus_rollout.py \
    --env alfworld \
    --model ./checkpoints/dapo/final \
    --total_envs 100 \
    --dump_path logs/dapo_eval.jsonl
```

---

## References

- [DPO Paper](https://arxiv.org/abs/2305.18290)
- [Verl DAPO Documentation](https://github.com/volcengine/verl/tree/main/recipe/dapo)
- [OpenManus-RL Main Documentation](../README.md)

---

**Note**: DPO/DAPO is an active area of research. Experiment with different configurations to find what works best for your specific agent tasks.
