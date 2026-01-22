# OpenManus-RL Troubleshooting Guide

This guide helps you diagnose and fix common issues when working with OpenManus-RL.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Training Issues](#training-issues)
- [Environment Issues](#environment-issues)
- [Rollout and Evaluation Issues](#rollout-and-evaluation-issues)
- [Performance Issues](#performance-issues)
- [Configuration Issues](#configuration-issues)
- [Debugging Tips](#debugging-tips)

---

## Installation Issues

### Issue: Submodule not initialized

**Symptoms**:
```
ImportError: No module named 'verl'
ModuleNotFoundError: No module named 'verl'
```

**Solution**:
```bash
# Initialize and update submodules
git submodule update --init --recursive

# If that doesn't work, try:
git submodule sync
git submodule update --init --recursive
```

---

### Issue: Flash Attention installation fails

**Symptoms**:
```
ERROR: Failed building wheel for flash-attn
ninja: build stopped: subcommand failed
```

**Solutions**:

1. **Check CUDA version compatibility**:
```bash
# Check CUDA version
nvcc --version

# Flash attention requires CUDA 11.6+
```

2. **Use pre-built wheels** (if available for your CUDA version):
```bash
pip install flash-attn --no-build-isolation
```

3. **Skip flash attention** (slower but works):
```bash
# Install without flash-attn
pip install -e .

# Training will use regular attention (slower)
```

4. **Install from source with more memory**:
```bash
export MAX_JOBS=4  # Reduce parallelism
pip install flash-attn --no-build-isolation
```

---

### Issue: vLLM installation fails

**Symptoms**:
```
ERROR: Could not install packages due to an OSError: vllm
```

**Solutions**:

1. **Check Python version**:
```bash
python --version  # Should be 3.8-3.11
```

2. **Install specific version**:
```bash
pip install vllm==0.6.3
```

3. **Install from source**:
```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

---

## Training Issues

### Issue: CUDA Out of Memory (OOM)

**Symptoms**:
```
RuntimeError: CUDA out of memory
torch.cuda.OutOfMemoryError
```

**Solutions**:

1. **Reduce batch size**:
```bash
data.train_batch_size=32  # Down from 128
```

2. **Enable model offloading**:
```bash
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
```

3. **Reduce vLLM memory usage**:
```bash
actor_rollout_ref.rollout.gpu_memory_utilization=0.3  # Down from 0.6
```

4. **Use gradient checkpointing**:
```bash
actor_rollout_ref.model.enable_gradient_checkpointing=True
```

5. **Reduce sampling for GRPO**:
```bash
actor_rollout_ref.rollout.n=4  # Down from 8
```

6. **Use smaller model**:
```bash
actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct
```

7. **Reduce sequence lengths**:
```bash
data.max_prompt_length=1024  # Down from 2048
data.max_response_length=256  # Down from 512
```

---

### Issue: Training divergence (NaN loss)

**Symptoms**:
```
Loss becomes NaN
Reward values explode to infinity
Policy gradient becomes inf
```

**Solutions**:

1. **Lower learning rate**:
```bash
actor_rollout_ref.actor.optim.lr=5e-7  # Down from 1e-6
```

2. **Enable gradient clipping**:
```bash
actor_rollout_ref.actor.grad_clip=1.0
```

3. **Increase KL penalty**:
```bash
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.05  # Up from 0.01
```

4. **Check data quality**:
```bash
# Inspect your training data for issues
python scripts/inspect_data.py --data_file train.parquet
```

5. **Use more stable optimizer**:
```bash
actor_rollout_ref.actor.optim.name=adamw
actor_rollout_ref.actor.optim.weight_decay=0.01
```

6. **Warm up critic (PPO only)**:
```bash
trainer.critic_warmup=5  # Warm up for 5 epochs
```

---

### Issue: Slow training speed

**Symptoms**:
- Training takes hours per epoch
- GPU utilization is low

**Solutions**:

1. **Use vLLM engine**:
```bash
actor_rollout_ref.rollout.name=vllm  # Not 'megatron'
```

2. **Enable tensor parallelism**:
```bash
actor_rollout_ref.rollout.tensor_model_parallel_size=2
```

3. **Increase batch size** (if memory allows):
```bash
data.train_batch_size=256
```

4. **Reduce validation frequency**:
```bash
trainer.test_freq=10  # Test every 10 epochs instead of 5
```

5. **Disable unnecessary logging**:
```bash
trainer.logger=['console']  # Remove 'wandb' temporarily
```

6. **Use mixed precision**:
```bash
actor_rollout_ref.model.dtype=bfloat16
```

---

### Issue: Training doesn't improve

**Symptoms**:
- Reward stays constant
- Success rate doesn't increase
- Policy doesn't learn

**Solutions**:

1. **Check reward computation**:
```python
# Add debugging to see if rewards are being computed
print(f"Reward: {reward}, Success: {info.get('success', False)}")
```

2. **Increase sampling diversity (GRPO)**:
```bash
actor_rollout_ref.rollout.val_kwargs.temperature=0.6  # Up from 0.4
actor_rollout_ref.rollout.n=16  # More samples
```

3. **Check if model is updating**:
```bash
# Monitor policy loss in WandB
# Should decrease over time
```

4. **Verify data quality**:
```bash
# Check if training data has successful trajectories
python -c "import pandas as pd; df = pd.read_parquet('train.parquet'); print(df.head())"
```

5. **Start with SFT**:
```bash
# Train with SFT first to learn basic behaviors
./scripts/run_sft.sh 4 model_path ...
```

6. **Adjust reward scale**:
```bash
# If rewards are too small, scale them up
algorithm.reward_scale=10.0
```

---

## Environment Issues

### Issue: Cannot connect to WebShop server

**Symptoms**:
```
ConnectionError: Cannot connect to WebShop at localhost:36001
requests.exceptions.ConnectionError
```

**Solutions**:

1. **Start WebShop server**:
```bash
cd openmanus_rl/environments/env_package/webshop/webshop/
conda activate agentenv_webshop
webshop --host 0.0.0.0 --port 36001
```

2. **Check if port is in use**:
```bash
lsof -i :36001
# If in use, kill the process or use different port
```

3. **Check firewall settings**:
```bash
# Allow port 36001
sudo ufw allow 36001
```

4. **Use different port**:
```bash
# In server:
webshop --host 0.0.0.0 --port 36002

# In training script:
env.port=36002
```

---

### Issue: ALFWorld data not found

**Symptoms**:
```
FileNotFoundError: ALFWorld data not found
RuntimeError: Please run 'alfworld-download'
```

**Solution**:
```bash
# Download ALFWorld data
alfworld-download -f

# Check data location
ls ~/.cache/alfworld/

# If needed, specify custom path
export ALFWORLD_DATA=/path/to/alfworld/data
```

---

### Issue: GAIA tools not working

**Symptoms**:
```
Error: Google API key not found
requests.exceptions.HTTPError: 403 Forbidden
```

**Solutions**:

1. **Set API keys**:
```bash
export GOOGLE_API_KEY=your_key
export GOOGLE_CX=your_custom_search_engine_id
```

2. **Use alternative tools**:
```bash
# Use only Python code generator (no search)
--gaia_tools python_code_generator
```

3. **Check API quota**:
```bash
# Visit Google Cloud Console to check quota
# Make sure billing is enabled
```

---

## Rollout and Evaluation Issues

### Issue: vLLM server not responding

**Symptoms**:
```
ConnectionError: Cannot connect to vLLM server
requests.exceptions.ReadTimeout
```

**Solutions**:

1. **Start vLLM server**:
```bash
bash scripts/serve_model.sh model_path
```

2. **Check server logs**:
```bash
# Look for errors in server output
tail -f /tmp/vllm_server.log
```

3. **Increase timeout**:
```python
# In rollout script
client = OpenAI(base_url="...", timeout=300)  # 5 minutes
```

4. **Check GPU memory**:
```bash
nvidia-smi
# Make sure GPU has enough free memory
```

---

### Issue: Rollout hangs or freezes

**Symptoms**:
- Script runs but produces no output
- Processes seem stuck

**Solutions**:

1. **Check concurrency**:
```bash
# Reduce concurrent workers
--concurrency 2  # Down from 10
```

2. **Add timeout**:
```bash
--max_steps 30  # Limit maximum steps
```

3. **Enable verbose logging**:
```python
# Add to rollout script
import logging
logging.basicConfig(level=logging.DEBUG)
```

4. **Check for deadlocks**:
```bash
# Kill and restart
pkill -f openmanus_rollout
python scripts/rollout/openmanus_rollout.py ...
```

---

## Performance Issues

### Issue: Low GPU utilization

**Symptoms**:
```bash
nvidia-smi  # Shows <50% GPU usage
```

**Solutions**:

1. **Increase batch size**:
```bash
data.train_batch_size=256
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8
```

2. **Enable tensor parallelism**:
```bash
actor_rollout_ref.rollout.tensor_model_parallel_size=2
```

3. **Use larger model**:
```bash
actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct
```

4. **Check for CPU bottlenecks**:
```bash
# Monitor CPU usage
htop
# If CPU is maxed out, increase data workers
data.num_workers=8
```

---

### Issue: Slow data loading

**Symptoms**:
- Training pauses between batches
- High CPU usage but low GPU usage

**Solutions**:

1. **Increase data workers**:
```bash
data.num_workers=8
```

2. **Use faster storage**:
```bash
# Move data to SSD if on HDD
cp /hdd/data/*.parquet /ssd/data/
```

3. **Preload data**:
```bash
data.prefetch_factor=4
```

4. **Cache data in memory**:
```bash
# For small datasets
data.cache_in_memory=True
```

---

## Configuration Issues

### Issue: Hydra configuration errors

**Symptoms**:
```
omegaconf.errors.ConfigAttributeError
hydra.errors.ConfigCompositionException
```

**Solutions**:

1. **Check parameter names**:
```bash
# Use correct parameter paths
actor_rollout_ref.model.path=...  # Not actor.model.path
```

2. **Quote complex values**:
```bash
# Use quotes for lists
trainer.logger="['console','wandb']"
```

3. **Escape special characters**:
```bash
# Use backslash for special chars
data.train_files=/path/to/file\ with\ spaces.parquet
```

4. **Check configuration file**:
```bash
# View the configuration
python -m verl.trainer.main_ppo --cfg job --resolve
```

---

### Issue: WandB logging not working

**Symptoms**:
```
wandb: ERROR Unable to log
wandb: WARNING Not authenticated
```

**Solutions**:

1. **Login to WandB**:
```bash
wandb login your_api_key
```

2. **Set environment variable**:
```bash
export WANDB_API_KEY=your_key
```

3. **Disable WandB temporarily**:
```bash
trainer.logger=['console']
```

4. **Check internet connection**:
```bash
ping api.wandb.ai
```

---

## Debugging Tips

### Enable Debug Logging

```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Check GPU Memory

```bash
# Watch GPU memory in real-time
watch -n 1 nvidia-smi

# Or use gpustat
pip install gpustat
gpustat -i 1
```

### Profile Training

```python
# Add profiling to training script
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    # Your training code
    pass

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Inspect Data

```python
# Check data quality
import pandas as pd

df = pd.read_parquet('train.parquet')
print(df.info())
print(df.head())
print(df['conversations'].apply(len).describe())
```

### Test Environment Locally

```bash
# Test environment connection
python -c "
from openmanus_rl.environments import create_environment
env = create_environment('alfworld/AlfredTWEnv')
obs = env.reset()
print(f'Environment working: {obs}')
"
```

### Verify Model Loading

```python
# Test model loading
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "Qwen/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
print(f"Model loaded successfully: {model.config}")
```

---

## Getting More Help

If your issue isn't covered here:

1. **Check documentation**:
   - [Development Guide](DEVELOPMENT_GUIDE_EN.md)
   - [Training Examples](TRAINING_EXAMPLES_GUIDE.md)
   - [GRPO Guide](GRPO_TRAINING_GUIDE.md)

2. **Search existing issues**:
   - GitHub Issues: https://github.com/OpenManus/OpenManus-RL/issues

3. **Create a new issue** with:
   - Error message (full traceback)
   - Configuration used
   - Environment details (OS, CUDA version, GPU model)
   - Steps to reproduce
   - What you've tried already

4. **Join the community**:
   - Feishu group (see README.md)
   - Discord/Slack (if available)

---

**Last Updated**: January 2025
