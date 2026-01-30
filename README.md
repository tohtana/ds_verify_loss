# Verification of Training with DeepSpeed

The scripts in this repository run training using DeepSpeed with different settings. They also plots loss curves and iteration times for comparison.

---

## Performance Test: Leaf Module Synchronization Fix (PR #7825)

This section documents the performance impact of the leaf module race condition fix in [DeepSpeed PR #7825](https://github.com/deepspeedai/DeepSpeed/pull/7825).

### Background

For ZeRO3 [leaf modules](https://deepspeed.readthedocs.io/en/latest/training.html#configuring-zero-leaf-modules), when a module returns multiple output tensors, PyTorch's autograd can trigger backward hooks from multiple threads concurrently. This causes race conditions in parameter state management. PR #7825 adds thread synchronization to prevent this.

### Test Configuration

| Parameter | Value |
|-----------|-------|
| **Model** | `mistralai/Mixtral-8x7B-v0.1` (full: 32 layers, reduced: 4 layers) |
| **Batch Size** | 4 |
| **Sequence Length** | 1024 |
| **ZeRO Stage** | 3 |
| **Hardware** | 8x NVIDIA H100 80GB HBM3 |
| **Leaf Modules** | `MixtralSparseMoeBlock` |
| **Warmup Steps** | 10 |
| **Benchmark Steps** | 30 |
| **Activation Checkpointing** | Enabled |
| **Mixed Precision** | BF16 with `torch_autocast` |
| **Low-Precision Master States** | `bf16_master_weights_and_grads: true`, `bf16_optimizer_states: true` |

### DeepSpeed Config

```json
{
    "bf16": {
        "enabled": true,
        "bf16_master_weights_and_grads": true,
        "bf16_optimizer_states": true
    },
    "torch_autocast": {
        "enabled": true,
        "dtype": "bfloat16"
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true
    },
    "gradient_accumulation_steps": 1,
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
```

### Results

#### Full Model (32 layers)

| Condition | Avg Iteration Time | Memory (Alloc) | Memory (Peak) |
|-----------|-------------------|----------------|---------------|
| **With Fix v2** (backward-only sync) | **2.5065s** | 59.72 GB | 67.83 GB |
| **With Fix v1** (forward+backward sync) | 2.5205s | 59.72 GB | 67.83 GB |
| **Without Fix** (master) | 2.5182s | 59.72 GB | 67.83 GB |
| **v2 vs Master** | **-0.46%** | 0% | 0% |

#### Reduced Model (4 layers, for quick testing)

| Condition | Avg Iteration Time | Memory (Alloc) | Memory (Peak) |
|-----------|-------------------|----------------|---------------|
| **With Fix v2** (backward-only sync) | **0.3687s** | 8.92 GB | 16.40 GB |
| **With Fix v1** (forward+backward sync) | 0.3693s | 8.92 GB | 16.40 GB |
| **Without Fix** (master) | 0.3660s | 8.92 GB | 16.40 GB |
| **v2 vs Master** | **+0.74%** | 0% | 0% |

**Note:** v2 skips synchronization on forward passes (which are single-threaded), only synchronizing during backward passes where the race condition can occur.

### Commands to Reproduce

```bash
# 1. Ensure DeepSpeed is installed from the fix branch
cd /home/ray/default/ds/DeepSpeed
git checkout tohtana/fix_leaf_module_race_condition
pip install -e .

# 2. Run benchmark WITH the fix
cd /home/ray/default/ds/ds_verify_loss
bash run.sh \
    --model "mistralai/Mixtral-8x7B-v0.1" \
    --batch_size 4 \
    --seq_length 1024 \
    --num_layers 4 \
    --bench_step 30 \
    --warmup_step 10 \
    --activation_checkpointing \
    --use_leaf_modules \
    2>&1 | tee perf_results/with_fix.log

# 3. Switch to master branch and run WITHOUT the fix
cd /home/ray/default/ds/DeepSpeed
git checkout master
pip install -e .

cd /home/ray/default/ds/ds_verify_loss
bash run.sh \
    --model "mistralai/Mixtral-8x7B-v0.1" \
    --batch_size 4 \
    --seq_length 1024 \
    --num_layers 4 \
    --bench_step 30 \
    --warmup_step 10 \
    --activation_checkpointing \
    --use_leaf_modules \
    2>&1 | tee perf_results/without_fix.log
```

**Date:** 2026-01-30

---

## Usage

### 1. Run training

### 1.1. Run basic conditions

```bash
./run_batch.sh
```

**What it does:**
- Runs training with different conditions (ZeRO-1/2/3 with and without DeepCompile)
- Records loss values and iteration times

### 1.2. Run specific condition sets

Instead of running all conditions, you can define and run specific condition sets using the configuration-based approach:

```bash
# List available condition sets
python generate_batch.py --list

# Show what commands would be run (dry run)
python generate_batch.py zero_vs_fsdp --show

# Generate a reusable bash script
python generate_batch.py zero_vs_fsdp --generate run_zero_vs_fsdp.sh

# Generate and run immediately
python generate_batch.py zero_vs_fsdp --run
```

**Usage modes:**

- `--list`: Display all available condition sets with descriptions
- `--show` (default): Preview the commands that would be executed without running them
- `--generate <script_file>`: Create a standalone bash script for later execution
- `--run`: Generate the batch script and execute it immediately

**Generated scripts include:**
- Timestamped results directory creation
- Comprehensive experiment logging
- Status tracking for each condition
- WandB run ID extraction
- Detailed log file management
- Summary table with success/failure status

**Pre-defined condition sets:**

- `zero_stages`: Compare ZeRO-1, ZeRO-2, and ZeRO-3
- `zero_vs_fsdp`: Compare ZeRO-3 with FSDP 
- `fsdp_vs_ddp`: Compare FSDP with DDP
- `compilation_comparison`: Compare with/without compilation
- `minimal_test`: Quick test with ZeRO-3 baseline and compiled

**Creating custom condition sets:**

Edit `conditions.yaml` to define your own condition sets:

```yaml
my_custom_test:
  description: "Custom comparison for my research"
  conditions:
    - name: "baseline"
      backend: "deepspeed"
      zero_stage: 3
      compile: false
    - name: "optimized" 
      backend: "deepspeed"
      zero_stage: 3
      compile: true
      passes: "INFERENCE"
```

Then run it:
```bash
python generate_batch.py my_custom_test
```

**Benefits:**
- No need to copy/modify batch files
- Easy to define targeted comparisons
- Reusable condition sets
- Clear documentation of what's being tested

### 2. Generate report
Comprehensive analysis script that extracts metrics and generates visualizations.

**Usage:**
```bash
python analyze_results.py <results_directory>
```

**Features:**
- Extracts loss values and iteration times from training logs
- Retrieves data from WandB runs when available
- Generates loss curve comparisons
- Creates iteration time performance charts
- Plots memory usage comparisons
- Produces comprehensive markdown report


## Generated Outputs

### Results Directory Structure
```
results_YYYYMMDD_HHMMSS/
├── experiment_log.txt           # Experiment summary
├── metadata.json               # Structured metadata
├── z1_baseline.log             # Raw experiment output
├── z1_baseline_detailed.log    # Detailed training log
├── z2_baseline.log
├── z2_baseline_detailed.log
├── z3_baseline.log
├── z3_baseline_detailed.log
├── z1_deepcompile.log
├── z1_deepcompile_detailed.log
├── z2_deepcompile.log
├── z2_deepcompile_detailed.log
├── z3_deepcompile.log
├── z3_deepcompile_detailed.log
├── loss_comparison.png         # Loss curves plot
├── iteration_time_comparison.png # Performance comparison
├── memory_usage_comparison.png # Memory usage plot
└── report.md                   # Comprehensive markdown report
```

### Report Contents

The generated `report.md` includes:

1. **Executive Summary** - High-level findings
2. **Experimental Setup** - Configuration details and status
3. **Results Analysis**:
   - Loss convergence comparison with embedded charts
   - Performance analysis with iteration time comparisons
   - Memory usage analysis
4. **Data Quality Assessment** - Data points collected per condition
5. **Recommendations** - Best performing configurations
6. **Appendix** - WandB run IDs and log file references



## Data Sources

The analysis script extracts data from multiple sources:

1. **Training Logs** - Iteration times, loss values, memory usage
2. **WandB Logs** - When available, provides additional metrics
3. **Experiment Metadata** - Configuration and execution details


## Run a specific configuration

The `run.sh` script provides a flexible way to run individual training experiments with customizable parameters. Unlike `run_batch.sh` which runs predefined batch experiments, `run.sh` allows you to test specific configurations.

### Basic Usage

```bash
./run.sh [Options]
```

### Configuration Items

The following parameters can be configured through command line arguments or environment variables:

#### Core Training Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | `meta-llama/Llama-2-7b-hf` | Hugging Face model identifier |
| `--batch_size` | `1` | Micro batch size per GPU |
| `--seq_length` | `512` | Maximum sequence length |
| `--num_epochs` | `5` | Number of training epochs |
| `--learning_rate` | `1e-6` | Learning rate for optimizer |
| `--max_grad_norm` | `1.0` | Gradient clipping threshold |
| `--gradient_accumulation_steps` | `1` | Steps to accumulate gradients |
| `--log_interval` | `10` | Steps between logging outputs |

#### Backend and Optimization
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--backend` | `deepspeed` | Training backend (`deepspeed`, `fsdp`, `ddp`, `singlegpu`) |
| `--zero_stage` | `3` | ZeRO optimization stage (0, 1, 2, 3) |
| `--activation_checkpointing` | `false` | Enable activation checkpointing |
| `--compile` | `false` | Enable PyTorch compilation |
| `--deepcompile` | `false` | Enable DeepSpeed compilation |
| `--passes` | `ALL` | Compilation passes to use |
| `--eager` | `false` | Use eager execution mode |
| `--offload_opt_states` | `false` | Offload optimizer states to CPU |

#### Memory and Performance
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--fp16` | `false` | Use 16-bit floating point (default: bf16) |
| `--deterministic` | `false` | Enable deterministic training |
| `--profile` | `false` | Enable performance profiling |
| `--profile_dir` | `None` | Directory for profiling outputs |
| `--bench_step` | `100` | Steps for benchmarking |
| `--warmup_step` | `15` | Warmup steps before benchmarking |

#### Data and Evaluation
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset_name` | `wikitext` | Dataset for pretraining evaluation |
| `--dataset_percentage` | `10.0` | Percentage of dataset to use |
| `--eval` | `false` | Enable evaluation mode |
| `--num_layers` | `0` | Override number of model layers (0 = use model default) |
| `--attn_impl` | `sdpa` | Attention implementation |

#### Model Persistence
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--save_weights` | `false` | Save model weights after training |
| `--load_weights` | `false` | Load model weights before training |

#### Distributed Training
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--host_ip` | `127.0.0.1` | Main process IP address |
| `--machine_rank` | `0` | Rank of current machine |
| `NUM_NODES` | `1` | Number of nodes (environment variable) |
| `NGPUS_PER_NODE` | `auto` | GPUs per node (environment variable) |

#### Logging and Monitoring
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_wandb` | `false` | Enable Weights & Biases logging |
| `--wandb_project` | `ds-verify-loss` | WandB project name |
| `--wandb_run_name` | `None` | Custom WandB run name |
| `--wandb_tags` | `[]` | Tags for WandB run |
| `--debug_log` | `false` | Enable detailed debug logging |

#### Synchronization (Debug)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sync_before_reduce` | `false` | Synchronize before gradient reduction |
| `--sync_after_reduce` | `false` | Synchronize after gradient reduction |
| `--sync_before_allgather` | `false` | Synchronize before allgather operations |
| `--sync_after_allgather` | `false` | Synchronize after allgather operations |

**Default Configuration:**
- Backend: DeepSpeed
- Model: meta-llama/Meta-Llama-3-8B
- ZeRO Stage: 3
- Batch Size: 1
- Sequence Length: 512
- Gradient Accumulation Steps: 1


### Output

The script generates:
- Configuration files in `configs/` directory
- Training logs in `logs/` directory with descriptive filenames
- Console output with real-time training progress

**Log File Naming Convention:**
```
logs/debug_n{rank}_{model}_{backend}_np{processes}z{zero_stage}c{compile}dc{deepcompile}E{eager}b{batch_size}seq{seq_length}g{gas}a{activation_checkpoint}p{passes}.log
```


## Requirements

The analysis script requires:
- Python 3.7+
- matplotlib
- seaborn
- pandas
- numpy

Install with:
```bash
pip install matplotlib seaborn pandas numpy
```

