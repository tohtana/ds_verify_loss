# Verification of Training with DeepSpeed

The scripts in this repository run training using DeepSpeed with different settings. They also plots loss curves and iteration times for comparison.

## Alpamayo2-Super FSDP versus ZeRO-3 throughput

`alpamayo2_benchmark.py` is a thin, purpose-built adapter for a reproducible
single-node, eight-H100 training-throughput comparison. It loads the 32B VLM
weights from `nvidia/Alpamayo2-Super` revision
`00554695e729a6ff0b6281fd2c81b18d06e33dbe`, makes the VLM trainable, and
sets `enable_expert=false` before model construction. The 2.3B action expert is
therefore not instantiated or loaded. This is a `vlm_train` execution-speed
benchmark; its loss observations are numerical-debug evidence only, not a
training-quality or convergence evaluation.

The two rows use micro-batch one per rank, gradient accumulation one, AdamW at
`1e-6`, activation checkpointing, SDPA, BF16 autocast, one warmup step, and
124 measured synchronized end-to-end train steps. FSDP uses `FULL_SHARD`
with an explicit `Qwen3VLTextDecoderLayer` wrap policy. DeepSpeed uses ZeRO-3
and the low-precision state plus `torch_autocast` fields in
`configs/alpamayo2_zero3.json`.

The optional `deepspeed-deepcompile` lane retains that exact ZeRO-3 precision,
optimizer, batch, and checkpointing contract and adds only
`"compile": {"deepcompile": true}` through
`configs/alpamayo2_zero3_deepcompile.json`. The runner calls
`engine.compile()` after `deepspeed.initialize()` and fails if
`engine.is_deepcompile_active()` is false. It passes no custom schedule: at the
pinned DeepSpeed revision, the default ZeRO-3 schedule applies gather/release at
global step 0 and gather/release plus prefetch and selective gather at global
step 5. This lane warms up on optimizer steps 0-19 and measures steps 20-124,
while still consuming all 1,000 samples exactly once over 125 optimizer steps.

Set the following to clean checkouts of NVlabs/alpamayo2 at revision
`beb2977d9a7e9d66837d4a3ad5144ff59de37519` and DeepSpeed at revision
`79046032e5d6800a547348f6b0c7b3e1f112e5ce`, the immutable prepared shared-asset
root, and a persistent output directory, respectively. The shared root must
contain `asset-preparation-manifest.json`, the checkpoint `.complete.json`, and
the prepared COCO dataset manifest and JSONL files. The launcher installs
DeepSpeed from the pinned source and rejects a mismatched imported revision.

```bash
export ALPAMAYO2_SOURCE_REPO=/path/to/alpamayo2
export DEEPSPEED_SOURCE_REPO=/path/to/DeepSpeed
export ALPAMAYO2_CACHE_ROOT=/path/to/prepared-assets
export ALPAMAYO2_OUTPUT_ROOT=/path/to/results
export ALPAMAYO2_RUN_ID=unique-run-id
export ALPAMAYO2_LOCAL_STAGE_ROOT=/mnt/local_storage/alpamayo2/$ALPAMAYO2_RUN_ID
export DS_VERIFY_LOSS_CANDIDATE_CONTENT_SHA256=<reviewed-content-digest>
scripts/run_alpamayo2_benchmark.sh
```

The command above keeps the existing default and runs FSDP followed by ordinary
ZeRO-3. To run only the DeepCompile row without rerunning either baseline, use a
fresh run-local path and add the lane selector:

```bash
export ALPAMAYO2_BENCHMARK_LANE=deepspeed-deepcompile
export ALPAMAYO2_RUN_ID=unique-deepcompile-run-id
export ALPAMAYO2_LOCAL_STAGE_ROOT=/mnt/local_storage/alpamayo2/$ALPAMAYO2_RUN_ID
scripts/run_alpamayo2_benchmark.sh
```

Before processor construction or model loading, the launcher validates the
prepared manifests and source files, checks node-local free space, copies the
checkpoint and prepared COCO tree through a temporary directory, validates the
copy, and atomically installs it at the required run-specific node-local root.
`ALPAMAYO2_LOCAL_STAGE_ROOT` must be under the platform's node-local storage
mount and end with the explicitly selected `ALPAMAYO2_RUN_ID`. Existing roots are rejected to
avoid consuming partial data. Manifest-recorded checkpoint files, COCO source
archives, and the 1,000 selected images are SHA-256 checked after copying; all
other copied files are checked by relative path and size. The shared source is
never changed or deleted. The launcher also checks the seven-file harness content
digest against the reviewed deployment identity before staging begins.

The launcher refuses any shape other than exactly eight visible H100s and uses
non-default distributed port `29673`. Before either backend starts, batch
construction processes all 1,000 ordered COCO training records into a verified
per-sample cache. At global step `s`, rank `r` consumes sample `8*s+r`, so each
backend consumes every sample exactly once in identical order over 125 optimizer
steps. Step 0 is warmup and steps 1-124 are measured. Cache reads,
processor/tokenizer work, and device transfers are outside the synchronized
train-step timer. The timer covers forward, backward, optimizer step, and
gradient clearing and records the maximum rank duration. Model, dataset,
processor/tokenizer assets, and batch caches resolve only under the run-local
root after staging; offline modes prevent checkpoint redownloads. JSON results
contain 124 max-rank step times, all 125 global-mean loss observations (with
warmup marked), mean, median, samples/s, per-rank and global peak CUDA memory,
effective backend settings, source identities, and exact-stage failures.
For the DeepCompile lane, the result instead contains 105 measured times and
records requested/configured/active state, the compile config, the pinned
default-schedule assumption and transition steps, and both timing ranges.

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
