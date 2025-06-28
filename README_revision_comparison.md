# DeepSpeed Revision Comparison

This directory contains tools to compare the performance of two different DeepSpeed revisions using the same test configuration.

## Files

- `compare_revisions.sh` - Main script to run comparison tests
- `analyze_revision_comparison.py` - Analysis script to compare results
- `README_revision_comparison.md` - This documentation

## Usage

### Basic Comparison

```bash
# Compare two DeepSpeed revisions
./compare_revisions.sh --revision1 v0.12.0 --revision2 main

# Compare specific commits
./compare_revisions.sh --revision1 abc123 --revision2 def456
```

### Advanced Options

```bash
# Custom test configuration
./compare_revisions.sh \
    --revision1 v0.12.0 \
    --revision2 main \
    --test_config "--zero_stage 3 --batch_size 2 --seq_length 1024" \
    --runs_per_revision 3 \
    --output_dir my_comparison_results

# Specify DeepSpeed path
./compare_revisions.sh \
    --revision1 v0.12.0 \
    --revision2 main \
    --deepspeed_path /path/to/DeepSpeed
```

### Analysis

After running the comparison, analyze the results:

```bash
# Analyze results (output directory from compare_revisions.sh)
python analyze_revision_comparison.py revision_comparison_20240621_143022

# With test config info for the report
python analyze_revision_comparison.py revision_comparison_20240621_143022 \
    --test_config "--zero_stage 3 --batch_size 2"
```

## Output

The comparison generates:
- **Structured logs** for each revision and run
- **Comparison plots** showing loss, iteration time, and memory usage
- **Markdown report** with detailed analysis and interpretation
- **Raw data** in JSON format for further analysis

## Example Output Structure

```
revision_comparison_20240621_143022/
├── logs_v0.12.0_run1/
│   └── debug_n0_Meta-Llama-3-8B_deepspeed_np2z3c0dc0E0b1seq512g1a1pALL.log
├── logs_main_run1/
│   └── debug_n0_Meta-Llama-3-8B_deepspeed_np2z3c0dc0E0b1seq512g1a1pALL.log
├── revision_comparison.png
├── revision_comparison_report.md
└── revision_comparison_data.json
```

## Requirements

- DeepSpeed repository available locally
- Ability to install different DeepSpeed versions
- All dependencies for running verify_loss.py and run.sh

## Safety Features

- Automatically backs up and restores original DeepSpeed installation
- Stashes uncommitted changes before switching revisions
- Cleanup on script exit (can be disabled with --no_cleanup)