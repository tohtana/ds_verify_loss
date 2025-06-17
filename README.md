# DeepSpeed Loss Verification with Enhanced Analysis

This enhanced version of the DeepSpeed loss verification suite includes automated data collection, visualization, and reporting capabilities.

## Overview

The suite now includes:

1. **Enhanced batch execution** with data collection
2. **Automatic metric extraction** from training logs and WandB
3. **Visualization generation** (loss curves, iteration times, memory usage)
4. **Markdown report generation** with embedded charts

## Scripts

### 1. `run_batch.sh` (Modified Original)
The original batch script, enhanced with data collection and organized output.

**Usage:**
```bash
./run_batch.sh
```

**What it does:**
- Runs six conditions (ZeRO-1/2/3 with and without DeepCompile)
- Creates timestamped results directory
- Captures all outputs and logs
- Provides analysis instructions

### 2. `run_batch_enhanced.sh` (New Enhanced Version)
A more comprehensive batch runner with detailed metadata collection.

**Usage:**
```bash
./run_batch_enhanced.sh
```

**What it does:**
- Same experiments as original but with enhanced tracking
- Captures WandB run IDs
- Creates structured metadata for analysis
- Provides detailed experiment logging

### 3. `analyze_results.py` (New Analysis Engine)
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

### 4. `demo_analysis.py` (New Demo Script)
Test the analysis functionality with existing log data.

**Usage:**
```bash
python demo_analysis.py
```

**What it does:**
- Uses existing logs in the `logs/` directory
- Creates sample analysis to demonstrate functionality
- Generates example plots and reports

## Quick Start

### Option 1: Run New Experiments
```bash
# Run experiments with enhanced tracking
./run_batch_enhanced.sh

# Analyze results (replace with actual directory name)
python analyze_results.py results_20250616_143000
```

### Option 2: Test with Existing Data
```bash
# Create demo analysis from existing logs
python demo_analysis.py
```

### Option 3: Use Modified Original Script
```bash
# Run with the enhanced original script
./run_batch.sh

# Analyze results (replace with actual directory name)
python analyze_results.py batch_results_20250616_143000
```

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

### Generated Charts

1. **Loss Comparison Plot** (`loss_comparison.png`)
   - Line plot showing training loss over steps
   - Different colors for each configuration
   - Legend identifying ZeRO stage and variant

2. **Iteration Time Comparison** (`iteration_time_comparison.png`)
   - Bar chart comparing average iteration times
   - Error bars showing standard deviation
   - Value labels on bars

3. **Memory Usage Comparison** (`memory_usage_comparison.png`)
   - Bar chart showing peak GPU memory usage
   - Values in GB for easy interpretation

## Data Sources

The analysis script extracts data from multiple sources:

1. **Training Logs** - Iteration times, loss values, memory usage
2. **WandB Logs** - When available, provides additional metrics
3. **Experiment Metadata** - Configuration and execution details

### Log Parsing

The script looks for patterns like:
```
Epoch 1, Step 10, Loss: 2.345678 sync: True time: 0.123 alloc_mem: 1234567890 peak_mem: 2345678901
```

And extracts:
- Epoch and step numbers
- Loss values
- Iteration times
- Memory allocation and peak usage

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

## Customization

### Adding New Conditions

To add new experimental conditions, modify the `CONDITIONS` array in the batch scripts:

```bash
CONDITIONS=(
    ["condition_name"]="--backend deepspeed --your_args_here --use_wandb"
)
```

### Customizing Analysis

The `analyze_results.py` script can be modified to:
- Extract additional metrics from logs
- Generate different types of plots
- Customize the report format
- Add statistical analysis

### Custom Log Parsing

To parse different log formats, modify the `parse_log_file()` function in `analyze_results.py`.

## Integration with Existing Workflow

These scripts are designed to work with the existing DeepSpeed verification setup:

- Use the same `run.sh` script for individual experiments
- Compatible with existing WandB logging
- Works with current log file formats
- Preserves all original functionality

## Troubleshooting

### No Data Found
- Ensure experiments completed successfully
- Check that logs contain the expected format
- Verify WandB integration is working

### Missing Plots
- Install required Python packages
- Check write permissions in results directory
- Ensure matplotlib backend is properly configured

### Analysis Errors
- Verify results directory structure
- Check metadata.json format
- Ensure log files are accessible

## Examples

See the generated reports in any results directory for examples of the analysis output and chart formats.
