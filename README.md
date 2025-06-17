# Verification of Training with DeepSpeed

The scripts in this repository run training using DeepSpeed with different settings. They also plots loss curves and iteration times for comparison.

## Usage

### 1. Run training

```bash
./run_batch.sh
```

**What it does:**
- Runs training with different conditions (ZeRO-1/2/3 with and without DeepCompile)
- Records loss values and iteration times


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

