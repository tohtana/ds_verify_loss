#!/usr/bin/env python3
"""
Analysis script to compare DeepSpeed revision performance results.
Parses logs from compare_revisions.sh output and generates comparison report.
"""

import os
import re
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import glob
from datetime import datetime

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def parse_log_file(log_path: str) -> Dict:
    """Extract metrics from training log file."""
    metrics = {
        'losses': [],
        'steps': [],
        'iteration_times': [],
        'memory_stats': [],
        'epochs': [],
        'revision': '',
        'run_number': 0
    }
    
    # Extract revision and run number from log path
    path_parts = Path(log_path).parts
    for part in path_parts:
        if part.startswith('logs_'):
            # Format: logs_<revision>_run<number>
            revision_match = re.match(r'logs_(.+)_run(\d+)', part)
            if revision_match:
                metrics['revision'] = revision_match.group(1).replace('_', '/')
                metrics['run_number'] = int(revision_match.group(2))
    
    if not os.path.exists(log_path):
        print(f"Warning: Log file not found: {log_path}")
        return metrics
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                # Look for training step logs with pattern:
                # "Epoch X, Step Y, Loss: Z sync: True/False time: W alloc_mem: A peak_mem: B"
                step_match = re.search(
                    r'Epoch (\d+), Step (\d+), Loss: ([\d.]+) sync: \w+ time: ([\d.]+) alloc_mem: (\d+) peak_mem: (\d+)',
                    line
                )
                if step_match:
                    epoch = int(step_match.group(1))
                    step = int(step_match.group(2))
                    loss = float(step_match.group(3))
                    iteration_time = float(step_match.group(4))
                    alloc_mem = int(step_match.group(5))
                    peak_mem = int(step_match.group(6))
                    
                    metrics['epochs'].append(epoch)
                    metrics['steps'].append(step)
                    metrics['losses'].append(loss)
                    metrics['iteration_times'].append(iteration_time)
                    metrics['memory_stats'].append({
                        'alloc_mem': alloc_mem,
                        'peak_mem': peak_mem
                    })
    
    except Exception as e:
        print(f"Error parsing log file {log_path}: {e}")
    
    return metrics

def find_log_files(results_dir: str) -> List[str]:
    """Find all log files in the results directory."""
    log_files = []
    
    # Look for log directories matching pattern logs_<revision>_run<number>
    log_dirs = glob.glob(os.path.join(results_dir, "logs_*_run*"))
    
    for log_dir in log_dirs:
        # Find .log files in each directory
        log_files_in_dir = glob.glob(os.path.join(log_dir, "*.log"))
        log_files.extend(log_files_in_dir)
    
    return log_files

def aggregate_runs(all_metrics: List[Dict]) -> Dict:
    """Aggregate metrics across multiple runs per revision."""
    revision_data = {}
    
    for metrics in all_metrics:
        revision = metrics['revision']
        if not revision:
            continue
            
        if revision not in revision_data:
            revision_data[revision] = []
        revision_data[revision].append(metrics)
    
    # Compute statistics for each revision
    aggregated = {}
    for revision, runs in revision_data.items():
        if not runs:
            continue
            
        # Aggregate across runs - focus on final values
        final_losses = []
        avg_iter_times = []
        peak_memories = []
        
        for run in runs:
            if run['losses']:
                final_losses.append(run['losses'][-1])
            if run['iteration_times']:
                avg_iter_times.append(np.mean(run['iteration_times']))
            if run['memory_stats']:
                peak_memories.append(max([mem['peak_mem'] for mem in run['memory_stats']]))
        
        aggregated[revision] = {
            'final_loss_mean': np.mean(final_losses) if final_losses else 0,
            'final_loss_std': np.std(final_losses) if len(final_losses) > 1 else 0,
            'avg_iter_time_mean': np.mean(avg_iter_times) if avg_iter_times else 0,
            'avg_iter_time_std': np.std(avg_iter_times) if len(avg_iter_times) > 1 else 0,
            'peak_memory_mean': np.mean(peak_memories) if peak_memories else 0,
            'peak_memory_std': np.std(peak_memories) if len(peak_memories) > 1 else 0,
            'num_runs': len(runs),
            'raw_runs': runs
        }
    
    return aggregated

def create_comparison_plots(aggregated_data: Dict, output_dir: str):
    """Create comparison plots between revisions."""
    revisions = list(aggregated_data.keys())
    if len(revisions) != 2:
        print(f"Warning: Expected 2 revisions, found {len(revisions)}: {revisions}")
        return
    
    rev1, rev2 = revisions
    data1 = aggregated_data[rev1]
    data2 = aggregated_data[rev2]
    
    # Create subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'DeepSpeed Revision Comparison: {rev1} vs {rev2}', fontsize=16)
    
    # 1. Loss comparison
    ax = axes[0, 0]
    losses = [data1['final_loss_mean'], data2['final_loss_mean']]
    errors = [data1['final_loss_std'], data2['final_loss_std']]
    ax.bar(revisions, losses, yerr=errors, capsize=5, alpha=0.7)
    ax.set_title('Final Loss Comparison')
    ax.set_ylabel('Loss')
    
    # 2. Iteration time comparison
    ax = axes[0, 1]
    times = [data1['avg_iter_time_mean'], data2['avg_iter_time_mean']]
    time_errors = [data1['avg_iter_time_std'], data2['avg_iter_time_std']]
    ax.bar(revisions, times, yerr=time_errors, capsize=5, alpha=0.7, color='orange')
    ax.set_title('Average Iteration Time Comparison')
    ax.set_ylabel('Time (seconds)')
    
    # 3. Memory usage comparison
    ax = axes[1, 0]
    memories = [data1['peak_memory_mean'] / (1024**3), data2['peak_memory_mean'] / (1024**3)]  # Convert to GB
    mem_errors = [data1['peak_memory_std'] / (1024**3), data2['peak_memory_std'] / (1024**3)]
    ax.bar(revisions, memories, yerr=mem_errors, capsize=5, alpha=0.7, color='green')
    ax.set_title('Peak Memory Usage Comparison')
    ax.set_ylabel('Memory (GB)')
    
    # 4. Loss curves over time (if multiple runs available)
    ax = axes[1, 1]
    colors = ['blue', 'red']
    for i, (revision, data) in enumerate(aggregated_data.items()):
        for run in data['raw_runs']:
            if run['losses'] and run['steps']:
                ax.plot(run['steps'], run['losses'], 
                       color=colors[i], alpha=0.5, linewidth=1,
                       label=f'{revision} (run {run["run_number"]})' if len(data['raw_runs']) > 1 else revision)
    
    ax.set_title('Loss Curves')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'revision_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to: {os.path.join(output_dir, 'revision_comparison.png')}")

def generate_report(aggregated_data: Dict, output_dir: str, test_config: str = ""):
    """Generate markdown report comparing revisions."""
    revisions = list(aggregated_data.keys())
    if len(revisions) != 2:
        print(f"Warning: Expected 2 revisions, found {len(revisions)}")
        return
    
    rev1, rev2 = revisions
    data1 = aggregated_data[rev1]
    data2 = aggregated_data[rev2]
    
    # Calculate relative differences
    loss_diff = ((data2['final_loss_mean'] - data1['final_loss_mean']) / data1['final_loss_mean']) * 100 if data1['final_loss_mean'] != 0 else 0
    time_diff = ((data2['avg_iter_time_mean'] - data1['avg_iter_time_mean']) / data1['avg_iter_time_mean']) * 100 if data1['avg_iter_time_mean'] != 0 else 0
    mem_diff = ((data2['peak_memory_mean'] - data1['peak_memory_mean']) / data1['peak_memory_mean']) * 100 if data1['peak_memory_mean'] != 0 else 0
    
    # Reference the comparison plot
    plot_exists = os.path.exists(os.path.join(output_dir, 'revision_comparison.png'))
    
    report = f"""# DeepSpeed Revision Comparison Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Test Configuration
```
{test_config}
```

## Revisions Compared
- **Revision 1:** {rev1}
- **Revision 2:** {rev2}

## Summary Results

### Final Loss
- **{rev1}:** {data1['final_loss_mean']:.4f} ± {data1['final_loss_std']:.4f} ({data1['num_runs']} runs)
- **{rev2}:** {data2['final_loss_mean']:.4f} ± {data2['final_loss_std']:.4f} ({data2['num_runs']} runs)
- **Difference:** {loss_diff:+.2f}% ({'+' if loss_diff > 0 else ''}{'worse' if loss_diff > 0 else 'better' if loss_diff < 0 else 'same'})

### Average Iteration Time
- **{rev1}:** {data1['avg_iter_time_mean']:.3f} ± {data1['avg_iter_time_std']:.3f} seconds
- **{rev2}:** {data2['avg_iter_time_mean']:.3f} ± {data2['avg_iter_time_std']:.3f} seconds
- **Difference:** {time_diff:+.2f}% ({'+' if time_diff > 0 else ''}{'slower' if time_diff > 0 else 'faster' if time_diff < 0 else 'same'})

### Peak Memory Usage
- **{rev1}:** {data1['peak_memory_mean']/1024**3:.2f} ± {data1['peak_memory_std']/1024**3:.2f} GB
- **{rev2}:** {data2['peak_memory_mean']/1024**3:.2f} ± {data2['peak_memory_std']/1024**3:.2f} GB
- **Difference:** {mem_diff:+.2f}% ({'+' if mem_diff > 0 else ''}{'higher' if mem_diff > 0 else 'lower' if mem_diff < 0 else 'same'})

## Comparison Plots

{'![Revision Comparison](revision_comparison.png)' if plot_exists else '(Plot file not found)'}

## Interpretation

"""
    
    # Add interpretation based on results
    if abs(loss_diff) < 1.0:
        report += "- **Loss:** No significant difference in training loss between revisions.\n"
    elif loss_diff > 0:
        report += f"- **Loss:** Revision 2 shows worse training loss ({loss_diff:.2f}% higher).\n"
    else:
        report += f"- **Loss:** Revision 2 shows better training loss ({abs(loss_diff):.2f}% lower).\n"
    
    if abs(time_diff) < 5.0:
        report += "- **Performance:** No significant difference in iteration time between revisions.\n"
    elif time_diff > 0:
        report += f"- **Performance:** Revision 2 is slower ({time_diff:.2f}% increase in iteration time).\n"
    else:
        report += f"- **Performance:** Revision 2 is faster ({abs(time_diff):.2f}% decrease in iteration time).\n"
    
    if abs(mem_diff) < 5.0:
        report += "- **Memory:** No significant difference in memory usage between revisions.\n"
    elif mem_diff > 0:
        report += f"- **Memory:** Revision 2 uses more memory ({mem_diff:.2f}% increase).\n"
    else:
        report += f"- **Memory:** Revision 2 uses less memory ({abs(mem_diff):.2f}% decrease).\n"
    
    report += f"""
## Files Generated
- `revision_comparison.png` - Comparison plots
- `revision_comparison_data.json` - Raw comparison data

## Raw Data
```json
{json.dumps(aggregated_data, indent=2, default=str)}
```
"""
    
    # Save report
    report_path = os.path.join(output_dir, 'revision_comparison_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Save raw data as JSON
    data_path = os.path.join(output_dir, 'revision_comparison_data.json')
    with open(data_path, 'w') as f:
        json.dump(aggregated_data, f, indent=2, default=str)
    
    print(f"Report saved to: {report_path}")
    print(f"Raw data saved to: {data_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze DeepSpeed revision comparison results')
    parser.add_argument('results_dir', help='Directory containing comparison results')
    parser.add_argument('--test_config', default='', help='Test configuration used (for report)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory does not exist: {args.results_dir}")
        return
    
    print(f"Analyzing results in: {args.results_dir}")
    
    # Find and parse log files
    log_files = find_log_files(args.results_dir)
    if not log_files:
        print("No log files found in results directory")
        return
    
    print(f"Found {len(log_files)} log files")
    
    # Parse all log files
    all_metrics = []
    for log_file in log_files:
        metrics = parse_log_file(log_file)
        if metrics['losses']:  # Only include if we found data
            all_metrics.append(metrics)
    
    if not all_metrics:
        print("No valid metrics found in log files")
        return
    
    # Aggregate results by revision
    aggregated_data = aggregate_runs(all_metrics)
    
    if len(aggregated_data) == 0:
        print("No data to compare")
        return
    
    print(f"Found data for revisions: {list(aggregated_data.keys())}")
    
    # Generate visualizations and report
    create_comparison_plots(aggregated_data, args.results_dir)
    generate_report(aggregated_data, args.results_dir, args.test_config)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()