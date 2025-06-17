#!/usr/bin/env python3
"""
Analysis script to extract metrics from training logs and wandb data,
generate loss curves, iteration time comparisons, and create a markdown report.
"""

import os
import re
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import wandb
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Set up plotting style with seaborn
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def parse_log_file(log_path: str) -> Dict:
    """Extract metrics from training log file."""
    metrics = {
        'losses': [],
        'steps': [],
        'iteration_times': [],
        'memory_stats': [],
        'epochs': []
    }
    
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

def extract_wandb_data(wandb_run_path: str, condition: str = None, offline_mode: bool = False) -> Dict:
    """Extract data from wandb run directory or API."""
    metrics = {
        'losses': [],
        'steps': [],
        'iteration_times': [],
        'learning_rates': [],
        'memory_allocated': [],
        'memory_peak': []
    }
    
    try:
        # First try to extract from wandb API if run ID is available and not in offline mode
        if condition and 'wandb_run' in str(wandb_run_path) and not offline_mode:
            run_id = wandb_run_path
            try:
                api = wandb.Api()
                runs = api.runs("ds-verify-loss")  # default project name
                
                target_run = None
                for run in runs:
                    if run.id == run_id or run_id in run.name:
                        target_run = run
                        break
                
                if target_run:
                    history = target_run.history()
                    if not history.empty:
                        # Extract metrics from wandb history
                        if 'train/loss' in history.columns:
                            loss_data = history[['_step', 'train/loss']].dropna()
                            metrics['steps'].extend(loss_data['_step'].tolist())
                            metrics['losses'].extend(loss_data['train/loss'].tolist())
                        
                        if 'train/iteration_time' in history.columns:
                            time_data = history['train/iteration_time'].dropna()
                            metrics['iteration_times'].extend(time_data.tolist())
                        
                        if 'train/memory_allocated' in history.columns:
                            mem_data = history['train/memory_allocated'].dropna()
                            metrics['memory_allocated'].extend(mem_data.tolist())
                        
                        if 'train/memory_peak' in history.columns:
                            peak_data = history['train/memory_peak'].dropna()
                            metrics['memory_peak'].extend(peak_data.tolist())
                        
                        print(f"✓ Extracted {len(metrics['losses'])} loss values and {len(metrics['iteration_times'])} timing values from wandb API for {condition}")
                        return metrics
            
            except Exception as e:
                print(f"Failed to extract from wandb API for {condition}: {e}")
        
        # Fallback to local wandb files
        run_path = Path(wandb_run_path)
        if not run_path.exists():
            return metrics
            
        # Try to find events.out.tfevents files or other wandb data files
        for file_path in run_path.rglob("*"):
            if file_path.name.endswith(".jsonl"):
                # Parse wandb jsonl files
                with open(file_path, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            if '_step' in data:
                                step = data['_step']
                                if 'train/loss' in data:
                                    metrics['steps'].append(step)
                                    metrics['losses'].append(data['train/loss'])
                                if 'train/iteration_time' in data:
                                    metrics['iteration_times'].append(data['train/iteration_time'])
                                if 'train/learning_rate' in data:
                                    metrics['learning_rates'].append(data['train/learning_rate'])
                                if 'train/memory_allocated' in data:
                                    metrics['memory_allocated'].append(data['train/memory_allocated'])
                                if 'train/memory_peak' in data:
                                    metrics['memory_peak'].append(data['train/memory_peak'])
                        except json.JSONDecodeError:
                            continue
    
    except Exception as e:
        print(f"Error extracting wandb data from {wandb_run_path}: {e}")
    
    return metrics

def parse_condition_name(condition: str) -> Tuple[str, str]:
    """Parse condition name to extract zero stage and variant."""
    if 'z1' in condition:
        zero_stage = 'ZeRO-1'
    elif 'z2' in condition:
        zero_stage = 'ZeRO-2'
    elif 'z3' in condition:
        zero_stage = 'ZeRO-3'
    else:
        zero_stage = 'Unknown'
    
    if 'deepcompile' in condition:
        variant = 'DeepCompile'
    else:
        variant = 'Baseline'
    
    return zero_stage, variant

def create_loss_comparison_plot(results_data: Dict, output_path: str):
    """Create loss curve comparison plot."""
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(results_data)))
    
    for i, (condition, data) in enumerate(results_data.items()):
        if data['losses'] and data['steps']:
            zero_stage, variant = parse_condition_name(condition)
            label = f"{zero_stage} ({variant})"
            plt.plot(data['steps'], data['losses'], 
                    label=label, color=colors[i], linewidth=2, alpha=0.8)
    
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison Across Conditions')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_iteration_time_plot(results_data: Dict, output_path: str, warmup_steps: int = 15):
    """Create iteration time comparison bar plot with enhanced styling, excluding warmup steps."""
    conditions = []
    avg_times = []
    std_times = []
    warmup_excluded_counts = []
    
    for condition, data in results_data.items():
        if data['iteration_times']:
            zero_stage, variant = parse_condition_name(condition)
            label = f"{zero_stage}\n({variant})"
            conditions.append(label)
            
            times = np.array(data['iteration_times'])
            
            # Exclude warmup steps
            if len(times) > warmup_steps:
                times_after_warmup = times[warmup_steps:]
                warmup_excluded = warmup_steps
            else:
                # If we have fewer steps than warmup, use all steps but log a warning
                times_after_warmup = times
                warmup_excluded = 0
                print(f"⚠️  Warning: {condition} has only {len(times)} steps, less than warmup_steps={warmup_steps}")
            
            warmup_excluded_counts.append(warmup_excluded)
            
            # Remove outliers (times > 99th percentile) for better visualization
            if len(times_after_warmup) > 0:
                p99 = np.percentile(times_after_warmup, 99)
                filtered_times = times_after_warmup[times_after_warmup <= p99]
                
                avg_times.append(np.mean(filtered_times))
                std_times.append(np.std(filtered_times))
            else:
                avg_times.append(0)
                std_times.append(0)
    
    if not conditions:
        print("No iteration time data found for plotting")
        # Create empty plot with message
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'No iteration time data available', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
        plt.title('Training Iteration Time Comparison')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Create figure with seaborn styling
    plt.figure(figsize=(12, 8))
    
    # Use seaborn color palette
    colors = sns.color_palette("husl", len(conditions))
    
    bars = plt.bar(conditions, avg_times, yerr=std_times, capsize=5, 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    plt.xlabel('Configuration', fontsize=14, fontweight='bold')
    plt.ylabel('Average Iteration Time (seconds)', fontsize=14, fontweight='bold')
    plt.title(f'Training Iteration Time Comparison\n(Excluding {warmup_steps} warmup steps)', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add value labels on bars
    for bar, avg_time, std_time, excluded in zip(bars, avg_times, std_times, warmup_excluded_counts):
        height = bar.get_height() + std_time
        plt.text(bar.get_x() + bar.get_width()/2, height + max(avg_times)*0.02,
                f'{avg_time:.3f}s\n±{std_time:.3f}\n({excluded} excluded)', 
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add note about warmup exclusion
    plt.figtext(0.02, 0.02, f'Note: First {warmup_steps} warmup steps excluded from analysis', 
                fontsize=10, style='italic', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Iteration time plot saved to: {output_path} (warmup steps: {warmup_steps} excluded)")

def create_memory_usage_plot(results_data: Dict, output_path: str):
    """Create memory usage comparison plot."""
    conditions = []
    peak_memories = []
    
    for condition, data in results_data.items():
        if data['memory_stats']:
            zero_stage, variant = parse_condition_name(condition)
            label = f"{zero_stage}\n({variant})"
            conditions.append(label)
            
            # Get max peak memory across all steps
            max_peak = max([stat['peak_mem'] for stat in data['memory_stats']])
            peak_memories.append(max_peak / (1024**3))  # Convert to GB
    
    if not conditions:
        print("No memory data found for plotting")
        return
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(conditions, peak_memories, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    
    plt.xlabel('Configuration')
    plt.ylabel('Peak Memory Usage (GB)')
    plt.title('Peak GPU Memory Usage Comparison')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, memory in zip(bars, peak_memories):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                f'{memory:.1f}GB', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_markdown_report(results_dir: str, results_data: Dict, metadata: Dict, warmup_steps: int = 15):
    """Generate a comprehensive markdown report."""
    report_path = os.path.join(results_dir, 'report.md')
    
    with open(report_path, 'w') as f:
        f.write("# DeepSpeed Loss Verification Experiment Report\n\n")
        f.write(f"**Generated:** {metadata.get('experiment_date', 'Unknown')}\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write("This report presents the results of a comprehensive comparison between different DeepSpeed ZeRO stages ")
        f.write("with and without DeepCompile optimization. The experiments evaluate:\n\n")
        f.write("- Training loss convergence across different configurations\n")
        f.write("- Iteration time performance\n")
        f.write("- Memory usage efficiency\n\n")
        
        # Experimental Setup
        f.write("## Experimental Setup\n\n")
        f.write("### Test Conditions\n\n")
        f.write("| Condition | Configuration | Status |\n")
        f.write("|-----------|---------------|--------|\n")
        
        for condition, info in metadata.get('conditions', {}).items():
            status = "✅ Success" if info.get('exit_code', 1) == 0 else "❌ Failed"
            zero_stage, variant = parse_condition_name(condition)
            f.write(f"| {condition} | {zero_stage} + {variant} | {status} |\n")
        
        f.write("\n### Model and Training Parameters\n\n")
        f.write("- **Model:** meta-llama/Meta-Llama-3-8B\n")
        f.write("- **Dataset:** WikiText (20% subset)\n")
        f.write("- **Batch Size:** 1 per GPU\n")
        f.write("- **Gradient Accumulation Steps:** 4\n")
        f.write("- **Sequence Length:** 512\n")
        f.write("- **Epochs:** 5\n")
        f.write(f"- **Warmup Steps Excluded:** {warmup_steps} (for timing analysis)\n\n")
        
        # Results Section
        f.write("## Results\n\n")
        
        # Loss Analysis
        f.write("### Loss Convergence Analysis\n\n")
        f.write("![Loss Comparison](loss_comparison.png)\n\n")
        
        # Analyze loss data
        loss_analysis = {}
        for condition, data in results_data.items():
            if data['losses']:
                zero_stage, variant = parse_condition_name(condition)
                final_loss = data['losses'][-1] if data['losses'] else None
                min_loss = min(data['losses']) if data['losses'] else None
                loss_analysis[condition] = {
                    'zero_stage': zero_stage,
                    'variant': variant,
                    'final_loss': final_loss,
                    'min_loss': min_loss
                }
        
        if loss_analysis:
            f.write("**Key Findings:**\n\n")
            for condition, analysis in loss_analysis.items():
                if analysis['final_loss'] is not None:
                    f.write(f"- **{analysis['zero_stage']} ({analysis['variant']}):** ")
                    f.write(f"Final loss: {analysis['final_loss']:.6f}, ")
                    f.write(f"Best loss: {analysis['min_loss']:.6f}\n")
            f.write("\n")
        
        # Performance Analysis
        f.write("### Performance Analysis\n\n")
        f.write("![Iteration Time Comparison](iteration_time_comparison.png)\n\n")
        
        # Analyze iteration times (excluding warmup steps)
        perf_analysis = {}
        for condition, data in results_data.items():
            if data['iteration_times']:
                zero_stage, variant = parse_condition_name(condition)
                times = np.array(data['iteration_times'])
                
                # Exclude warmup steps
                if len(times) > warmup_steps:
                    times_after_warmup = times[warmup_steps:]
                else:
                    times_after_warmup = times
                
                if len(times_after_warmup) > 0:
                    # Filter outliers
                    p99 = np.percentile(times_after_warmup, 99)
                    filtered_times = times_after_warmup[times_after_warmup <= p99]
                    perf_analysis[condition] = {
                        'zero_stage': zero_stage,
                        'variant': variant,
                        'avg_time': np.mean(filtered_times),
                        'std_time': np.std(filtered_times),
                        'warmup_excluded': min(warmup_steps, len(times)),
                        'total_steps': len(times)
                    }
        
        if perf_analysis:
            f.write(f"**Performance Summary** *(excluding first {warmup_steps} warmup steps)*:\n\n")
            f.write("| Configuration | Avg. Iteration Time | Std. Dev | Steps Used |\n")
            f.write("|---------------|--------------------|---------|-----------|\n")
            for condition, analysis in perf_analysis.items():
                steps_used = analysis['total_steps'] - analysis['warmup_excluded']
                f.write(f"| {analysis['zero_stage']} ({analysis['variant']}) | ")
                f.write(f"{analysis['avg_time']:.3f}s | {analysis['std_time']:.3f}s | {steps_used}/{analysis['total_steps']} |\n")
            f.write("\n")
        
        # Memory Analysis
        f.write("### Memory Usage Analysis\n\n")
        f.write("![Memory Usage Comparison](memory_usage_comparison.png)\n\n")
        
        # Data Quality Section
        f.write("## Data Quality Assessment\n\n")
        f.write("| Condition | Loss Data Points | Timing Data Points | Memory Data Points |\n")
        f.write("|-----------|------------------|--------------------|-----------------|\n")
        
        for condition, data in results_data.items():
            zero_stage, variant = parse_condition_name(condition)
            loss_points = len(data['losses'])
            timing_points = len(data['iteration_times'])
            memory_points = len(data['memory_stats'])
            f.write(f"| {zero_stage} ({variant}) | {loss_points} | {timing_points} | {memory_points} |\n")
        
        f.write("\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        f.write("Based on the experimental results:\n\n")
        
        if perf_analysis:
            # Find fastest configuration
            fastest = min(perf_analysis.items(), key=lambda x: x[1]['avg_time'])
            f.write(f"- **Fastest Configuration:** {fastest[1]['zero_stage']} ({fastest[1]['variant']}) ")
            f.write(f"with {fastest[1]['avg_time']:.3f}s average iteration time\n")
        
        if loss_analysis:
            # Find best convergence
            best_loss = min(loss_analysis.items(), key=lambda x: x[1]['min_loss'] if x[1]['min_loss'] else float('inf'))
            f.write(f"- **Best Loss Convergence:** {best_loss[1]['zero_stage']} ({best_loss[1]['variant']}) ")
            f.write(f"with minimum loss of {best_loss[1]['min_loss']:.6f}\n")
        
        f.write("\n## Appendix\n\n")
        f.write("### Wandb Run IDs\n\n")
        for condition, info in metadata.get('conditions', {}).items():
            wandb_run = info.get('wandb_run', 'unknown')
            if wandb_run != 'unknown':
                f.write(f"- **{condition}:** {wandb_run}\n")
        
        f.write("\n### Log Files\n\n")
        f.write("Detailed logs for each condition are available in the results directory:\n\n")
        for condition in metadata.get('conditions', {}).keys():
            f.write(f"- `{condition}_detailed.log`\n")
    
    print(f"Report generated: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze DeepSpeed training results')
    parser.add_argument('results_dir', help='Path to results directory')
    parser.add_argument('--wandb-base-dir', default='wandb', 
                       help='Base directory for wandb runs (default: wandb)')
    parser.add_argument('--offline', action='store_true',
                       help='Run in offline mode (no wandb API calls)')
    parser.add_argument('--warmup-steps', type=int, default=15,
                       help='Number of warmup steps to exclude from timing analysis (default: 15)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory {args.results_dir} not found")
        return
    
    if args.offline:
        print("Running in offline mode - will parse logs only")
        os.environ['WANDB_MODE'] = 'offline'
    
    # Load metadata
    metadata_path = os.path.join(args.results_dir, 'metadata.json')
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    print(f"Analyzing results from: {args.results_dir}")
    
    # Extract data for each condition
    results_data = {}
    for condition, info in metadata.get('conditions', {}).items():
        print(f"Processing condition: {condition}")
        
        # Initialize metrics structure
        metrics = {
            'losses': [], 'steps': [], 'iteration_times': [], 
            'memory_stats': [], 'epochs': [],
            'memory_allocated': [], 'memory_peak': []
        }
        
        # Try to get wandb data first (preferred source for iteration times)
        wandb_run = info.get('wandb_run')
        if wandb_run and wandb_run != 'unknown' and not args.offline:
            print(f"  Extracting from wandb run: {wandb_run}")
            try:
                wandb_metrics = extract_wandb_data(wandb_run, condition, args.offline)
                if wandb_metrics['losses'] or wandb_metrics['iteration_times']:
                    metrics.update(wandb_metrics)
                    print(f"  ✓ Extracted {len(metrics['losses'])} loss values and {len(metrics['iteration_times'])} timing values from wandb")
                else:
                    print(f"  ⚠ No data found in wandb, falling back to logs")
            except Exception as e:
                print(f"  ⚠ Wandb extraction failed: {e}, falling back to logs")
        
        # Fallback to log files if wandb data is not available or incomplete
        if not metrics['losses'] and not metrics['iteration_times']:
            print(f"  Extracting from log files...")
            detailed_log = os.path.join(args.results_dir, f"{condition}_detailed.log")
            condition_log = os.path.join(args.results_dir, f"{condition}.log")
            
            # Try detailed log first, then condition log
            for log_path in [detailed_log, condition_log]:
                if os.path.exists(log_path):
                    log_metrics = parse_log_file(log_path)
                    # Merge metrics
                    for key in ['losses', 'steps', 'iteration_times', 'memory_stats', 'epochs']:
                        if key in log_metrics:
                            metrics[key].extend(log_metrics[key])
                    print(f"  ✓ Extracted from log: {len(metrics['losses'])} loss values, {len(metrics['iteration_times'])} timing measurements")
                    break
        
        results_data[condition] = metrics
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    plots_dir = args.results_dir
    loss_plot_path = os.path.join(plots_dir, 'loss_comparison.png')
    time_plot_path = os.path.join(plots_dir, 'iteration_time_comparison.png')
    memory_plot_path = os.path.join(plots_dir, 'memory_usage_comparison.png')
    
    create_loss_comparison_plot(results_data, loss_plot_path)
    create_iteration_time_plot(results_data, time_plot_path, args.warmup_steps)
    create_memory_usage_plot(results_data, memory_plot_path)
    
    # Generate report
    print("Generating markdown report...")
    generate_markdown_report(args.results_dir, results_data, metadata, args.warmup_steps)
    
    print("\n✅ Analysis complete!")
    print(f"📊 Plots saved to: {plots_dir}")
    print(f"📋 Report saved to: {os.path.join(args.results_dir, 'report.md')}")
    print(f"\nTo view the report:")
    print(f"  cat {os.path.join(args.results_dir, 'report.md')}")
    print(f"  # or open in a markdown viewer")

if __name__ == "__main__":
    main()
