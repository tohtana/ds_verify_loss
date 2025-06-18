#!/usr/bin/env python3
"""
Configuration-based batch experiment runner.
Parses YAML configuration and generates shell commands for running experiments.
"""

import yaml
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and parse the YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        sys.exit(1)


def expand_template_vars(text: str, variables: Dict[str, str]) -> str:
    """Expand template variables in text using {variable_name} format."""
    try:
        return text.format(**variables)
    except KeyError as e:
        print(f"Error: Undefined variable {e} in template")
        sys.exit(1)


def list_condition_sets(config: Dict[str, Any]) -> None:
    """List all available condition sets."""
    print("Available condition sets:")
    print("=" * 50)
    
    for set_name, set_config in config['condition_sets'].items():
        description = set_config.get('description', 'No description')
        conditions = list(set_config['conditions'].keys())
        
        print(f"\n📋 {set_name}")
        print(f"   Description: {description}")
        print(f"   Conditions: {', '.join(conditions)} ({len(conditions)} total)")


def generate_run_commands(config: Dict[str, Any], condition_set: str) -> List[Dict[str, str]]:
    """Generate run commands for a specific condition set."""
    
    if condition_set not in config['condition_sets']:
        print(f"Error: Condition set '{condition_set}' not found.")
        print("Available sets:", list(config['condition_sets'].keys()))
        sys.exit(1)
    
    # Prepare template variables
    common_options = config.get('common_options', {})
    base_args = config.get('base_args', '')
    
    # Create variables dictionary
    variables = dict(common_options)
    variables['base_args'] = expand_template_vars(base_args, common_options)
    
    # Get condition set
    set_config = config['condition_sets'][condition_set]
    conditions = set_config['conditions']
    
    # Generate commands
    commands = []
    for condition_name, condition_config in conditions.items():
        args = expand_template_vars(condition_config['args'], variables)
        description = condition_config.get('description', condition_name)
        
        commands.append({
            'name': condition_name,
            'args': args,
            'description': description
        })
    
    return commands


def print_commands(commands: List[Dict[str, str]], condition_set: str, config: Dict[str, Any]) -> None:
    """Print the generated commands."""
    set_config = config['condition_sets'][condition_set]
    set_description = set_config.get('description', 'No description')
    
    print(f"Condition Set: {condition_set}")
    print(f"Description: {set_description}")
    print(f"Total Conditions: {len(commands)}")
    print("=" * 80)
    
    for i, cmd in enumerate(commands, 1):
        print(f"\n{i}. {cmd['name']}")
        print(f"   Description: {cmd['description']}")
        print(f"   Command: bash ./run.sh {cmd['args']}")


def generate_bash_script(commands: List[Dict[str, str]], condition_set: str, 
                        config: Dict[str, Any], output_file: str = None) -> str:
    """Generate a bash script to run the conditions."""
    
    set_config = config['condition_sets'][condition_set]
    set_description = set_config.get('description', 'No description')
    
    script_lines = [
        "#!/bin/bash",
        "",
        f"# Generated batch script for condition set: {condition_set}",
        f"# Description: {set_description}",
        f"# Generated on: $(date)",
        "",
        "# Create results directory",
        'RESULTS_BASE_DIR="results"',
        'mkdir -p "${RESULTS_BASE_DIR}"',
        f'RESULTS_DIR="${{RESULTS_BASE_DIR}}/results_{condition_set}_$(date +%Y%m%d_%H%M%S)"',
        'mkdir -p "${RESULTS_DIR}"',
        "",
        'echo "Starting batch experiments - results will be saved to ${RESULTS_DIR}"',
        "",
        "# Define test conditions",
        "declare -A CONDITIONS=(",
    ]
    
    # Add condition definitions
    for cmd in commands:
        script_lines.append(f'    ["{cmd["name"]}"]={repr(cmd["args"])}')
    
    script_lines.extend([
        ")",
        "",
        "# Track completion and wandb runs",
        "declare -A STATUS",
        "declare -A WANDB_RUNS",
        "",
        'echo "=== Starting batch experiments ===" | tee "${RESULTS_DIR}/experiment_log.txt"',
        'echo "Start time: $(date)" | tee -a "${RESULTS_DIR}/experiment_log.txt"',
        "",
        "# Define the order for running conditions"
    ])
    
    # Build condition order array
    condition_names = " ".join(f'"{cmd["name"]}"' for cmd in commands)
    script_lines.append(f'CONDITION_ORDER=({condition_names})')
    
    script_lines.extend([
        "",
        "# Run each condition in defined order",
        'for condition in "${CONDITION_ORDER[@]}"; do',
        '    echo "" | tee -a "${RESULTS_DIR}/experiment_log.txt"',
        '    echo "Running condition: ${condition}" | tee -a "${RESULTS_DIR}/experiment_log.txt"',
        '    echo "Command: bash ./run.sh ${CONDITIONS[$condition]}" | tee -a "${RESULTS_DIR}/experiment_log.txt"',
        '    ',
        '    # Run the experiment',
        '    bash ./run.sh ${CONDITIONS[$condition]} 2>&1 | tee "${RESULTS_DIR}/${condition}.log"',
        '    exit_code=$?',
        '    ',
        '    STATUS[$condition]=$exit_code',
        '    ',
        '    echo "Condition ${condition} completed with exit code ${exit_code}" | tee -a "${RESULTS_DIR}/experiment_log.txt"',
        '    ',
        '    # Extract wandb run ID if available',
        '    if [ -f "wandb/latest-run" ]; then',
        '        wandb_run=$(cat wandb/latest-run 2>/dev/null || echo "unknown")',
        '        WANDB_RUNS[$condition]=$wandb_run',
        '        echo "WandB run ID: ${wandb_run}" | tee -a "${RESULTS_DIR}/experiment_log.txt"',
        '    fi',
        '    ',
        '    # Copy the latest log file to results directory',
        '    LOG_DIR=logs',
        '    latest_log=$(ls -t ${LOG_DIR}/*.log 2>/dev/null | head -1)',
        '    if [ -n "$latest_log" ]; then',
        '        cp "$latest_log" "${RESULTS_DIR}/${condition}_detailed.log"',
        '        echo "Detailed log saved: ${condition}_detailed.log" | tee -a "${RESULTS_DIR}/experiment_log.txt"',
        '    fi',
        'done',
        "",
        'echo "" | tee -a "${RESULTS_DIR}/experiment_log.txt"',
        'echo "=== Experiment Summary ===" | tee -a "${RESULTS_DIR}/experiment_log.txt"',
        'echo "End time: $(date)" | tee -a "${RESULTS_DIR}/experiment_log.txt"',
        'echo "" | tee -a "${RESULTS_DIR}/experiment_log.txt"',
        "",
        "# Generate metadata.json for analyze_results.py",
        'cat > "${RESULTS_DIR}/metadata.json" << EOF',
        '{',
        f'  "experiment_name": "{condition_set}",',
        f'  "description": "{set_description}",',
        '  "timestamp": "$(date -Iseconds)",',
        '  "condition_order": [' + ', '.join(f'"{cmd["name"]}"' for cmd in commands) + '],',
        '  "conditions": {'
    ])
    
    # Add condition metadata
    for i, cmd in enumerate(commands):
        comma = "," if i < len(commands) - 1 else ""
        script_lines.extend([
            f'    "{cmd["name"]}": {{',
            f'      "description": "{cmd["description"]}",',
            f'      "args": "{cmd["args"]}",',
            '      "status": "${STATUS[' + cmd["name"] + ']:-unknown}",',
            '      "wandb_run": "${WANDB_RUNS[' + cmd["name"] + ']:-unknown}"',
            f'    }}{comma}'
        ])
    
    script_lines.extend([
        '  }',
        '}',
        'EOF',
        "",
        "# Generate summary table",
        'echo "| Condition | Status | WandB Run |" | tee -a "${RESULTS_DIR}/experiment_log.txt"',
        'echo "|-----------|--------|-----------|" | tee -a "${RESULTS_DIR}/experiment_log.txt"',
        "",
        'for condition in "${CONDITION_ORDER[@]}"; do',
        '    status_text="SUCCESS"',
        '    if [ "${STATUS[$condition]}" -ne 0 ]; then',
        '        status_text="FAILED"',
        '    fi',
        '    ',
        '    wandb_run="${WANDB_RUNS[$condition]:-unknown}"',
        '    echo "| ${condition} | ${status_text} | ${wandb_run} |" | tee -a "${RESULTS_DIR}/experiment_log.txt"',
        'done',
        "",
        'echo "" | tee -a "${RESULTS_DIR}/experiment_log.txt"',
        'echo "Results saved to: ${RESULTS_DIR}" | tee -a "${RESULTS_DIR}/experiment_log.txt"',
        "",
        'echo ""',
        'echo "Batch experiments completed!"',
        'echo "Results directory: ${RESULTS_DIR}"',
        'echo ""',
        'echo "📊 Key Points:"',
        'echo "  - Loss curves and iteration times will be extracted from WandB logs"',
        'echo "  - Total runtime shown above is for the entire script execution"',
        'echo "  - Actual per-iteration timing is available in WandB for accurate analysis"',
        'echo ""',
        'echo "🚀 Next steps:"',
        'echo "1. Run: python analyze_results.py ${RESULTS_DIR}"',
        'echo "2. Check the generated report in ${RESULTS_DIR}/report.md"',
        'echo "3. Use --offline flag if WandB API is not available"',
        'echo "4. Use --warmup-steps N to exclude N warmup steps from timing analysis"',
    ])
    
    script_content = "\n".join(script_lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(script_content)
        # Make the script executable
        import os
        os.chmod(output_file, 0o755)
        print(f"Bash script generated: {output_file}")
    
    return script_content


def main():
    parser = argparse.ArgumentParser(
        description="Configuration-based batch experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available condition sets
  python generate_batch.py --list
  
  # Show commands for zero vs fsdp comparison
  python generate_batch.py zero_vs_fsdp --show
  
  # Generate bash script for zero vs fsdp comparison
  python generate_batch.py zero_vs_fsdp --generate run_zero_vs_fsdp.sh
  
  # Generate and run immediately
  python generate_batch.py compile_comparison --run
        """
    )
    
    parser.add_argument(
        'condition_set', 
        nargs='?',
        help='Name of the condition set to use'
    )
    parser.add_argument(
        '--config', 
        default='batch_conditions.yaml',
        help='Path to the YAML configuration file (default: batch_conditions.yaml)'
    )
    parser.add_argument(
        '--list', 
        action='store_true',
        help='List all available condition sets'
    )
    parser.add_argument(
        '--show', 
        action='store_true',
        help='Show the commands that would be run (dry run)'
    )
    parser.add_argument(
        '--generate', 
        metavar='SCRIPT_FILE',
        help='Generate bash script file'
    )
    parser.add_argument(
        '--run', 
        action='store_true',
        help='Generate and run the batch script immediately'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # List condition sets
    if args.list:
        list_condition_sets(config)
        return
    
    # Require condition set for other operations
    if not args.condition_set:
        parser.error("condition_set is required unless using --list")
    
    # Generate commands
    commands = generate_run_commands(config, args.condition_set)
    
    # Show commands (dry run)
    if args.show:
        print_commands(commands, args.condition_set, config)
        return
    
    # Generate bash script
    if args.generate:
        generate_bash_script(commands, args.condition_set, config, args.generate)
        return
    
    # Generate and run immediately
    if args.run:
        import tempfile
        import subprocess
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            script_content = generate_bash_script(commands, args.condition_set, config)
            f.write(script_content)
            temp_script = f.name
        
        try:
            import os
            os.chmod(temp_script, 0o755)
            print(f"Running generated script...")
            subprocess.run(['bash', temp_script], check=True)
        finally:
            os.unlink(temp_script)
        return
    
    # Default: show commands
    print_commands(commands, args.condition_set, config)


if __name__ == '__main__':
    main()
