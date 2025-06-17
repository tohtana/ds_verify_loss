#!/bin/bash

# Enhanced batch runner with metrics collection
# This script runs six conditions and collects metrics for analysis

COMPILE_OPTS="--compile"
DC_OPTS="${COMPILE_OPTS} --deepcompile"
ACC_OPTS="--gradient_accumulation_steps 4"
AC_OPTS="--activation-checkpointing"

COMMON_ARGS="${ACC_OPTS} --dataset_percentage 20.0"


# Create results directory
RESULTS_DIR="results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULTS_DIR}"

echo "Starting batch experiments - results will be saved to ${RESULTS_DIR}"

# Define test conditions
declare -A CONDITIONS=(
    ["z1_baseline"]="--backend deepspeed --zero_stage 1 ${COMMON_ARGS} --use_wandb"
    ["z2_baseline"]="--backend deepspeed --zero_stage 2 ${COMMON_ARGS} --use_wandb"
    ["z3_baseline"]="--backend deepspeed --zero_stage 3 ${COMMON_ARGS} --use_wandb"
    ["z1_deepcompile"]="--backend deepspeed --zero_stage 1 ${COMMON_ARGS} ${DC_OPTS} --use_wandb"
    ["z2_deepcompile"]="--backend deepspeed --zero_stage 2 ${COMMON_ARGS} ${DC_OPTS} --use_wandb"
    ["z3_deepcompile"]="--backend deepspeed --zero_stage 3 ${COMMON_ARGS} ${DC_OPTS} --use_wandb"
)

# Track completion and wandb runs
declare -A STATUS
declare -A WANDB_RUNS

echo "=== Starting batch experiments ===" | tee "${RESULTS_DIR}/experiment_log.txt"
echo "Start time: $(date)" | tee -a "${RESULTS_DIR}/experiment_log.txt"

# Run each condition
for condition in "${!CONDITIONS[@]}"; do
    echo "" | tee -a "${RESULTS_DIR}/experiment_log.txt"
    echo "Running condition: ${condition}" | tee -a "${RESULTS_DIR}/experiment_log.txt"
    echo "Command: bash ./run.sh ${CONDITIONS[$condition]}" | tee -a "${RESULTS_DIR}/experiment_log.txt"
    
    # Run the experiment
    bash ./run.sh ${CONDITIONS[$condition]} 2>&1 | tee "${RESULTS_DIR}/${condition}.log"
    exit_code=$?
    
    STATUS[$condition]=$exit_code
    
    echo "Condition ${condition} completed with exit code ${exit_code}" | tee -a "${RESULTS_DIR}/experiment_log.txt"
    
    # Extract wandb run ID if available
    if [ -f "wandb/latest-run" ]; then
        wandb_run=$(cat wandb/latest-run 2>/dev/null || echo "unknown")
        WANDB_RUNS[$condition]=$wandb_run
        echo "WandB run ID: ${wandb_run}" | tee -a "${RESULTS_DIR}/experiment_log.txt"
    fi
    
    # Copy the latest log file to results directory
    LOG_DIR=logs
    latest_log=$(ls -t ${LOG_DIR}/*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ]; then
        cp "$latest_log" "${RESULTS_DIR}/${condition}_detailed.log"
        echo "Detailed log saved: ${condition}_detailed.log" | tee -a "${RESULTS_DIR}/experiment_log.txt"
    fi
done

echo "" | tee -a "${RESULTS_DIR}/experiment_log.txt"
echo "=== Experiment Summary ===" | tee -a "${RESULTS_DIR}/experiment_log.txt"
echo "End time: $(date)" | tee -a "${RESULTS_DIR}/experiment_log.txt"
echo "" | tee -a "${RESULTS_DIR}/experiment_log.txt"

# Generate summary table
echo "| Condition | Status | WandB Run |" | tee -a "${RESULTS_DIR}/experiment_log.txt"
echo "|-----------|--------|-----------|" | tee -a "${RESULTS_DIR}/experiment_log.txt"

for condition in "${!CONDITIONS[@]}"; do
    status_text="SUCCESS"
    if [ "${STATUS[$condition]}" -ne 0 ]; then
        status_text="FAILED"
    fi
    
    wandb_run="${WANDB_RUNS[$condition]:-unknown}"
    echo "| ${condition} | ${status_text} | ${wandb_run} |" | tee -a "${RESULTS_DIR}/experiment_log.txt"
done

echo "" | tee -a "${RESULTS_DIR}/experiment_log.txt"
echo "Results saved to: ${RESULTS_DIR}" | tee -a "${RESULTS_DIR}/experiment_log.txt"

# Generate metadata file for analysis script
cat > "${RESULTS_DIR}/metadata.json" << EOF
{
    "experiment_date": "$(date -Iseconds)",
    "conditions": {
EOF

first=true
for condition in "${!CONDITIONS[@]}"; do
    if [ "$first" = true ]; then
        first=false
    else
        echo "," >> "${RESULTS_DIR}/metadata.json"
    fi
    
    wandb_run="${WANDB_RUNS[$condition]:-unknown}"
    cat >> "${RESULTS_DIR}/metadata.json" << EOF
        "${condition}": {
            "command": "${CONDITIONS[$condition]}",
            "exit_code": ${STATUS[$condition]},
            "wandb_run": "${wandb_run}"
        }
EOF
done

cat >> "${RESULTS_DIR}/metadata.json" << EOF

    }
}
EOF

echo ""
echo "Batch experiments completed!"
echo "Results directory: ${RESULTS_DIR}"
echo ""
echo "📊 Key Points:"
echo "  - Loss curves and iteration times will be extracted from WandB logs"
echo "  - Total runtime shown above is for the entire script execution"
echo "  - Actual per-iteration timing is available in WandB for accurate analysis"
echo ""
echo "🚀 Next steps:"
echo "1. Run: python analyze_results.py ${RESULTS_DIR}"
echo "2. Check the generated report in ${RESULTS_DIR}/report.md"
echo "3. Use --offline flag if WandB API is not available"
