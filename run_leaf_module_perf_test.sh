#!/bin/bash
# Performance test for leaf module synchronization fix (PR #7825)
# Tests Mixtral with and without the fix to measure any performance impact

set -e

DEEPSPEED_DIR=${DEEPSPEED_DIR:-/home/ray/default/ds/DeepSpeed}
RESULTS_DIR=perf_results
mkdir -p ${RESULTS_DIR}

MODEL="mistralai/Mixtral-8x7B-v0.1"
BATCH_SIZE=4
SEQ_LENGTH=1024
BENCH_STEPS=30
WARMUP_STEPS=10
NUM_LAYERS=4  # Use fewer layers for faster testing

echo "======================================"
echo "Leaf Module Synchronization Perf Test"
echo "======================================"
echo "Model: ${MODEL}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Seq Length: ${SEQ_LENGTH}"
echo "Bench Steps: ${BENCH_STEPS}"
echo "Warmup Steps: ${WARMUP_STEPS}"
echo "Num Layers: ${NUM_LAYERS}"
echo "======================================"

run_test() {
    local branch=$1
    local label=$2
    local output_file="${RESULTS_DIR}/${label}.log"
    
    echo ""
    echo "======================================" 
    echo "Running test: ${label}"
    echo "Branch: ${branch}"
    echo "Output: ${output_file}"
    echo "======================================"
    
    # Checkout the specified branch in DeepSpeed
    pushd ${DEEPSPEED_DIR}
    git checkout ${branch}
    pip install -e . --quiet
    popd
    
    # Run the benchmark
    ./run.sh \
        --model "${MODEL}" \
        --batch_size ${BATCH_SIZE} \
        --seq_length ${SEQ_LENGTH} \
        --num_layers ${NUM_LAYERS} \
        --bench_step ${BENCH_STEPS} \
        --warmup_step ${WARMUP_STEPS} \
        --activation_checkpointing \
        --use_leaf_modules \
        2>&1 | tee ${output_file}
    
    # Extract iteration time from log
    local iter_time=$(grep "iteration time:" ${output_file} | tail -1 | sed 's/.*iteration time: \([0-9.]*\).*/\1/')
    echo "Iteration time for ${label}: ${iter_time}s"
    echo "${label}: ${iter_time}" >> ${RESULTS_DIR}/summary.txt
}

# Clear previous results
rm -f ${RESULTS_DIR}/summary.txt

# Test 1: With the fix (current branch)
run_test "tohtana/fix_leaf_module_race_condition" "with_fix"

# Test 2: Without the fix (master branch)
run_test "master" "without_fix"

# Return to the fix branch
pushd ${DEEPSPEED_DIR}
git checkout tohtana/fix_leaf_module_race_condition
pip install -e . --quiet
popd

echo ""
echo "======================================"
echo "RESULTS SUMMARY"
echo "======================================"
cat ${RESULTS_DIR}/summary.txt
echo "======================================"
