#!/bin/bash

# Script to compare two DeepSpeed revisions using the same test configuration
# This script installs each revision, runs the same test, and compares results

set -e

# Default values
DEEPSPEED_PATH="/home/mtanaka/work/dc/DeepSpeed"
REVISION1=""
REVISION2=""
TEST_CONFIG=""
OUTPUT_DIR="revision_comparison_$(date +%Y%m%d_%H%M%S)"
RUNS_PER_REVISION=1
CLEANUP=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --deepspeed_path)
            DEEPSPEED_PATH="$2"
            shift 2
            ;;
        --revision1)
            REVISION1="$2"
            shift 2
            ;;
        --revision2)
            REVISION2="$2"
            shift 2
            ;;
        --test_config)
            TEST_CONFIG="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --runs_per_revision)
            RUNS_PER_REVISION="$2"
            shift 2
            ;;
        --no_cleanup)
            CLEANUP=0
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --revision1 <rev1> --revision2 <rev2> [options]"
            echo ""
            echo "Required arguments:"
            echo "  --revision1 <rev1>        First DeepSpeed revision (commit, tag, or branch)"
            echo "  --revision2 <rev2>        Second DeepSpeed revision (commit, tag, or branch)"
            echo ""
            echo "Optional arguments:"
            echo "  --deepspeed_path <path>   Path to DeepSpeed repository (default: ../DeepSpeed)"
            echo "  --test_config <config>    Test configuration string (default: basic test)"
            echo "  --output_dir <dir>        Output directory for results (default: auto-generated)"
            echo "  --runs_per_revision <n>   Number of runs per revision (default: 1)"
            echo "  --no_cleanup              Don't cleanup intermediate files"
            echo "  -h, --help                Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --revision1 v0.12.0 --revision2 main"
            echo "  $0 --revision1 abc123 --revision2 def456 --test_config '--zero_stage 3 --batch_size 2'"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$REVISION1" || -z "$REVISION2" ]]; then
    echo "Error: Both --revision1 and --revision2 are required"
    echo "Use --help for usage information"
    exit 1
fi

# Check if DeepSpeed path exists
if [[ ! -d "$DEEPSPEED_PATH" ]]; then
    echo "Error: DeepSpeed path does not exist: $DEEPSPEED_PATH"
    exit 1
fi

# Default test configuration if not provided
if [[ -z "$TEST_CONFIG" ]]; then
    TEST_CONFIG="--backend deepspeed --zero_stage 3 --batch_size 1 --seq_length 512"
fi

echo "DeepSpeed Revision Comparison"
echo "=============================="
echo "DeepSpeed Path: $DEEPSPEED_PATH"
echo "Revision 1: $REVISION1"
echo "Revision 2: $REVISION2"
echo "Test Config: $TEST_CONFIG"
echo "Output Dir: $OUTPUT_DIR"
echo "Runs per revision: $RUNS_PER_REVISION"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

# Store original DeepSpeed commit for restoration
echo "Storing original DeepSpeed state..."
cd "$DEEPSPEED_PATH"
ORIGINAL_COMMIT=$(git rev-parse HEAD)
ORIGINAL_BRANCH=$(git branch --show-current || echo "detached")
cd - > /dev/null

# Function to install DeepSpeed revision
install_deepspeed_revision() {
    local revision=$1
    echo "Installing DeepSpeed revision: $revision"
    
    cd "$DEEPSPEED_PATH"
    
    # Stash any uncommitted changes
    git stash push -m "compare_revisions_temp_stash" || true
    
    # Checkout the revision
    git checkout "$revision"
    
    # Install the revision
    pip3 uninstall -y deepspeed || true
    pip3 install --user -e . --no-build-isolation
    
    cd - > /dev/null
    
    # Verify installation
    python3 -c "import deepspeed; print(f'DeepSpeed version: {deepspeed.__version__}')"
}

# Function to run test with current DeepSpeed revision
run_test() {
    local revision=$1
    local run_number=$2
    
    echo "Running test for revision $revision (run $run_number)..."
    
    # Create revision-specific log directory
    local log_dir="logs_${revision//\//_}_run${run_number}"
    mkdir -p "$log_dir"
    
    # Store the absolute path to the revision log directory
    local abs_log_dir="$(pwd)/$log_dir"
    
    # Run the test using run.sh
    cd ..
    
    # Get timestamp before running to identify new log files
    local before_files=$(ls -1 logs/*.log 2>/dev/null | wc -l)
    
    # Run the test
    ./run.sh $TEST_CONFIG
    
    # Find and move the newest log file to the revision directory
    local newest_log=$(ls -1t logs/*.log 2>/dev/null | head -1)
    if [[ -n "$newest_log" && -f "$newest_log" ]]; then
        echo "Moving log file: $newest_log -> $abs_log_dir/"
        mv "$newest_log" "$abs_log_dir/"
    else
        echo "Warning: No log file found after test run"
    fi
    
    cd "$OUTPUT_DIR"
    
    echo "Test completed for revision $revision (run $run_number)"
}

# Function to restore original DeepSpeed state
restore_original_deepspeed() {
    echo "Restoring original DeepSpeed state..."
    cd "$DEEPSPEED_PATH"
    
    if [[ "$ORIGINAL_BRANCH" != "detached" ]]; then
        git checkout "$ORIGINAL_BRANCH" || git checkout "$ORIGINAL_COMMIT"
    else
        git checkout "$ORIGINAL_COMMIT"
    fi
    
    # Restore stashed changes if any
    git stash list | grep "compare_revisions_temp_stash" && git stash pop || true
    
    # Reinstall original version
    pip uninstall -y deepspeed || true
    pip install --user -e . --no-build-isolation
    
    cd - > /dev/null
    echo "Original DeepSpeed state restored"
}

# Trap to ensure cleanup on exit
if [[ "$CLEANUP" == "1" ]]; then
    trap restore_original_deepspeed EXIT
fi

# Main execution
echo "Starting revision comparison..."

# Test revision 1
for ((run=1; run<=RUNS_PER_REVISION; run++)); do
    install_deepspeed_revision "$REVISION1"
    run_test "$REVISION1" "$run"
done

# Test revision 2
for ((run=1; run<=RUNS_PER_REVISION; run++)); do
    install_deepspeed_revision "$REVISION2"
    run_test "$REVISION2" "$run"
done

echo ""
echo "All tests completed!"
echo "Results stored in: $OUTPUT_DIR"
echo ""
echo "To analyze results, run:"
echo "  python ../analyze_revision_comparison.py $OUTPUT_DIR"
