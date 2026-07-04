#!/bin/bash
#SBATCH --job-name=tfb-qwen3-sweep
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=logs/sweep-h100-%j.out
#SBATCH --error=logs/sweep-h100-%j.err
#
# Full Qwen3 training-framework benchmark sweep on ONE 8xH100 node.
# Submit from the repo root:
#     sbatch scripts/sweep_h100.sbatch
# Scope it down via --export (ALL keeps your env), e.g. a quick smoke:
#     sbatch --export=ALL,MBS=1,SEQS=2048,AC_MODES=on scripts/sweep_h100.sbatch
# Per-model framework sets are overridable too (FW_14B / FW_32B / FW_30B).
#
# NOTE: H100 = 80 GB (vs H200 140 GB) -> expect MORE OOMs, especially 32B/30B. Cells that
# OOM are capped at CELL_TIMEOUT and marked failed; the sweep continues. The venvs on /fsx
# are sm_90 (Hopper) builds, so they run on H100 unchanged.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
export NGPUS_PER_NODE=8

MBS="${MBS:-1 2 4}"
SEQS="${SEQS:-1024 2048 4096}"
AC_MODES="${AC_MODES:-on off}"
CELL_TIMEOUT="${CELL_TIMEOUT:-900}"

echo "=== $(date -u) on $(hostname) ==="
nvidia-smi -L || true
echo "MBS=[$MBS]  SEQS=[$SEQS]  AC=[$AC_MODES]  cell_timeout=${CELL_TIMEOUT}s"

run_model () {   # $1=model  $2=frameworks
  local model="$1" fw="$2" ac
  for ac in $AC_MODES; do
    echo "############ $model | ac=$ac | frameworks=[$fw] ############"
    scripts/run_deepcompile_repro_matrix.sh --model "$model" \
      --frameworks "$fw" --mbs "$MBS" --seqs "$SEQS" --ac "$ac" \
      --cell-timeout-s "$CELL_TIMEOUT" || echo "WARN: $model ac=$ac returned $?"
  done
}

# DeepCompile keeps the full param set materialized per rank, so it OOMs at 30B-scale
# (even on H200); on H100's 80 GB it is 14B-only. The other 4 run for all three models.
run_model Qwen/Qwen3-14B      "${FW_14B:-fsdp deepspeed deepcompile torchtitan megatron}"
run_model Qwen/Qwen3-32B      "${FW_32B:-fsdp deepspeed torchtitan megatron}"
run_model Qwen/Qwen3-30B-A3B  "${FW_30B:-fsdp deepspeed torchtitan megatron}"

echo "=== ALL DONE $(date -u). Results: repro_matrix_runs/*-8xh100-repro/ ==="
