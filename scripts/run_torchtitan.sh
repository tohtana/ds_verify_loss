#!/usr/bin/env bash
# Launch a single TorchTitan (Qwen3-14B, FSDP2) benchmark cell and write a
# metrics.json in the ds_verify_loss schema. Accepts the same flags the matrix
# runner passes to run.sh (deepspeed-only flags are ignored).
#
#   scripts/run_torchtitan.sh --model Qwen/Qwen3-14B --batch-size 1 --seq-length 2048 \
#       --bench_step 25 --warmup_step 5 --activation_checkpointing \
#       --metrics_output runs/torchtitan-mb1-seq2048/metrics.json
#
# Env: NGPUS_PER_NODE, TORCHTITAN_PYTHON (default ./.venv-torchtitan/bin/python).
# Needs the Qwen3-14B tokenizer assets (see scripts/setup_torchtitan_env.sh).
set -uo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

MODEL="Qwen/Qwen3-14B"; BATCH=1; SEQ=2048; GAS=1
BENCH_STEP=25; WARMUP=5; MEAS_STEPS=20; AC=0; METRICS_OUTPUT=""
NGPUS_PER_NODE="${NGPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}"
PY="${TORCHTITAN_PYTHON:-$repo_root/.venv-torchtitan/bin/python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model|--model_name|--model-name) MODEL="$2"; shift 2 ;;
    --batch_size|--batch-size) BATCH="$2"; shift 2 ;;
    --seq_length|--seq-length) SEQ="$2"; shift 2 ;;
    --gradient_accumulation_steps|--gradient-accumulation-steps) GAS="$2"; shift 2 ;;
    --bench_step|--bench-step) BENCH_STEP="$2"; shift 2 ;;
    --warmup_step|--warmup-step) WARMUP="$2"; shift 2 ;;
    --measured-steps|--measured_steps) MEAS_STEPS="$2"; shift 2 ;;
    --activation_checkpointing|--activation-checkpointing) AC=1; shift ;;
    --metrics_output|--metrics-output) METRICS_OUTPUT="$2"; shift 2 ;;
    *) if [[ $# -gt 1 && ! "$2" =~ ^-- ]]; then shift 2; else shift; fi ;;
  esac
done

[[ -n "$METRICS_OUTPUT" ]] || { echo "ERROR: --metrics_output required" >&2; exit 2; }
[[ -x "$PY" ]] || { echo "ERROR: torchtitan python not found at $PY (run scripts/setup_torchtitan_env.sh)" >&2; exit 2; }

# metrics.json path may be relative to repo_root; make it absolute (we cd into the submodule).
case "$METRICS_OUTPUT" in /*) : ;; *) METRICS_OUTPUT="$repo_root/$METRICS_OUTPUT" ;; esac
results_dir="$(dirname "$METRICS_OUTPUT")"; mkdir -p "$results_dir"
fw_log="$results_dir/framework.log"
mem_log="$results_dir/gpu_mem_mib.log"

export NCCL_DEBUG=WARN
export PYTORCH_ALLOC_CONF="expandable_segments:True"

global_batch=$(( BATCH * NGPUS_PER_NODE * GAS ))
# torchtitan's qwen3_14b config already defaults to FullAC (full activation
# checkpointing), matching the matrix's --activation_checkpointing, so we don't
# pass an AC override (this commit selects AC via a `activation-checkpoint:<mode>`
# variant, not a --activation-checkpoint.mode flag).
# shellcheck disable=SC2207
EXTRA=( $(grep -vE '^\s*(#|$)' configs/torchtitan/qwen3_14b.flags 2>/dev/null) )

echo "[run_torchtitan] model=$MODEL mb=$BATCH seq=$SEQ gbs=$global_batch steps=$BENCH_STEP ac=full(default) nproc=$NGPUS_PER_NODE"

( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null; sleep 0.2; done ) > "$mem_log" &
sampler=$!
trap 'kill "$sampler" 2>/dev/null' EXIT

# torchtitan resolves dataset/tokenizer paths relative to its own root.
cd "$repo_root/third_party/torchtitan"
set +e
"$PY" -m torch.distributed.run --standalone --nproc_per_node="$NGPUS_PER_NODE" \
  -m torchtitan.train --module qwen3 --config qwen3_14b \
  --training.steps "$BENCH_STEP" --training.seq_len "$SEQ" \
  --training.local_batch_size "$BATCH" --training.global_batch_size "$global_batch" \
  --parallelism.data_parallel_shard_degree "$NGPUS_PER_NODE" \
  --metrics.log_freq 1 \
  "${EXTRA[@]}" 2>&1 | tee "$fw_log"
rc=${PIPESTATUS[0]}
cd "$repo_root"
# NOTE: stay under `set +e` here. `wait` on the SIGTERM'd sampler returns non-zero,
# which (with set -e) would abort BEFORE emit and lose metrics.json on a SUCCESSFUL run.
kill "$sampler" 2>/dev/null || true; wait "$sampler" 2>/dev/null || true; trap - EXIT

"$PY" scripts/emit_matrix_metrics.py --framework torchtitan --log "$fw_log" \
  --metrics-output "$METRICS_OUTPUT" --model "$MODEL" --batch-size "$BATCH" --seq-length "$SEQ" \
  --gradient-accumulation-steps "$GAS" --num-processes "$NGPUS_PER_NODE" \
  --warmup-step "$WARMUP" --measured-steps "$MEAS_STEPS" --gpu-mem-log "$mem_log" --return-code "$rc" \
  $([[ "$AC" == "1" ]] && echo --activation-checkpointing)

exit "$rc"
