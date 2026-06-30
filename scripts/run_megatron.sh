#!/usr/bin/env bash
# Launch a single Megatron-LM (Qwen3-14B, Megatron-FSDP) benchmark cell and write
# a metrics.json in the ds_verify_loss schema. Accepts the same flags the matrix
# runner passes to run.sh (deepspeed-only flags are ignored).
#
#   scripts/run_megatron.sh --model Qwen/Qwen3-14B --batch-size 1 --seq-length 2048 \
#       --bench_step 25 --warmup_step 5 --activation_checkpointing \
#       --metrics_output runs/megatron-mb1-seq2048/metrics.json
#
# Env: NGPUS_PER_NODE, MEGATRON_PYTHON (default ./.venv-megatron/bin/python).
set -uo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

MODEL="Qwen/Qwen3-14B"; BATCH=1; SEQ=2048; GAS=1
BENCH_STEP=25; WARMUP=5; MEAS_STEPS=20; AC=0; METRICS_OUTPUT=""
NGPUS_PER_NODE="${NGPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}"
PY="${MEGATRON_PYTHON:-$repo_root/.venv-megatron/bin/python}"

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
    *)  # ignore unknown flag (+ its value if present) so deepspeed-only args don't break us
        if [[ $# -gt 1 && ! "$2" =~ ^-- ]]; then shift 2; else shift; fi ;;
  esac
done

[[ -n "$METRICS_OUTPUT" ]] || { echo "ERROR: --metrics_output required" >&2; exit 2; }
[[ -x "$PY" ]] || { echo "ERROR: megatron python not found at $PY (run scripts/setup_megatron_env.sh)" >&2; exit 2; }

results_dir="$(dirname "$METRICS_OUTPUT")"; mkdir -p "$results_dir" logs
fw_log="$results_dir/framework.log"
mem_log="$results_dir/gpu_mem_mib.log"

export PYTHONPATH="$repo_root/third_party/Megatron-LM:${PYTHONPATH:-}"
# Put the venv's bin first on PATH so Megatron's dataset-helper Makefile uses the
# venv's python3 / python3-config (3.12 + pybind11), not the system python3 (3.8).
export PATH="$(cd "$(dirname "$PY")" && pwd):$PATH"
# Force ALL CUDA runtime libs to the venv's cu12 set (what torch cu126 was built
# against), so cuDNN's main lib AND its bare-SONAME sub-libs (libcudnn_graph.so.9,
# libcudnn_engines_*.so.9, ...) plus libcudart all come from ONE self-consistent cu12
# install. This box scatters cuDNN 9.5/9.7/9.13 across cuda-12.6/12.8/12.9/13.0; without
# this, the main lib loads from one toolkit while a sub-lib dlopen resolves (via
# LD_LIBRARY_PATH / ldconfig) to cuda-13.0's cuDNN, which drags in libcudart.so.13 and
# makes TE abort: "Multiple libcudart libraries found: libcudart.so.12 and .so.13".
sp="$repo_root/.venv-megatron/lib/python3.12/site-packages/nvidia"
venv_cuda_libs="$(find "$sp" -maxdepth 2 -name lib -type d 2>/dev/null | paste -sd:)"
export CUDNN_HOME="$sp/cudnn"           # TE searches CUDNN_HOME before CUDA_HOME -> venv cuDNN
export LD_LIBRARY_PATH="${venv_cuda_libs}:${LD_LIBRARY_PATH:-}"   # venv cu12 libs win every lookup
export CUDA_HOME=/usr/local/cuda-12.6   # headers / nvcc / nvrtc+curand search (all cu12)
export CUDA_PATH=/usr/local/cuda-12.6
export NCCL_DEBUG=WARN
# NOTE: do NOT set CUDA_DEVICE_MAX_CONNECTIONS=1 here — Megatron-FSDP asserts it must
# be >1 or unset. (It's a tensor-parallel comm-overlap setting; we run TP=1.)
unset CUDA_DEVICE_MAX_CONNECTIONS || true

global_batch=$(( BATCH * NGPUS_PER_NODE * GAS ))
ac_args=()
if [[ "$AC" == "1" ]]; then
  ac_args=(--recompute-granularity full --recompute-method uniform --recompute-num-layers 1)
fi
# shellcheck disable=SC2207
MA=( $(grep -vE '^\s*(#|$)' configs/megatron/qwen3_14b.args) )

echo "[run_megatron] model=$MODEL mb=$BATCH seq=$SEQ gbs=$global_batch iters=$BENCH_STEP ac=$AC nproc=$NGPUS_PER_NODE"

# out-of-process peak GPU memory sampler (peak across GPUs over the whole run)
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null; sleep 0.2; done ) > "$mem_log" &
sampler=$!
trap 'kill "$sampler" 2>/dev/null' EXIT

set +e
"$PY" -m torch.distributed.run --standalone --nproc_per_node="$NGPUS_PER_NODE" \
  third_party/Megatron-LM/pretrain_gpt.py "${MA[@]}" \
  --micro-batch-size "$BATCH" --global-batch-size "$global_batch" \
  --seq-length "$SEQ" --train-iters "$BENCH_STEP" "${ac_args[@]}" 2>&1 | tee "$fw_log"
rc=${PIPESTATUS[0]}
# Stay under `set +e`: `wait` on the SIGTERM'd sampler returns non-zero, which with
# set -e would abort before emit and lose metrics.json on a successful run.
kill "$sampler" 2>/dev/null || true; wait "$sampler" 2>/dev/null || true; trap - EXIT

"$PY" scripts/emit_matrix_metrics.py --framework megatron --log "$fw_log" \
  --metrics-output "$METRICS_OUTPUT" --model "$MODEL" --batch-size "$BATCH" --seq-length "$SEQ" \
  --gradient-accumulation-steps "$GAS" --num-processes "$NGPUS_PER_NODE" \
  --warmup-step "$WARMUP" --measured-steps "$MEAS_STEPS" --gpu-mem-log "$mem_log" --return-code "$rc" \
  $([[ "$AC" == "1" ]] && echo --activation-checkpointing)

exit "$rc"
