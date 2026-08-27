#!/usr/bin/env bash
set -euo pipefail

BASELINE_SHA="715965e027894a2e72ac2e27f2daed2c599e99f0"
AGENT_SHA="1b506f73ffd5cf5c5c938d1e7864343b280e0e09"

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_qwen_deepcompile_cell.sh --variant baseline|agent [options]

Runs one exact 1-node x 8-GPU Qwen/Qwen3-14B synthetic-token DeepCompile cell.
The default and first supported row is the known-success mb1/seq1024 cell.

Options:
  --variant baseline|agent  Required. Selects DeepSpeed 715965e or 1b506f.
  --results-dir DIR         Required; must not already exist.
  --batch-size N            Default: 1.
  --seq-length N            Default: 1024.
  --main-process-port N     Default: 29541.
  --agent-max-iterations N  Default: 3.
  --agent-max-retries-per-iteration N  Default: 1.
  --agent-timeout-sec N     Default: 300.
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
harness_root="$(cd "$repo_root/.." && pwd)"
workspace_root="$(cd "$harness_root/../.." && pwd)"

variant=""
results_dir=""
batch_size="1"
seq_length="1024"
main_process_port="29541"
agent_max_iterations="3"
agent_max_retries_per_iteration="1"
agent_timeout_sec="300"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant) variant="$2"; shift 2 ;;
    --results-dir) results_dir="$2"; shift 2 ;;
    --batch-size|--batch_size) batch_size="$2"; shift 2 ;;
    --seq-length|--seq_length) seq_length="$2"; shift 2 ;;
    --main-process-port|--main_process_port) main_process_port="$2"; shift 2 ;;
    --agent-max-iterations|--agent_max_iterations) agent_max_iterations="$2"; shift 2 ;;
    --agent-max-retries-per-iteration|--agent_max_retries_per_iteration)
      agent_max_retries_per_iteration="$2"; shift 2 ;;
    --agent-timeout-sec|--agent_timeout_sec) agent_timeout_sec="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$results_dir" ]]; then
  echo "--results-dir is required" >&2
  exit 2
fi
if [[ -e "$results_dir" ]]; then
  echo "refusing to overwrite existing results path: $results_dir" >&2
  exit 2
fi

case "$variant" in
  baseline)
    deepspeed_root="$harness_root/deepspeed-baseline"
    expected_sha="$BASELINE_SHA"
    optimizer_args=()
    ;;
  agent)
    deepspeed_root="$harness_root/deepspeed-agent"
    expected_sha="$AGENT_SHA"
    optimizer_args=(
      --zero3-tuning-strategy agent
      --agent-backend codex
      --agent-max-iterations "$agent_max_iterations"
      --agent-max-retries-per-iteration "$agent_max_retries_per_iteration"
      --agent-timeout-sec "$agent_timeout_sec"
    )
    ;;
  *)
    echo "--variant must be baseline or agent" >&2
    exit 2
    ;;
esac

mkdir -p "$results_dir"
results_dir="$(cd "$results_dir" && pwd)"

export PYTHONPATH="$deepspeed_root${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$workspace_root/cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$workspace_root/cache/xdg}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$workspace_root/cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$workspace_root/cache/triton}"
export CODEX_HOME="${CODEX_HOME:-$workspace_root/tools/codex-home}"
export CODEX_BIN="${CODEX_BIN:-$workspace_root/tools/codex-cli/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex}"
export DEEPCOMPILE_AGENT_ARTIFACT_ROOT="$results_dir/agent-artifacts"

mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

python "$repo_root/scripts/verify_qwen_repro_environment.py" \
  --expected-deepspeed-sha "$expected_sha" \
  --expected-gpus 8 | tee "$results_dir/environment.json"

cmd=(
  bash "$repo_root/run.sh"
    --model Qwen/Qwen3-14B
    --backend deepspeed
    --zero-stage 3
    --batch-size "$batch_size"
    --seq-length "$seq_length"
    --gradient-accumulation-steps 1
    --dataset_name synthetic
    --dataset_samples 8192
    --dataset_percentage 1.0
    --seed 42
    --bench_step 30
    --warmup_step 10
    --log_interval 1
    --metrics_output "$results_dir/metrics.json"
    --compile
    --deepcompile
    --passes z3
    "${optimizer_args[@]}"
    --learning_rate 1e-4
)

printf '%q ' "${cmd[@]}" > "$results_dir/command.txt"
printf '\n' >> "$results_dir/command.txt"
printf '%s\n' "$expected_sha" > "$results_dir/expected-deepspeed-sha.txt"

cd "$repo_root"
set +e
NUM_NODES=1 NGPUS_PER_NODE=8 MAIN_PROCESS_PORT="$main_process_port" \
  "${cmd[@]}" >"$results_dir/train.log" 2>&1
rc=$?
set -e

printf '{"return_code": %s}\n' "$rc" > "$results_dir/runner-status.json"
[[ -f configs/config.yaml ]] && cp configs/config.yaml "$results_dir/accelerate_config.yaml"
[[ -f configs/ds_config.json ]] && cp configs/ds_config.json "$results_dir/ds_config.json"
exit "$rc"
