#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_diagnostic.sh [options] [-- extra verify_loss.py options]

Options:
  --name NAME                         Run name used in the output directory.
  --results-root DIR                  Default: diagnostic_runs.
  --results-dir DIR                   Exact output directory.
  --model MODEL                       Default: openlm-research/open_llama_7b_v2.
  --backend BACKEND                   Default: deepspeed.
  --zero-stage N                      Default: 3.
  --batch-size N                      Default: 1.
  --seq-length N                      Default: 512.
  --gradient-accumulation-steps N     Default: 1.
  --dataset-percentage FLOAT          Default: 1.0.
  --dataset-name NAME                 Default: synthetic.
  --dataset-samples N                 Default: 1024 for synthetic.
  --bench-step N                      Default: 8.
  --warmup-step N                     Default: 2.
  --log-interval N                    Default: 1.
  --profile                          Enable a short PyTorch profiler window.
  --profile-wait-steps N              Default: 0.
  --profile-warmup-steps N            Default: 2.
  --profile-active-steps N            Default: 2.
  --fp16                              Generate an fp16 DeepSpeed config.
  --no-activation-checkpointing       Do not pass --activation_checkpointing.
  --zero-stage3-offload-param-device DEVICE
                                      Enable ZeRO-3 parameter offload; e.g. cpu.
  --zero-stage3-offload-param-pin-memory BOOL
                                      Default: true when parameter offload is enabled.
  --no-zero-stage3-offload-param-pin-memory
                                      Disable pinned host memory for parameter offload.
  --chunked-causal-lm-loss-tokens N   Compute causal-LM loss in token chunks.
  --chunked-causal-lm-loss-empty-cache
                                      Empty CUDA cache before chunked loss upcasts.
  --chunked-causal-lm-loss-device DEV Device for chunked CE: cuda or cpu.
  --next-command CMD                  Suggested follow-up command for report.md.
  -h, --help                          Show this help.

The script writes command.txt, environment.json, train.log, metrics.json,
profile_summary.json when profiling is enabled, summary.json, and report.md.
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

name="z3-smoke"
results_root="diagnostic_runs"
results_dir=""
model="openlm-research/open_llama_7b_v2"
backend="deepspeed"
zero_stage="3"
batch_size="1"
seq_length="512"
gradient_accumulation_steps="1"
dataset_percentage="1.0"
dataset_name="synthetic"
dataset_samples="1024"
bench_step="8"
warmup_step="2"
log_interval="1"
profile=0
profile_wait_steps="0"
profile_warmup_steps="2"
profile_active_steps="2"
fp16=0
activation_checkpointing=1
zero_stage3_offload_param_device=""
zero_stage3_offload_param_pin_memory="true"
chunked_causal_lm_loss_tokens="0"
chunked_causal_lm_loss_empty_cache="0"
chunked_causal_lm_loss_device="cuda"
next_command=""
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) name="$2"; shift 2 ;;
    --results-root) results_root="$2"; shift 2 ;;
    --results-dir) results_dir="$2"; shift 2 ;;
    --model|--model-name|--model_name) model="$2"; shift 2 ;;
    --backend) backend="$2"; shift 2 ;;
    --zero-stage|--zero_stage) zero_stage="$2"; shift 2 ;;
    --batch-size|--batch_size) batch_size="$2"; shift 2 ;;
    --seq-length|--seq_length) seq_length="$2"; shift 2 ;;
    --gradient-accumulation-steps|--gradient_accumulation_steps) gradient_accumulation_steps="$2"; shift 2 ;;
    --dataset-percentage|--dataset_percentage) dataset_percentage="$2"; shift 2 ;;
    --dataset-name|--dataset_name) dataset_name="$2"; shift 2 ;;
    --dataset-samples|--dataset_samples) dataset_samples="$2"; shift 2 ;;
    --bench-step|--bench_step) bench_step="$2"; shift 2 ;;
    --warmup-step|--warmup_step) warmup_step="$2"; shift 2 ;;
    --log-interval|--log_interval) log_interval="$2"; shift 2 ;;
    --profile) profile=1; shift ;;
    --profile-wait-steps|--profile_wait_steps) profile_wait_steps="$2"; shift 2 ;;
    --profile-warmup-steps|--profile_warmup_steps) profile_warmup_steps="$2"; shift 2 ;;
    --profile-active-steps|--profile_active_steps) profile_active_steps="$2"; shift 2 ;;
    --fp16) fp16=1; shift ;;
    --bf16) fp16=0; shift ;;
    --no-activation-checkpointing) activation_checkpointing=0; shift ;;
    --zero-stage3-offload-param-device|--zero_stage3_offload_param_device)
      zero_stage3_offload_param_device="$2"; shift 2 ;;
    --zero-stage3-offload-param-pin-memory|--zero_stage3_offload_param_pin_memory)
      zero_stage3_offload_param_pin_memory="$2"; shift 2 ;;
    --no-zero-stage3-offload-param-pin-memory|--no_zero_stage3_offload_param_pin_memory)
      zero_stage3_offload_param_pin_memory="false"; shift ;;
    --chunked-causal-lm-loss-tokens|--chunked_causal_lm_loss_tokens)
      chunked_causal_lm_loss_tokens="$2"; shift 2 ;;
    --chunked-causal-lm-loss-empty-cache|--chunked_causal_lm_loss_empty_cache)
      chunked_causal_lm_loss_empty_cache="1"; shift ;;
    --chunked-causal-lm-loss-device|--chunked_causal_lm_loss_device)
      chunked_causal_lm_loss_device="$2"; shift 2 ;;
    --next-command) next_command="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; extra_args+=("$@"); break ;;
    *) extra_args+=("$1"); shift ;;
  esac
done

if [[ -z "$results_dir" ]]; then
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  results_dir="${results_root}/${timestamp}-${name}"
fi

mkdir -p "$results_dir"
metrics_file="${results_dir}/metrics.json"
profile_summary_file="${results_dir}/profile_summary.json"
environment_file="${results_dir}/environment.json"
command_file="${results_dir}/command.txt"
train_log="${results_dir}/train.log"

python - "$environment_file" <<'PY'
import importlib.metadata as md
import json
import platform
import socket
import subprocess
import sys

out = sys.argv[1]
payload = {
    "hostname": socket.gethostname(),
    "python": platform.python_version(),
    "platform": platform.platform(),
}

for package in ("torch", "deepspeed", "accelerate", "transformers"):
    try:
        payload[package] = md.version(package)
    except md.PackageNotFoundError:
        payload[package] = "not installed"

try:
    import torch
    payload["cuda_available"] = bool(torch.cuda.is_available())
    payload["cuda_device_count"] = int(torch.cuda.device_count())
    payload["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
except Exception as exc:
    payload["cuda_error"] = f"{type(exc).__name__}: {exc}"

try:
    payload["nvidia_smi"] = subprocess.check_output(
        ["nvidia-smi"], text=True, stderr=subprocess.STDOUT, timeout=20
    )
except Exception as exc:
    payload["nvidia_smi"] = f"{type(exc).__name__}: {exc}"

with open(out, "w") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")
PY

run_args=(
  --backend "$backend"
  --zero_stage "$zero_stage"
  --model "$model"
  --batch_size "$batch_size"
  --seq_length "$seq_length"
  --gradient_accumulation_steps "$gradient_accumulation_steps"
  --dataset_percentage "$dataset_percentage"
  --dataset_name "$dataset_name"
  --dataset_samples "$dataset_samples"
  --bench_step "$bench_step"
  --warmup_step "$warmup_step"
  --log_interval "$log_interval"
  --metrics_output "$metrics_file"
  --profile_summary_output "$profile_summary_file"
)

if [[ "$activation_checkpointing" == "1" ]]; then
  run_args+=(--activation_checkpointing)
fi
if [[ "$fp16" == "1" ]]; then
  run_args+=(--fp16)
fi
if [[ -n "$zero_stage3_offload_param_device" ]]; then
  run_args+=(
    --zero_stage3_offload_param_device "$zero_stage3_offload_param_device"
    --zero_stage3_offload_param_pin_memory "$zero_stage3_offload_param_pin_memory"
  )
fi
if [[ "$chunked_causal_lm_loss_tokens" != "0" ]]; then
  run_args+=(--chunked_causal_lm_loss_tokens "$chunked_causal_lm_loss_tokens")
fi
if [[ "$chunked_causal_lm_loss_empty_cache" == "1" ]]; then
  run_args+=(--chunked_causal_lm_loss_empty_cache)
fi
if [[ "$chunked_causal_lm_loss_device" != "cuda" ]]; then
  run_args+=(--chunked_causal_lm_loss_device "$chunked_causal_lm_loss_device")
fi
if [[ "$profile" == "1" ]]; then
  run_args+=(
    --profile
    --profile_dir "${results_dir}/profile"
    --profile_wait_steps "$profile_wait_steps"
    --profile_warmup_steps "$profile_warmup_steps"
    --profile_active_steps "$profile_active_steps"
  )
fi
run_args+=("${extra_args[@]}")

cmd=(bash ./run.sh "${run_args[@]}")
printf '%q ' "${cmd[@]}" > "$command_file"
printf '\n' >> "$command_file"

set +e
bash -o pipefail -c '"$@" 2>&1 | tee "$0"' "$train_log" "${cmd[@]}"
status=$?
set -e

deepspeed_config_file="${results_dir}/ds_config.json"
accelerate_config_file="${results_dir}/accelerate_config.yaml"
if [[ -f configs/ds_config.json ]]; then
  cp configs/ds_config.json "$deepspeed_config_file"
else
  deepspeed_config_file=""
fi
if [[ -f configs/config.yaml ]]; then
  cp configs/config.yaml "$accelerate_config_file"
else
  accelerate_config_file=""
fi

python scripts/generate_diagnostic_report.py \
  --results-dir "$results_dir" \
  --exit-code "$status" \
  --command-file "$command_file" \
  --log-file "$train_log" \
  --metrics-file "$metrics_file" \
  --profile-summary-file "$profile_summary_file" \
  --environment-file "$environment_file" \
  --deepspeed-config-file "$deepspeed_config_file" \
  --accelerate-config-file "$accelerate_config_file" \
  --next-command "${next_command:-${cmd[*]}}"

echo "Report: ${results_dir}/report.md"
exit "$status"
