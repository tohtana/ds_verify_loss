#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_deepcompile_repro_matrix.sh [options]

Runs the reproduction matrix:
  framework in {fsdp, deepspeed, deepcompile}
  batch size in {1,2,4}
  sequence length in {1024,2048,4096}

Options:
  --results-root DIR     Default: repro_matrix_runs.
  --model MODEL          Default: Qwen/Qwen3-14B.
  --nproc N              Default: NGPUS_PER_NODE or detected GPU count.
  --mbs "1 2 4"          Micro-batches per GPU.
  --seqs "1024 2048 4096"
  --frameworks "fsdp deepspeed deepcompile"
  --measured-steps N     Default: 20.
  --eager-warmup N       Default: 5.
  --deepcompile-warmup N Default: 10.
  --dataset-samples N    Default: 8192.
  --seed N               Default: 42.
  --zero3-tuning-strategy baseline|agent
                         Default: baseline. Applied only to DeepCompile rows.
  --agent-backend codex  Required for the agent tuning strategy.
  --agent-max-iterations N             Default: 3.
  --agent-max-retries-per-iteration N  Default: 1.
  --agent-timeout-sec N                Default: 300.
  --base-port N          Default: 29531.
  --cell-timeout-s N     Default: 3600. Use 0 to disable.
  --dry-run              Materialize and print commands without launching GPUs.
  -h, --help             Show this help.
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

results_root="repro_matrix_runs"
model="Qwen/Qwen3-14B"
nproc="${NGPUS_PER_NODE:-}"
mbs="1 2 4"
seqs="1024 2048 4096"
frameworks="fsdp deepspeed deepcompile"
measured_steps="20"
eager_warmup="5"
deepcompile_warmup="10"
dataset_samples="8192"
seed="42"
zero3_tuning_strategy="baseline"
agent_backend=""
agent_max_iterations="3"
agent_max_retries_per_iteration="1"
agent_timeout_sec="300"
base_port="29531"
cell_timeout_s="3600"
dry_run="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --results-root) results_root="$2"; shift 2 ;;
    --model|--model-name|--model_name) model="$2"; shift 2 ;;
    --nproc) nproc="$2"; shift 2 ;;
    --mbs) mbs="$2"; shift 2 ;;
    --seqs) seqs="$2"; shift 2 ;;
    --frameworks) frameworks="$2"; shift 2 ;;
    --measured-steps|--measured_steps) measured_steps="$2"; shift 2 ;;
    --eager-warmup|--eager_warmup) eager_warmup="$2"; shift 2 ;;
    --deepcompile-warmup|--deepcompile_warmup) deepcompile_warmup="$2"; shift 2 ;;
    --dataset-samples|--dataset_samples) dataset_samples="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --zero3-tuning-strategy|--zero3_tuning_strategy) zero3_tuning_strategy="$2"; shift 2 ;;
    --agent-backend|--agent_backend) agent_backend="$2"; shift 2 ;;
    --agent-max-iterations|--agent_max_iterations) agent_max_iterations="$2"; shift 2 ;;
    --agent-max-retries-per-iteration|--agent_max_retries_per_iteration)
      agent_max_retries_per_iteration="$2"; shift 2 ;;
    --agent-timeout-sec|--agent_timeout_sec) agent_timeout_sec="$2"; shift 2 ;;
    --base-port|--base_port) base_port="$2"; shift 2 ;;
    --cell-timeout-s|--cell_timeout_s) cell_timeout_s="$2"; shift 2 ;;
    --dry-run) dry_run="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "$zero3_tuning_strategy" != "baseline" && "$zero3_tuning_strategy" != "agent" ]]; then
  echo "--zero3-tuning-strategy must be baseline or agent" >&2
  exit 2
fi
if [[ "$zero3_tuning_strategy" == "agent" && -z "$agent_backend" ]]; then
  echo "--agent-backend is required for --zero3-tuning-strategy agent" >&2
  exit 2
fi

if [[ -z "$nproc" ]]; then
  nproc="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_root="${results_root}/${timestamp}-qwen3-14b-8xh100-repro"
mkdir -p "$run_root"

{
  echo "run_root=$run_root"
  echo "model=$model"
  echo "nproc=$nproc"
  echo "frameworks=$frameworks"
  echo "mbs=$mbs"
  echo "seqs=$seqs"
  echo "measured_steps=$measured_steps"
  echo "eager_warmup=$eager_warmup"
  echo "deepcompile_warmup=$deepcompile_warmup"
  echo "dataset_samples=$dataset_samples"
  echo "seed=$seed"
  echo "zero3_tuning_strategy=$zero3_tuning_strategy"
  echo "agent_backend=${agent_backend:-none}"
  echo "agent_max_iterations=$agent_max_iterations"
  echo "agent_max_retries_per_iteration=$agent_max_retries_per_iteration"
  echo "agent_timeout_sec=$agent_timeout_sec"
  echo "cell_timeout_s=$cell_timeout_s"
  git rev-parse HEAD 2>/dev/null | sed 's/^/source_commit=/'
} | tee "$run_root/matrix-config.txt"

cell_index=0
fail_count=0

for framework in $frameworks; do
  for mb in $mbs; do
    for seq in $seqs; do
      cell_index=$((cell_index + 1))
      warmup="$eager_warmup"
      backend="$framework"
      extra=(--activation_checkpointing)
      if [[ "$framework" == "deepcompile" ]]; then
        backend="deepspeed"
        warmup="$deepcompile_warmup"
        extra=(--compile --deepcompile --passes z3)
        if [[ "$zero3_tuning_strategy" == "agent" ]]; then
          extra+=(
            --zero3-tuning-strategy agent
            --agent-backend "$agent_backend"
            --agent-max-iterations "$agent_max_iterations"
            --agent-max-retries-per-iteration "$agent_max_retries_per_iteration"
            --agent-timeout-sec "$agent_timeout_sec"
          )
        fi
      fi
      bench_step=$((warmup + measured_steps))
      port=$((base_port + cell_index))
      cell_id="${framework}-mb${mb}-seq${seq}"
      results_dir="${run_root}/${cell_id}"
      mkdir -p "$results_dir"

      echo ">>> [$cell_index] $cell_id warmup=$warmup measured=$measured_steps port=$port"
      cmd=(
        bash ./run.sh
          --model "$model"
          --backend "$backend"
          --zero-stage 3
          --batch-size "$mb"
          --seq-length "$seq"
          --gradient-accumulation-steps 1
          --dataset_name synthetic
          --dataset_samples "$dataset_samples"
          --dataset_percentage 1.0
          --seed "$seed"
          --bench_step "$bench_step"
          --warmup_step "$warmup"
          --log_interval 1
          --metrics_output "$results_dir/metrics.json"
          "${extra[@]}"
          --learning_rate 1e-4
      )
      printf '%q ' "${cmd[@]}" > "$results_dir/command.txt"
      printf '\n' >> "$results_dir/command.txt"

      if [[ "$dry_run" == "true" ]]; then
        cat "$results_dir/command.txt"
        printf '{"return_code": 0, "dry_run": true}\n' > "$results_dir/runner-status.json"
        continue
      fi

      if [[ "$cell_timeout_s" != "0" ]] && command -v timeout >/dev/null 2>&1; then
        MAIN_PROCESS_PORT="$port" NGPUS_PER_NODE="$nproc" \
          DEEPCOMPILE_AGENT_ARTIFACT_ROOT="$results_dir/agent-artifacts" \
          timeout "$cell_timeout_s" "${cmd[@]}" >"$results_dir/train.log" 2>&1
      else
        MAIN_PROCESS_PORT="$port" NGPUS_PER_NODE="$nproc" \
          DEEPCOMPILE_AGENT_ARTIFACT_ROOT="$results_dir/agent-artifacts" \
          "${cmd[@]}" >"$results_dir/train.log" 2>&1
      fi
      rc=$?
      printf '{"return_code": %s}\n' "$rc" > "$results_dir/runner-status.json"
      [[ -f configs/config.yaml ]] && cp configs/config.yaml "$results_dir/accelerate_config.yaml"
      [[ -f configs/ds_config.json ]] && cp configs/ds_config.json "$results_dir/ds_config.json"
      if [[ "$rc" -ne 0 ]]; then
        echo "cell $cell_id exited rc=$rc; continuing" | tee -a "$run_root/matrix-failures.txt"
        fail_count=$((fail_count + 1))
      fi
      sleep 5
    done
  done
done

if [[ "$dry_run" != "true" ]]; then
  python scripts/summarize_repro_matrix.py --runs-root "$run_root" --out-dir "$run_root" || \
    echo "WARNING: matrix summary failed" >&2
fi

echo "RUN_ROOT=$run_root"
echo "FAIL_COUNT=$fail_count"
exit 0
