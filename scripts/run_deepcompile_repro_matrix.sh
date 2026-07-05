#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_deepcompile_repro_matrix.sh [options]

Runs the reproduction matrix:
  framework in {fsdp, deepspeed, deepcompile, megatron, torchtitan}
  batch size in {1,2,4}
  sequence length in {1024,2048,4096}

  fsdp/deepspeed/deepcompile run via run.sh (accelerate + verify_loss.py).
  megatron/torchtitan run via scripts/run_{megatron,torchtitan}.sh (own venvs,
  Qwen3-14B, FSDP-equivalent sharding) and emit the same metrics.json schema.

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
  --ac on|off            Activation checkpointing for fsdp/deepspeed/torchtitan/megatron.
                         Default: on. (deepcompile always manages its own activations.)
  --dataset-samples N    Default: 8192.
  --base-port N          Default: 29531.
  --cell-timeout-s N     Default: 3600. Use 0 to disable.
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
ac_mode="on"          # on|off : activation checkpointing for fsdp/deepspeed/torchtitan/megatron
dataset_samples="8192"
base_port="29531"
cell_timeout_s="3600"

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
    --ac) ac_mode="$2"; shift 2 ;;
    --dataset-samples|--dataset_samples) dataset_samples="$2"; shift 2 ;;
    --base-port|--base_port) base_port="$2"; shift 2 ;;
    --cell-timeout-s|--cell_timeout_s) cell_timeout_s="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$nproc" ]]; then
  nproc="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
# Model-derived slug so different models don't collide in one qwen3-14b dir:
# Qwen/Qwen3-30B-A3B -> qwen3-30b-a3b.
model_slug="$(basename "$model" | tr 'A-Z' 'a-z')"
# Auto-detect the GPU so the run dir is labeled accurately (h100 vs h200 vs ...),
# instead of a hardcoded tag. Override with HW_TAG=... if detection is wrong.
if [[ -z "${HW_TAG:-}" ]]; then
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  case "$gpu_name" in
    *H200*) HW_TAG=h200 ;; *H100*) HW_TAG=h100 ;; *B200*) HW_TAG=b200 ;;
    *A100*) HW_TAG=a100 ;; *) HW_TAG="$(echo "${gpu_name:-gpu}" | tr ' A-Z' '-a-z' | tr -cd 'a-z0-9-')" ;;
  esac
fi
run_root="${results_root}/${timestamp}-${model_slug}-${nproc}x${HW_TAG}-repro"
mkdir -p "$run_root"

{
  echo "run_root=$run_root"
  echo "model=$model"
  echo "nproc=$nproc"
  echo "frameworks=$frameworks"
  echo "mbs=$mbs"
  echo "seqs=$seqs"
  echo "measured_steps=$measured_steps"
  echo "ac_mode=$ac_mode"
  echo "eager_warmup=$eager_warmup"
  echo "deepcompile_warmup=$deepcompile_warmup"
  echo "dataset_samples=$dataset_samples"
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
      # AC knob for fsdp/deepspeed/torchtitan/megatron. The launchers each turn
      # --activation_checkpointing into their native AC (HF gradient ckpt / Megatron
      # recompute / torchtitan activation-checkpoint:full); its absence -> AC off.
      if [[ "$ac_mode" == "off" ]]; then extra=(); else extra=(--activation_checkpointing); fi
      if [[ "$framework" == "deepcompile" ]]; then
        # deepcompile always manages its own selective activation persistence (the
        # harness skips HF gradient checkpointing for it) -> unaffected by --ac.
        backend="deepspeed"
        warmup="$deepcompile_warmup"
        extra=(--compile --deepcompile --passes z3)
      fi
      # torchtitan compiles too (inductor + FlexAttention autotune), so the first
      # several steps are compile, not steady state -> use the compile-sized warmup.
      if [[ "$framework" == "torchtitan" ]]; then
        warmup="$deepcompile_warmup"
      fi
      bench_step=$((warmup + measured_steps))
      port=$((base_port + cell_index))
      cell_id="${framework}-mb${mb}-seq${seq}"
      results_dir="${run_root}/${cell_id}"
      mkdir -p "$results_dir"

      echo ">>> [$cell_index] $cell_id warmup=$warmup measured=$measured_steps port=$port"
      # fsdp/deepspeed/deepcompile run the colleague's run.sh, which calls bare
      # `accelerate`/`python` (PATH-resolved). Pin them to .venv-ds so the run doesn't
      # depend on which env happens to be active. megatron/torchtitan launchers manage
      # their own venvs, so leave PATH alone for them (ds_path empty).
      ds_path=""
      case "$framework" in
        megatron|torchtitan)
          # External frameworks: run via their own launcher (own venv), same knobs.
          # They write metrics.json in the same schema (scripts/emit_matrix_metrics.py).
          cmd=(
            bash "scripts/run_${framework}.sh"
              --model "$model"
              --batch-size "$mb"
              --seq-length "$seq"
              --gradient-accumulation-steps 1
              --bench_step "$bench_step"
              --warmup_step "$warmup"
              --measured-steps "$measured_steps"
              --metrics_output "$results_dir/metrics.json"
              "${extra[@]}"
          )
          ;;
        *)
          ds_path="${DS_VENV:-$PWD/.venv-ds}/bin"
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
              --bench_step "$bench_step"
              --warmup_step "$warmup"
              --log_interval 1
              --metrics_output "$results_dir/metrics.json"
              "${extra[@]}"
              --learning_rate 1e-4
          )
          ;;
      esac
      printf '%q ' "${cmd[@]}" > "$results_dir/command.txt"
      printf '\n' >> "$results_dir/command.txt"

      if [[ "$cell_timeout_s" != "0" ]] && command -v timeout >/dev/null 2>&1; then
        PATH="${ds_path:+$ds_path:}$PATH" MAIN_PROCESS_PORT="$port" NGPUS_PER_NODE="$nproc" timeout "$cell_timeout_s" "${cmd[@]}" >"$results_dir/train.log" 2>&1
      else
        PATH="${ds_path:+$ds_path:}$PATH" MAIN_PROCESS_PORT="$port" NGPUS_PER_NODE="$nproc" "${cmd[@]}" >"$results_dir/train.log" 2>&1
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

python scripts/summarize_repro_matrix.py --runs-root "$run_root" --out-dir "$run_root" || \
  echo "WARNING: matrix summary failed" >&2

echo "RUN_ROOT=$run_root"
echo "FAIL_COUNT=$fail_count"
exit 0
