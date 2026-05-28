# Diagnostic Optimization Pilot Prep

This branch prepares the Llama DeepSpeed harness for short diagnostic-first
experiments. It is not the full optimization campaign and should not run the
full multi-shape matrix by default.

## Quick ZeRO-3 Smoke

Run a short bf16 ZeRO-3 baseline from a GPU-visible shell:

```bash
NGPUS_PER_NODE=4 MAIN_PROCESS_PORT=29531 \
  scripts/run_diagnostic.sh \
  --name z3-smoke \
  --model openlm-research/open_llama_7b_v2 \
  --zero-stage 3 \
  --bench-step 8 \
  --warmup-step 2 \
  --dataset-name synthetic \
  --dataset-samples 1024
```

The wrapper records the exact command, environment, train log, metrics JSON,
summary JSON, and report under `diagnostic_runs/<timestamp>-z3-smoke/`.

If bf16 is unavailable in the environment, add `--fp16`. The public
`openlm-research/open_llama_7b_v2` model avoids gated Meta Llama access for
first-pass harness smoke tests; switch to `meta-llama/Meta-Llama-3-8B` only
when Hugging Face auth is confirmed. Keep the smoke short until the harness,
model access, and dataset path are known-good.

## Short Profile

Run the same smoke with a small profiler window:

```bash
NGPUS_PER_NODE=4 MAIN_PROCESS_PORT=29532 \
  scripts/run_diagnostic.sh \
  --name z3-profile \
  --profile \
  --profile-warmup-steps 2 \
  --profile-active-steps 2 \
  --model openlm-research/open_llama_7b_v2 \
  --zero-stage 3 \
  --bench-step 8 \
  --warmup-step 2 \
  --dataset-name synthetic \
  --dataset-samples 1024
```

The profile path writes PyTorch profiler traces under the run directory and
adds a coarse profile summary to `profile_summary.json` and `report.md`.

## Anyscale Workspace Pattern

For a CPU-head plus single 4xH100 worker Workspace, push or clone this branch
into a path visible from the worker, such as `/mnt/cluster_storage/ds_verify_loss`.
Then launch the GPU-visible command through Ray:

```bash
python scripts/workspace_gpu_runner.py \
  --num-gpus 4 \
  --cwd /mnt/cluster_storage/ds_verify_loss \
  --timeout-seconds 2400 \
  --summary-output diagnostic_runs/workspace-runner-summary.json \
  --command-log diagnostic_runs/workspace-runner-command.log \
  -- bash -lc 'NGPUS_PER_NODE=4 MAIN_PROCESS_PORT=29531 scripts/run_diagnostic.sh --name z3-smoke --zero-stage 3 --bench-step 8 --warmup-step 2 --dataset-name synthetic --dataset-samples 1024'
```

Use `bash -o pipefail` for any manual command that pipes through `tee`. The
provided wrappers already preserve the underlying training exit code.

## Report Contract

Every diagnostic run should produce:

- `command.txt`: exact training command.
- `environment.json`: Python, package, CUDA, and `nvidia-smi` context.
- `ds_config.json`: generated DeepSpeed config snapshot, including
  `zero_optimization.offload_param` when parameter offload is enabled and
  `zero_optimization.offload_optimizer` when optimizer offload is enabled.
- `accelerate_config.yaml`: generated Accelerate config snapshot.
- `train.log`: captured stdout/stderr.
- `metrics.json`: avg step time, samples/s, tokens/s, CUDA allocated and
  reserved memory, success status, and error summary when available.
- `profile_summary.json`: coarse diagnosis when `--profile` is enabled.
- `report.md`: one-file handoff for the next agent.

Qwen-style causal-LM loss can be computed with
`--chunked-causal-lm-loss-tokens N` when the diagnostic target is the loss
upcast itself. This leaves the model, labels, and shifted cross-entropy
semantics unchanged, but upcasts `[batch, chunk, vocab]` logits at a time
instead of materializing one full fp32 `[batch, seq, vocab]` logits tensor.
When allocator cache pressure is the remaining failure mode, add
`--chunked-causal-lm-loss-empty-cache` to release unallocated cached blocks
before those chunk upcasts.
If the GPU-side chunk allocation itself remains impossible, use
`--chunked-causal-lm-loss-device cpu` to run the fp32 chunked cross entropy on
CPU while preserving the same shifted CE formula.

## Still Missing Before Full Optimization Rounds

- A reproducible Anyscale Job form for final accepted evidence.
- A full matrix runner that consumes `report.md` outputs without masking failed
  subprocesses.
- More precise per-rank profiler trace aggregation for NCCL and optimizer-tail
  attribution.
- A model-access fallback policy for environments without access to gated
  Meta Llama repositories.
- Real dataset smoke coverage after the `datasets`/`huggingface_hub` version
  combination is pinned or updated; synthetic tokens are the default for
  launcher, memory, and report-path validation.

## Preparation Validation Record

Date: 2026-05-27 UTC

Workspace:

- Id: `expwrk_qeeei1bv25rps3hqt5j6raz6kd`
- Name: `ds-verify-loss-diagnostic-pilot-20260527T220438Z`
- Cloud: `rkn-gpu-cloud`
- Image: `anyscale/image/dev-ds-h100-cu128-torch210-efa-te:1`
- Compute config: `default-queue-h100-4x`
- Shape: CPU head plus one 4xH100 worker

Setup commands:

```bash
anyscale workspace_v2 create \
  --name ds-verify-loss-diagnostic-pilot-20260527T220438Z \
  --cloud rkn-gpu-cloud \
  --compute-config default-queue-h100-4x \
  --image-uri anyscale/image/dev-ds-h100-cu128-torch210-efa-te:1 \
  --env NCCL_DEBUG=WARN \
  --tag project=ds_verify_loss \
  --tag task=diagnostic-optimization-pilot
anyscale workspace_v2 start --id expwrk_qeeei1bv25rps3hqt5j6raz6kd
anyscale workspace_v2 wait --id expwrk_qeeei1bv25rps3hqt5j6raz6kd --state RUNNING --timeout-s 3600
anyscale workspace_v2 push --id expwrk_qeeei1bv25rps3hqt5j6raz6kd --local-dir /Users/mtanaka/work/ds_verify_loss --push-git-state
```

Worker setup used `/home/ray/anaconda3/bin/python -m venv --system-site-packages`
at `/mnt/cluster_storage/ds_verify_loss/.venv`, then:

```bash
DS_BUILD_OPS=0 python -m pip install 'deepspeed>=0.16.0' accelerate transformers datasets wandb sentencepiece protobuf tiktoken
```

Validated commands:

```bash
NGPUS_PER_NODE=4 MAIN_PROCESS_PORT=29535 \
  scripts/run_diagnostic.sh \
  --results-dir diagnostic_runs/z3-openllama-synthetic-smoke \
  --name z3-openllama-synthetic-smoke \
  --model openlm-research/open_llama_7b_v2 \
  --zero-stage 3 \
  --bench-step 4 \
  --warmup-step 1 \
  --seq-length 256 \
  --dataset-name synthetic \
  --dataset-samples 1024
```

Smoke result: success on 4 H100s with ZeRO-3 bf16. Metrics from
`diagnostic_runs/z3-openllama-synthetic-smoke/metrics.json`:

- Avg step time: `0.8386181990305582` sec
- Samples/s: `4.769751007817378`
- Tokens/s: `1221.0562580012488`
- `torch.cuda.max_memory_allocated()`: `36057900544` bytes
- Reserved memory: `45864714240` bytes
- Profile summary fields: rank imbalance `low`, allocator pressure `medium`,
  other profiler-only fields `unknown` because profiling was disabled

```bash
NGPUS_PER_NODE=4 MAIN_PROCESS_PORT=29537 \
  scripts/run_diagnostic.sh \
  --results-dir diagnostic_runs/z3-openllama-synthetic-profile \
  --name z3-openllama-synthetic-profile \
  --profile \
  --profile-warmup-steps 2 \
  --profile-active-steps 2 \
  --model openlm-research/open_llama_7b_v2 \
  --zero-stage 3 \
  --bench-step 6 \
  --warmup-step 1 \
  --seq-length 256 \
  --dataset-name synthetic \
  --dataset-samples 1024
```

Profile result: success. Metrics from
`diagnostic_runs/z3-openllama-synthetic-profile/metrics.json`:

- Avg step time: `1.1706783771514893` sec
- Samples/s: `3.4168223126601647`
- Tokens/s: `874.7065120410022`
- `torch.cuda.max_memory_allocated()`: `36057900544` bytes
- Reserved memory: `46022000640` bytes
- Profile summary fields: rank imbalance `low`, allocator pressure `medium`,
  compile/graph-break signal `not-measured`, and other fields `unknown`; note:
  rank-0 profiler reported no CUDA self time for this short DeepSpeed run

Observed setup blockers now documented or worked around:

- `meta-llama/Meta-Llama-3-8B` failed with Hugging Face gated-repo `401`; use a
  public Llama-family smoke model until HF auth is confirmed.
- The default WikiText path failed under the installed
  `datasets`/`huggingface_hub` versions; `dataset_name=synthetic` is now the
  default smoke path.

Cleanup status: `anyscale workspace_v2 terminate --id expwrk_qeeei1bv25rps3hqt5j6raz6kd`
completed, and `anyscale workspace_v2 wait --id expwrk_qeeei1bv25rps3hqt5j6raz6kd --state TERMINATED`
verified state `TERMINATED`.
