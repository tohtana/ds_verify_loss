# Qwen3-14B DeepCompile agent reproduction

This port is based on client commit `d14152aefc86930f448f9742f7cafa2fa7a3437f`. It retains that commit's
synthetic fixed-token dataset, ZeRO-3/DeepCompile setup, metrics JSON, cross-rank timing and memory evidence, and full
matrix summarizer. The only benchmark-path addition is selection of DeepSpeed's single graph-agent loop at the
warmup optimization point.

The persistent worktrees are:

- current-master baseline: `../deepspeed-baseline` at `715965e027894a2e72ac2e27f2daed2c599e99f0`
- agent candidate: `../deepspeed-agent` at `1b506f73ffd5cf5c5c938d1e7864343b280e0e09`

The persistent `.venv-qwen` environment resolves to torch `2.10.0+cu128`, transformers `4.51.3`, accelerate `1.14.0`,
datasets `5.0.0`, and wandb `0.28.0`. The cell launcher prepends the selected DeepSpeed worktree to `PYTHONPATH` and
refuses a wrong DeepSpeed SHA, package version, or visible GPU count.

## First same-input pair: mb1/seq1024

Run these sequentially from one 8xH100 node. Both commands use Qwen/Qwen3-14B, 8 ranks, ZeRO-3 + DeepCompile,
synthetic fixed tokens, seed 42, micro-batch 1, sequence length 1024, gradient accumulation 1, 10 warmup steps, and 20
measured steps. The baseline uses the fixed current-master prefetch/selective-gather optimizer; the candidate replaces
that warmup optimizer with the graph-agent loop. Both retain the same step-0 `z3` pass.

```bash
HARNESS=/mnt/cluster_storage/dc-agent-two-agent-rebuild-20260826/work/qwen-agent-harness
PAIR_ROOT=/mnt/cluster_storage/dc-agent-two-agent-rebuild-20260826/qwen-results/pair-$(date -u +%Y%m%dT%H%M%SZ)
export PATH="$HARNESS/.venv-qwen/bin:/usr/bin:/bin"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd "$HARNESS/client"
scripts/run_qwen_deepcompile_cell.sh \
  --variant baseline \
  --results-dir "$PAIR_ROOT/baseline-mb1-seq1024" \
  --batch-size 1 \
  --seq-length 1024 \
  --main-process-port 29541

scripts/run_qwen_deepcompile_cell.sh \
  --variant agent \
  --results-dir "$PAIR_ROOT/agent-mb1-seq1024" \
  --batch-size 1 \
  --seq-length 1024 \
  --main-process-port 29542
```

The candidate command inherits the persistent Codex credentials and pins every graph-agent call to `gpt-5.6-sol` with
`xhigh` reasoning. One external runner handles both schema-v4 stages: `accepted_graph` returns `continue` with the exact
`GraphEditPayload`, and `candidate` returns `accept` or `reject`. Set `CODEX_BIN` or `CODEX_HOME` explicitly before
launch only if the persistent workspace installation has moved.

## Confirmed end-to-end run (2026-08-27)

The agent variant completed one full Qwen/Qwen3-14B run on one 8xH100 node with 8 ranks, ZeRO-3 + DeepCompile,
micro-batch 1, sequence length 1024, synthetic 8192-sample input, seed 42, 10 warmup steps, and 20 measured steps.
The step-0 `z3` pass remained fixed; the single graph agent was invoked at step 5 using `gpt-5.6-sol` with `xhigh`
reasoning.

- Forward: rank 0 proposed one generic `reorder` operation. All 8 ranks replayed it successfully with one identical
  graph fingerprint and ABI-compatible outputs. The measured candidate regressed device time from 473.946 ms to
  521.412 ms (+10.01%), so the same agent rejected it and then returned `finish` on the retained graph.
- Backward: rank 0 proposed one generic `reorder` operation. All 8 ranks again replayed it successfully with one
  identical graph fingerprint and ABI-compatible outputs. The measured candidate regressed device time from 547.098
  ms to 551.474 ms (+0.80%), so the same agent rejected it and then returned `finish` on the retained graph.
- Training resumed after both graph-agent loops and completed step 30 with return code 0. Final loss was `12.940394`,
  mean measured step time was `0.566943` seconds, and aggregate throughput was `14449.417` tokens/second.

This run validates the end-to-end control path: rank-0 agent edit generation, deterministic edit finalization, edit
broadcast, all-rank replay and profiling, measured candidate evaluation by the same agent, and continuation of normal
training. Neither measured rewrite was accepted because both were slower; that is an optimization result rather than a
control-path failure.

## Established full matrix

After the first pair proves wiring, the original 27-cell matrix remains available: framework in
`{fsdp,deepspeed,deepcompile}`, micro-batch in `{1,2,4}`, and sequence length in `{1024,2048,4096}`. Each row records its
command, generated configs, logs, result status, metrics, and all-rank evidence. Failed or OOM cells do not prevent later
rows from running.

Baseline matrix:

```bash
HARNESS=/mnt/cluster_storage/dc-agent-two-agent-rebuild-20260826/work/qwen-agent-harness
export PATH="$HARNESS/.venv-qwen/bin:/usr/bin:/bin"
export PYTHONPATH="$HARNESS/deepspeed-baseline"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=/mnt/cluster_storage/dc-agent-two-agent-rebuild-20260826/cache/huggingface
export XDG_CACHE_HOME=/mnt/cluster_storage/dc-agent-two-agent-rebuild-20260826/cache/xdg
cd "$HARNESS/client"
scripts/run_deepcompile_repro_matrix.sh \
  --results-root /mnt/cluster_storage/dc-agent-two-agent-rebuild-20260826/qwen-results/baseline-matrix \
  --nproc 8 \
  --model Qwen/Qwen3-14B \
  --frameworks "fsdp deepspeed deepcompile" \
  --mbs "1 2 4" \
  --seqs "1024 2048 4096" \
  --deepcompile-warmup 10 \
  --measured-steps 20 \
  --dataset-samples 8192 \
  --seed 42
```

Agent-candidate matrix (in a fresh shell, or after replacing `PYTHONPATH`):

```bash
HARNESS=/mnt/cluster_storage/dc-agent-two-agent-rebuild-20260826/work/qwen-agent-harness
export PATH="$HARNESS/.venv-qwen/bin:/usr/bin:/bin"
export PYTHONPATH="$HARNESS/deepspeed-agent"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=/mnt/cluster_storage/dc-agent-two-agent-rebuild-20260826/cache/huggingface
export XDG_CACHE_HOME=/mnt/cluster_storage/dc-agent-two-agent-rebuild-20260826/cache/xdg
export CODEX_HOME=/mnt/cluster_storage/dc-agent-two-agent-rebuild-20260826/tools/codex-home
export CODEX_BIN=/mnt/cluster_storage/dc-agent-two-agent-rebuild-20260826/tools/codex-cli/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex
cd "$HARNESS/client"
scripts/run_deepcompile_repro_matrix.sh \
  --results-root /mnt/cluster_storage/dc-agent-two-agent-rebuild-20260826/qwen-results/agent-matrix \
  --nproc 8 \
  --model Qwen/Qwen3-14B \
  --frameworks "fsdp deepspeed deepcompile" \
  --mbs "1 2 4" \
  --seqs "1024 2048 4096" \
  --deepcompile-warmup 10 \
  --measured-steps 20 \
  --dataset-samples 8192 \
  --seed 42 \
  --zero3-tuning-strategy agent \
  --agent-backend codex \
  --agent-max-iterations 3 \
  --agent-max-retries-per-iteration 1 \
  --agent-timeout-sec 300
```
