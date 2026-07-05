# Megatron-LM & TorchTitan in the reproduction matrix

This adds **Megatron-LM** and **TorchTitan** as two more frameworks in the same
`mb × seq` matrix as `fsdp` / `deepspeed` / `deepcompile`, on the same model
(**Qwen/Qwen3-14B**) with the same per-cell knobs (micro-batch, seq length,
activation checkpointing, warmup + measured steps). Each cell writes a
`metrics.json` in the exact schema `verify_loss.py` produces, so they appear in
`scripts/summarize_repro_matrix.py` tables and ratios next to the others. Models:
**Qwen3-14B** (dense), **Qwen3-32B** (dense), **Qwen3-30B-A3B** (MoE).

## Clone & setup

All three `third_party` submodules are public, so a plain recursive clone works:

```bash
git clone --recurse-submodules <repo-url> && cd <repo>
#   already cloned?  ->  git submodule update --init --recursive
```

Each framework family gets its own venv (their deps conflict). **Run these on a GPU
node** — setup builds GPU extensions (TE / apex) and the runs need GPUs:

```bash
bash scripts/setup_ds_env.sh          # .venv-ds:        fsdp / deepspeed / deepcompile
                                      #                  (editable -e third_party/DeepSpeed)
bash scripts/setup_torchtitan_env.sh  # .venv-torchtitan: torch nightly + Qwen3 tokenizers
bash scripts/setup_megatron_env.sh    # .venv-megatron:  TransformerEngine (cu12) + Megatron-FSDP
```

Then run the matrix (the runner pins `.venv-ds` onto PATH per-cell, so no venv activation
needed; the run dir auto-tags the GPU, e.g. `…-qwen3-14b-8xh100-repro`):

```bash
NGPUS_PER_NODE=8 scripts/run_deepcompile_repro_matrix.sh \
  --model Qwen/Qwen3-14B \
  --frameworks "fsdp deepspeed deepcompile torchtitan megatron" \
  --mbs "1 2 4" --seqs "1024 2048 4096" --ac on --cell-timeout-s 900
```

For batch submission on H100 use `sbatch scripts/sweep_h100.sbatch`. Results land in
`repro_matrix_runs/<ts>-<model>-<hw>-repro/`; see `RESULTS_qwen3_14b.md` for reference numbers.

## Layout

```
third_party/
  Megatron-LM/        # submodule (NVIDIA), pinned to core_v0.18.0
  torchtitan/         # submodule (pytorch), pinned; ignore=dirty (runtime-patched)
  DeepSpeed/          # submodule (deepspeedai, public), pinned; installed editable
configs/
  megatron/qwen3_{14b,32b,30b_a3b}.args   # static Megatron args (model + impl); launcher adds per-cell
  torchtitan/qwen3_{14b,32b,30b_a3b}.flags# optional static torchtitan overrides
scripts/
  setup_{ds,megatron,torchtitan}_env.sh   # build the three venvs
  run_megatron.sh / run_torchtitan.sh     # one cell -> metrics.json (own venv)
  patch_torchtitan.py / patch_te_cuda13_probe.py  # env-specific runtime patches
  emit_matrix_metrics.py                  # parse framework log -> ds_verify_loss metrics schema
```

## Why separate venvs

These frameworks have dependencies that **conflict** with the DeepSpeed harness env
(`transformers==4.51.3` + DeepSpeed master). Megatron `core_v0.18.0` needs **Python
3.12** (`typing.override`); torchtitan tracks recent `main`. So each gets its own venv;
the launchers use `MEGATRON_PYTHON` / `TORCHTITAN_PYTHON` (default `./.venv-{fw}/bin/python`).
See **Clone & setup** above for the build commands.

## Run

The full-matrix command is in **Clone & setup** above. To run a single cell directly:

```bash
NGPUS_PER_NODE=8 scripts/run_torchtitan.sh \
  --model Qwen/Qwen3-14B --batch-size 1 --seq-length 2048 \
  --bench_step 25 --warmup_step 5 --activation_checkpointing \
  --metrics_output runs/torchtitan-mb1-seq2048/metrics.json
```

## How the metrics are produced

* **Step time** comes from each framework's own per-iteration logging
  (Megatron `elapsed time per iteration (ms)`, TorchTitan per-device `tps`),
  averaged over the measured steps after `warmup_step` — same methodology as
  `verify_loss.py`.
* **Peak memory** is sampled out-of-process via `nvidia-smi memory.used` (peak across
  all GPUs over the run) and reported as **cross-rank reserved** (GiB). This is
  GPU-used memory (≈ torch reserved + CUDA context), so the summarizer's *peak
  allocated* table shows `-` for these two frameworks while *peak reserved* is
  populated. `fsdp`/`deepspeed`/`deepcompile` still report torch's exact figures.

## Comparability notes (read before interpreting)

Both run **ZeRO-3-equivalent full sharding** so memory is comparable:

| Framework | Sharding | Attention | Recompute (when `--activation_checkpointing`) |
|-----------|----------|-----------|------------|
| TorchTitan | FSDP2 full-shard | FlexAttention | full (FullAC) |
| Megatron | Megatron-FSDP `optim_grads_params` | **TE cuDNN fused** | full (`--recompute-granularity full`) |

* **TorchTitan** is native PyTorch FSDP2 (same engine as the `fsdp` cells) — the most
  directly comparable cross-framework point.
* **Megatron** runs the **representative TransformerEngine path**
  (`--transformer-impl transformer_engine`): TE cuDNN fused attention + TENorm, the
  optimized Megatron config. This is the fast, real-world Megatron number (not the
  earlier unfused pure-PyTorch path).
* **Attention differs per framework** (SDPA for fsdp/deepspeed/deepcompile, FlexAttention
  for TorchTitan, TE-cuDNN for Megatron), so this is a *framework + its native attention*
  comparison, not pure sharding overhead.

## Pitfalls already handled (from v0)

Baked into the configs/scripts so you don't rediscover them: Megatron needs Python
3.12; torch is **cu126** (matches nvcc, so TE/apex build); **TransformerEngine** is the
cu12 core + source-built pytorch bindings (sm_90), and on this multi-CUDA box (cuda-12.6
+ cuda-13.0 both in ldconfig) TE's cudart-version probe finds both `libcudart.so.12` and
`.so.13` and aborts — `scripts/patch_te_cuda13_probe.py` rewrites the `.so.13` probe so
only cu12 is found, and `run_megatron.sh` pins `CUDNN_HOME`/`LD_LIBRARY_PATH` to the
venv's cu12 cuDNN; uv venvs omit `python3-config` and `pybind11` (dataset-helper build) —
symlinked/installed; on the TE path apex is optional (TE supplies `multi_tensor`);
`--eval-interval` must be set even with eval off; Megatron-FSDP forbids
`CUDA_DEVICE_MAX_CONNECTIONS=1` and requires `--ckpt-format fsdp_dtensor`.
TorchTitan needs the per-model Qwen3 tokenizer (downloaded); its configs default to
FullAC; and it calls `create_block_mask(separate_full_blocks=...)`, a torch nightly-only
kwarg — `scripts/patch_torchtitan.py` strips unsupported kwargs so it runs across torch
versions. DeepCompile keeps the full param set per rank, so it OOMs at Qwen3-32B/30B-A3B.

> **Comparability caveat:** torchtitan's editable install pulls a *recent* torch
> (its own venv), so torchtitan may run on a newer torch than the fsdp/deepspeed
> cells (torch 2.6). Same framework comparison still holds, but absolute kernel
> performance isn't on an identical torch.
