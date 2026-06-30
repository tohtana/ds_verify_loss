# Megatron-LM & TorchTitan in the reproduction matrix

This adds **Megatron-LM** and **TorchTitan** as two more frameworks in the same
`mb × seq` matrix as `fsdp` / `deepspeed` / `deepcompile`, on the same model
(**Qwen/Qwen3-14B**) with the same per-cell knobs (micro-batch, seq length,
activation checkpointing, warmup + measured steps). Each cell writes a
`metrics.json` in the exact schema `verify_loss.py` produces, so they appear in
`scripts/summarize_repro_matrix.py` tables and ratios next to the others.

## Layout

```
third_party/
  Megatron-LM/        # submodule, pinned to core_v0.18.0
  torchtitan/         # submodule, pinned (qwen3_14b native config)
configs/
  megatron/qwen3_14b.args      # static Megatron args (model + impl); launcher adds per-cell
  torchtitan/qwen3_14b.flags   # optional static torchtitan overrides
scripts/
  setup_megatron_env.sh        # builds ./.venv-megatron  (Python 3.12 + apex)
  setup_torchtitan_env.sh      # builds ./.venv-torchtitan (Python 3.12, torch-only)
  run_megatron.sh              # one cell -> metrics.json
  run_torchtitan.sh            # one cell -> metrics.json
  emit_matrix_metrics.py       # parse framework log -> ds_verify_loss metrics schema
```

## Why separate venvs

These frameworks have dependencies that **conflict** with the DeepSpeed harness env
(`transformers==4.51.3` + DeepSpeed master). Megatron `core_v0.18.0` needs **Python
3.12** (`typing.override`); torchtitan tracks recent `main`. So each gets its own venv;
the launchers use `MEGATRON_PYTHON` / `TORCHTITAN_PYTHON` (default `./.venv-{fw}/bin/python`).

```bash
bash scripts/setup_torchtitan_env.sh    # fast, torch-only; downloads Qwen3-14B tokenizer
bash scripts/setup_megatron_env.sh      # slow: builds NVIDIA Apex (~30-60 min)
```

## Run

In the matrix, just add them to `--frameworks`:

```bash
NGPUS_PER_NODE=8 scripts/run_deepcompile_repro_matrix.sh \
  --frameworks "fsdp deepspeed deepcompile megatron torchtitan" \
  --mbs "1 2 4" --seqs "1024 2048 4096" \
  --measured-steps 20 --eager-warmup 5
```

Or a single cell directly:

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
| TorchTitan | FSDP2 full-shard | SDPA (flash/mem-efficient) | full |
| Megatron | Megatron-FSDP `optim_grads_params` | **unfused** (no flash) | full (`--recompute-granularity full`) |

* **TorchTitan** is native PyTorch FSDP2 (same engine as the `fsdp` cells) — the most
  directly comparable cross-framework point.
* **Megatron** here is the **pure-PyTorch path** (no TransformerEngine, no flash-attn):
  attention is **unfused**, so its step time is **not representative** of a TE/flash
  Megatron — expect it slower. It's a correct "Megatron runs Qwen3-14B with FSDP"
  data point. For a representative Megatron, install TE + flash-attn and switch
  `configs/megatron/qwen3_14b.args` to `--transformer-impl transformer_engine` +
  `--attention-backend flash`.

## Pitfalls already handled (from v0)

Baked into the configs/scripts so you don't rediscover them: Megatron needs Python
3.12; TE's cu13 wheel is incompatible with cu124 torch (skipped → local impl); uv
venvs omit `python3-config` and `pybind11` (dataset-helper build) — symlinked/installed;
RoPE/grad-accum/swiglu fusions + persistent-layernorm require TE/apex (disabled);
`--eval-interval` must be set even with eval off; Megatron-FSDP's `multi_tensor`
fallback needs apex (installed); Megatron-FSDP forbids `CUDA_DEVICE_MAX_CONNECTIONS=1`.
TorchTitan needs the Qwen3-14B tokenizer (downloaded); its `qwen3_14b` config already
defaults to FullAC; and it calls `create_block_mask(separate_full_blocks=...)`, a torch
nightly-only kwarg — `scripts/patch_torchtitan.py` strips unsupported kwargs so it runs
on stable torch.

> **Comparability caveat:** torchtitan's editable install pulls a *recent* torch
> (its own venv), so torchtitan may run on a newer torch than the fsdp/deepspeed
> cells (torch 2.6). Same framework comparison still holds, but absolute kernel
> performance isn't on an identical torch.
