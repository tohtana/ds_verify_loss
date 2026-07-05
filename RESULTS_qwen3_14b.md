# Qwen3-14B Training-Framework Benchmark — H100 & H200

## PR description

This PR extends the DeepCompile reproduction matrix so the **same Qwen3 workload can be
benchmarked across five training frameworks** — FSDP, DeepSpeed (ZeRO-3), DeepCompile,
TorchTitan, and Megatron-LM — on one 8-GPU node. It adds per-framework launchers
(`scripts/run_megatron.sh`, `scripts/run_torchtitan.sh`) that emit `metrics.json` in the
existing schema, pins Megatron-LM / TorchTitan / DeepSpeed as `third_party` submodules
(DeepSpeed tracks the personal fork so local edits are testable), and wires the matrix
runner to select model + activation-checkpointing per cell. Model coverage is Qwen3-14B
(dense), with Qwen3-32B (dense) and Qwen3-30B-A3B (MoE) configs also included.

Results below are Qwen3-14B, micro-batch 1, sequence 2048, 8×H200 and 8×H100, for both
activation-checkpointing (AC) on and off. Headline: **Megatron (Megatron-FSDP + TE fused
attention) and DeepCompile are the fastest**, DeepSpeed the slowest at this scale. With AC
**on**, DeepCompile leads (it does not recompute — see notes); with AC **off**, Megatron
leads. Megatron is also the most hardware-robust — only ~3–4% slower H100-vs-H200, versus
up to ~1.5× for FSDP — because its TE kernels are compute- rather than bandwidth-bound.

## Setup (identical across frameworks unless noted)

| | |
|---|---|
| Model | Qwen3-14B, **random-init from HF config** (no pretrained weights) |
| Batch / seq | micro-batch 1, grad-accum 1, global batch 8, seq 2048 |
| Parallelism | 8-way data-parallel, ZeRO-3-equivalent sharding (FSDP FULL_SHARD / ZeRO-3 / Megatron-FSDP `optim_grads_params`) |
| Precision | bf16 |
| Optimizer | Adam/AdamW |
| Attention | FSDP/DeepSpeed/DeepCompile: HF SDPA · TorchTitan: FlexAttention · Megatron: TE cuDNN fused |
| Measured | mean of 20 steps after warmup (5 eager / 10 compiled) |
| Data | synthetic |

Step time is the primary, directly-comparable metric. Throughput = 16384 tokens / step.

---

## Results — 8×H200 (140 GB/GPU)

### AC on
| method | step time (s) | peak mem (GiB) |
|---|---|---|
| deepcompile | **0.588** | 79.8 |
| megatron | 0.620 | 60.5 † |
| torchtitan | 0.693 | 42.6 † |
| fsdp | 0.843 | 65.2 |
| deepspeed | 1.118 | 39.4 |

### AC off
| method | step time (s) | peak mem (GiB) |
|---|---|---|
| megatron | **0.525** | 74.7 † |
| deepcompile | 0.586 | 79.8 |
| torchtitan | 0.599 | 58.4 † |
| fsdp | 0.607 | 70.4 |
| deepspeed | 0.915 | 61.4 |

---

## Results — 8×H100 (80 GB/GPU)

### AC on
| method | step time (s) | peak mem (GiB) |
|---|---|---|
| megatron | **0.640** | 60.5 † |
| deepcompile | 0.666 | 58.2 |
| torchtitan | 0.754 | 42.6 † |
| deepspeed | 1.237 | 39.9 |
| fsdp | 1.256 | 40.0 |

### AC off
| method | step time (s) | peak mem (GiB) |
|---|---|---|
| megatron | **0.546** | 74.7 † |
| torchtitan | 0.649 | 58.4 † |
| fsdp | 0.666 | 70.4 |
| deepcompile | 0.666 | 59.4 |
| deepspeed | 1.022 | 62.0 |

---

## Notes

- **Memory is measured with two rulers**, so compare *within* a type, not across: FSDP /
  DeepSpeed / DeepCompile report torch allocator `reserved`; Megatron / TorchTitan (†)
  report NVML device-used peak, which includes the CUDA context + cuDNN/TE workspaces and
  so reads higher. `reserved` is also allocator-dependent (e.g. DeepCompile reserves ~80
  GiB on H200 but ~58 GiB on H100 simply because there is less room to cache), so treat it
  as a rough envelope rather than a fixed footprint.
- **DeepCompile ignores `--ac`** (identical step time and memory on/off) — the harness
  skips HF gradient-checkpointing for it because DeepCompile does its own compile-planned
  selective activation persistence. This is also why it wins with AC on (it is not paying
  the recompute tax the others are) and why it is the most memory-hungry method.
- **AC off is faster but uses more memory** for the other four (no recompute) — the
  expected trade-off; the winner flips from DeepCompile (AC on) to Megatron (AC off).
- **fsdp AC-on on H100** is a slowdown outlier (1.256 s, 1.49× its H200 time) despite low
  reserved memory (40 GiB) — likely allocator pressure / param all-gather churn under
  H100's tighter memory + lower bandwidth.
- DeepCompile does **not** fit Qwen3-32B / 30B-A3B on either H100 or H200 (it holds the
  full param set per rank → OOM); the other four run for those larger models.

_Runs: H200 `20260705T033838Z` (AC on) / `20260705T041338Z` (AC off);
H100 `20260705T045109Z` (AC on) / `20260705T051155Z` (AC off), under `repro_matrix_runs/`._
