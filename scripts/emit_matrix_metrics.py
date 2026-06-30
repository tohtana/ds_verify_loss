#!/usr/bin/env python3
"""
Parse a Megatron-LM or TorchTitan training log + run parameters and write a
metrics.json in the SAME schema that verify_loss.py produces, so the cells show
up in scripts/summarize_repro_matrix.py tables alongside fsdp/deepspeed/deepcompile.

Step time comes from the framework's own per-iteration logging:
  - megatron:    "elapsed time per iteration (ms): <N>"
  - torchtitan:  "tps: <N>"  (tokens/sec PER DEVICE) -> step_time = tokens/gpu / tps

Peak memory is sampled out-of-process via `nvidia-smi memory.used` (peak across all
GPUs over the run); it is reported as the cross-rank reserved figure. This is
GPU-used memory (≈ torch reserved + CUDA context), not torch's allocated figure,
so `cuda_max_memory_allocated_cross_rank_bytes` is left null. The summarizer's
"peak reserved (GiB)" table is therefore populated; "peak allocated" shows "-".
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import socket
from datetime import datetime
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def megatron_step_times_sec(text: str) -> list[float]:
    ms = [float(x) for x in re.findall(r"elapsed time per iteration \(ms\):\s*([\d.]+)", text)]
    return [v / 1000.0 for v in ms]


def torchtitan_step_times_sec(text: str, tokens_per_step_per_gpu: int) -> list[float]:
    text = ANSI.sub("", text)
    tps = [float(x.replace(",", "")) for x in re.findall(r"tps:\s*([\d,]+)", text)]
    return [tokens_per_step_per_gpu / t for t in tps if t > 0]


def peak_reserved_bytes(gpu_mem_log: str | None) -> int | None:
    if not gpu_mem_log or not Path(gpu_mem_log).is_file():
        return None
    vals = []
    for line in Path(gpu_mem_log).read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            vals.append(float(line))  # MiB
        except ValueError:
            continue
    return int(max(vals) * 1024 * 1024) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", required=True, choices=["megatron", "torchtitan"])
    ap.add_argument("--log", required=True)
    ap.add_argument("--metrics-output", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch-size", type=int, required=True, help="micro-batch per GPU")
    ap.add_argument("--seq-length", type=int, required=True)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=1)
    ap.add_argument("--num-processes", type=int, required=True)
    ap.add_argument("--warmup-step", type=int, required=True)
    ap.add_argument("--measured-steps", type=int, default=0)
    ap.add_argument("--activation-checkpointing", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--gpu-mem-log", default=None)
    ap.add_argument("--return-code", type=int, default=0)
    args = ap.parse_args()

    text = Path(args.log).read_text(errors="ignore") if Path(args.log).is_file() else ""
    samples_per_step = args.batch_size * args.num_processes * args.gradient_accumulation_steps
    tokens_per_step = samples_per_step * args.seq_length
    tokens_per_step_per_gpu = args.batch_size * args.gradient_accumulation_steps * args.seq_length

    if args.framework == "megatron":
        step_times = megatron_step_times_sec(text)
    else:
        step_times = torchtitan_step_times_sec(text, tokens_per_step_per_gpu)

    measured = step_times[args.warmup_step:] if len(step_times) > args.warmup_step else []
    peak = peak_reserved_bytes(args.gpu_mem_log)
    success = bool(measured) and args.return_code == 0

    payload: dict = {
        "status": "success" if success else "failed",
        "success": success,
        "generated_at": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "model_name": args.model,
        "backend": args.framework,
        "num_processes": args.num_processes,
        "batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "seq_length": args.seq_length,
        "activation_checkpointing": bool(args.activation_checkpointing),
        "compile": bool(args.compile),
        "deepcompile": False,
        "global_samples_per_step": samples_per_step,
        "global_tokens_per_step": tokens_per_step,
        "warmup_step": args.warmup_step,
        "measured_steps": len(measured),
        "cuda_max_memory_allocated_cross_rank_bytes": None,
        "cuda_max_memory_reserved_cross_rank_bytes": peak,
        "cuda_max_memory_allocated_bytes": None,
        "cuda_max_memory_reserved_bytes": peak,
        "memory_source": "nvidia-smi memory.used peak across GPUs (~torch reserved)",
    }

    if success:
        avg = sum(measured) / len(measured)
        payload.update({
            "error_summary": None,
            "avg_step_time_sec": avg,
            "samples_per_second": samples_per_step / avg if avg > 0 else None,
            "tokens_per_second": tokens_per_step / avg if avg > 0 else None,
        })
    else:
        tail = "\n".join(text.splitlines()[-60:])
        payload.update({
            "error_summary": {
                "type": "RunFailed",
                "message": f"return_code={args.return_code}, parsed_steps={len(step_times)}, "
                           f"measured_after_warmup={len(measured)}",
                "traceback_tail": tail[-6000:],
            },
            "avg_step_time_sec": None,
            "samples_per_second": None,
            "tokens_per_second": None,
        })

    out = Path(args.metrics_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[emit_metrics] {args.framework}: success={success} measured_steps={len(measured)} "
          f"avg_step_sec={payload.get('avg_step_time_sec')} tokens_per_sec={payload.get('tokens_per_second')} "
          f"peak_reserved_bytes={peak} -> {out}")


if __name__ == "__main__":
    main()
