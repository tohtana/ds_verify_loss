#!/usr/bin/env python3
"""Fail fast unless the active Python/DeepSpeed selection matches the Qwen contract."""

import argparse
import json
from pathlib import Path
import subprocess

import deepspeed
import torch
import transformers


EXPECTED_TORCH = "2.10.0+cu128"
EXPECTED_TRANSFORMERS = "4.51.3"


def git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-deepspeed-sha", required=True)
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--skip-gpu-check", action="store_true")
    args = parser.parse_args()

    deepspeed_root = Path(deepspeed.__file__).resolve().parents[1]
    detected = {
        "python": __import__("sys").executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "deepspeed_file": str(Path(deepspeed.__file__).resolve()),
        "deepspeed_sha": git_head(deepspeed_root),
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
    }

    errors = []
    if detected["torch"] != EXPECTED_TORCH:
        errors.append(f"torch must be {EXPECTED_TORCH}, found {detected['torch']}")
    if detected["transformers"] != EXPECTED_TRANSFORMERS:
        errors.append(f"transformers must be {EXPECTED_TRANSFORMERS}, found {detected['transformers']}")
    if detected["deepspeed_sha"] != args.expected_deepspeed_sha:
        errors.append(f"DeepSpeed must be {args.expected_deepspeed_sha}, found {detected['deepspeed_sha']}")
    if not args.skip_gpu_check:
        if not detected["cuda_available"]:
            errors.append("CUDA is not available")
        if detected["gpu_count"] != args.expected_gpus:
            errors.append(f"expected {args.expected_gpus} visible GPUs, found {detected['gpu_count']}")

    print(json.dumps(detected, indent=2, sort_keys=True))
    if errors:
        raise SystemExit("environment contract failed: " + "; ".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
