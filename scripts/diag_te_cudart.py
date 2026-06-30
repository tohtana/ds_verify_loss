#!/usr/bin/env python3
"""Diagnose TE's 'Multiple libcudart' (cu12 vs cu13) on this box.

Run on the GPU node with the SAME env the megatron launcher sets:

    cd ~/workspace/training-framework-benchmarking-v1
    export CUDA_HOME=/usr/local/cuda-12.6 CUDA_PATH=/usr/local/cuda-12.6
    .venv-megatron/bin/python scripts/diag_te_cudart.py

It prints which libcudart/libcudnn are mapped after each stage, so we can see
exactly which import drags in cuda-13.0's libcudart.so.13.
"""
import os, sys


def mapped():
    libs = {}
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                p = line.split()[-1]
                if ("libcudart.so" in p or "libcudnn" in p) and p not in libs:
                    libs[p] = True
    except OSError:
        pass
    return sorted(libs)


def show(stage):
    print(f"\n=== after {stage} ===")
    for p in mapped():
        tag = "  <-- CU13!" if ".so.13" in p or "cuda-13" in p else ""
        print(f"  {p}{tag}")


print("CUDA_HOME =", os.environ.get("CUDA_HOME"))
print("CUDA_PATH =", os.environ.get("CUDA_PATH"))
print("CUDNN_HOME =", os.environ.get("CUDNN_HOME"))
print("LD_LIBRARY_PATH =", os.environ.get("LD_LIBRARY_PATH", "")[:400])
show("startup")

import torch  # noqa: E402
print("\ntorch", torch.__version__, "cuda", torch.version.cuda)
torch.cuda.init()
torch.zeros(1, device="cuda")
show("import torch + cuda init")

import transformer_engine  # noqa: E402
show("import transformer_engine")

import transformer_engine.pytorch as te  # noqa: E402
show("import transformer_engine.pytorch")

# Trigger the cuDNN fused-attention backend (where the run actually died).
try:
    import transformer_engine.pytorch.attention as _a  # noqa: F401
    dpa = te.DotProductAttention(num_attention_heads=8, kv_channels=128,
                                 attention_dropout=0.0).cuda()
    # qkv_format defaults to 'sbhd' -> (seq, batch, heads, head_dim)
    b, s, h, d = 2, 256, 8, 128
    q = torch.randn(s, b, h, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(s, b, h, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(s, b, h, d, device="cuda", dtype=torch.bfloat16)
    out = dpa(q, k, v)
    torch.cuda.synchronize()
    print("\nfused attention forward OK, out", tuple(out.shape))
except Exception as e:  # noqa: BLE001
    print("\nfused attention FAILED:", type(e).__name__, str(e)[:300])
show("fused attention attempt")
print("\n(any line tagged CU13 above is the leak)")
