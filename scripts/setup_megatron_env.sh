#!/usr/bin/env bash
# Build the Megatron-LM benchmarking venv: ./.venv-megatron (Python 3.12).
#
# Megatron core_v0.18.0 + Megatron-FSDP on the TransformerEngine path
# (--transformer-impl transformer_engine): TE fused attention (cuDNN) + TENorm. This is
# the representative, optimized Megatron path and the one our matrix benchmarks.
#
# Hard-won notes baked in here (see also the v0 debugging):
#   - Megatron core_v0.18.0 uses `from typing import override` -> needs Python 3.12+.
#   - torch is cu126 (not cu124): nvcc on this box is 12.6+, and TE's bindings + Apex's
#     cuda_ext must match torch's CUDA. CUDA_HOME=cuda-12.6.
#   - TE has NO prebuilt `transformer-engine-torch` wheel: the bindings build from source.
#     `transformer-engine[pytorch,core_cu12]` gives a prebuilt cu12 core + source bindings.
#   - This box has cuda-12.6/12.8/12.9 AND cuda-13.0 all in ldconfig; TE's cudart-version
#     probe then dlopens both libcudart.so.12 and .so.13 and aborts ("Multiple libcudart").
#     scripts/patch_te_cuda13_probe.py rewrites the .so.13 probe -> .so.99 to fix it, and
#     scripts/run_megatron.sh pins CUDNN_HOME/LD_LIBRARY_PATH to the venv's cu12 cuDNN.
#   - uv venvs omit python3-config (the dataset-helper Makefile needs it) + pybind11.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

VENV=".venv-megatron"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export PATH="$CUDA_HOME/bin:$PATH"

uv venv "$VENV" --python 3.12
PY="$repo_root/$VENV/bin/python"

# python3-config (for Megatron's megatron/core/datasets helper Makefile extension suffix)
PYBINDIR="$(dirname "$(readlink -f "$VENV/bin/python")")"
[ -e "$PYBINDIR/python3-config" ] && ln -sf "$PYBINDIR/python3-config" "$VENV/bin/python3-config"

# cu126 (not cu124): Apex's cuda_ext build requires nvcc's CUDA version to match
# torch's, and the toolkits on this box are 12.6+ (no 12.4). CUDA_HOME=cuda-12.6.
uv pip install -p "$PY" "torch==2.6.*" --index-url https://download.pytorch.org/whl/cu126
uv pip install -p "$PY" setuptools wheel packaging psutil pyyaml ninja einops sentencepiece tiktoken numpy regex pybind11

# uv venvs ship no pip; TE's source build + Apex's legacy setup.py both need it.
uv pip install -p "$PY" pip

# TransformerEngine 2.16.1: prebuilt cu12 core + source-built pytorch bindings. Restrict
# the bindings compile to sm_90 (H100/H200) with MAX_JOBS to avoid an OOM on a busy node.
echo ">> installing TransformerEngine (cu12 core + pytorch bindings, sm_90)..."
MAX_JOBS="${MAX_JOBS:-4}" TORCH_CUDA_ARCH_LIST="9.0" NVTE_FRAMEWORK=pytorch \
  "$PY" -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
  "transformer-engine[pytorch,core_cu12]==2.16.1"
# Fix TE's cudart-version probe on this multi-CUDA box (see header + the script's docstring).
"$PY" scripts/patch_te_cuda13_probe.py

# Apex is OPTIONAL on the TE path: TE supplies the multi_tensor ops Megatron-FSDP needs.
# (It was required only on the old local/unfused path.) Build it with SKIP_APEX=0 if you
# want apex's fused optimizers too; SLOW (~30-60 min), needs nvcc, uses pip's --build-option.
if [ "${SKIP_APEX:-1}" != "1" ]; then
  echo ">> building NVIDIA Apex with cuda_ext (optional; the long pole)..."
  "$PY" -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext --cuda_ext" \
    "git+https://github.com/NVIDIA/apex.git" || echo "WARN: Apex build failed (optional)"
fi

"$PY" -c "import torch; print('torch', torch.__version__)"
"$PY" -c "import transformer_engine as te; print('transformer_engine', te.__version__)"
echo "OK: Megatron venv at $VENV  (Megatron runs from third_party/Megatron-LM via PYTHONPATH)"
