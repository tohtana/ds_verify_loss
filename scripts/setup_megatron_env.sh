#!/usr/bin/env bash
# Build the Megatron-LM benchmarking venv: ./.venv-megatron (Python 3.12).
#
# Megatron core_v0.18.0 + Megatron-FSDP, run WITHOUT TransformerEngine and WITHOUT
# flash-attn (pure-PyTorch local impl + unfused attention). Apex IS installed,
# because Megatron-FSDP's optimizer/grad-clip/mixed-precision fall back to a triton
# `multi_tensor_applier` without it, and that triton path crashes ("0 active
# drivers"). Apex provides the native multi_tensor ops instead.
#
# Hard-won notes baked in here (see also the v0 debugging):
#   - Megatron core_v0.18.0 uses `from typing import override` -> needs Python 3.12+.
#   - TE metapackage resolves to a cu13 wheel on this box -> won't load vs cu124 torch.
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

uv pip install -p "$PY" "torch==2.6.*" --index-url https://download.pytorch.org/whl/cu124
uv pip install -p "$PY" setuptools wheel packaging psutil pyyaml ninja einops sentencepiece tiktoken numpy regex pybind11

# Apex with C++/CUDA extensions (multi_tensor ops). SLOW build (~30-60 min); needs nvcc.
# Set SKIP_APEX=1 to skip, but Megatron-FSDP will then crash in the triton fallback.
if [ "${SKIP_APEX:-0}" != "1" ]; then
  echo ">> building NVIDIA Apex with cuda_ext (this is the long pole)..."
  uv pip install -p "$PY" -v --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" \
    "git+https://github.com/NVIDIA/apex.git" || \
    echo "WARN: Apex build failed -> Megatron-FSDP will hit the triton multi_tensor fallback"
fi

"$PY" -c "import torch; print('torch', torch.__version__)"
"$PY" -c "import apex; from apex.multi_tensor_apply import multi_tensor_applier; print('apex OK')" 2>&1 | tail -1
echo "OK: Megatron venv at $VENV  (Megatron runs from third_party/Megatron-LM via PYTHONPATH)"
