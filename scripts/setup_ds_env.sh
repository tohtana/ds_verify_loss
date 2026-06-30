#!/usr/bin/env bash
# Build the DeepSpeed / FSDP / DeepCompile harness venv: ./.venv-ds
# (the colleague's env, from the repo README). run.sh uses `accelerate launch` from
# PATH, so ACTIVATE this venv before running fsdp/deepspeed/deepcompile cells:
#
#   bash scripts/setup_ds_env.sh
#   source .venv-ds/bin/activate
#   NGPUS_PER_NODE=8 scripts/run_deepcompile_repro_matrix.sh --frameworks "fsdp deepspeed"
#
# megatron/torchtitan use their own venvs (via MEGATRON_PYTHON/TORCHTITAN_PYTHON),
# so the full 5-framework matrix also works with .venv-ds active.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

VENV=".venv-ds"
uv venv "$VENV" --python 3.10
PY="$repo_root/$VENV/bin/python"

# torch 2.6 (cu124): DeepCompile needs torch >= 2.6; matches the validated v0 stack.
uv pip install -p "$PY" "torch==2.6.*" --index-url https://download.pytorch.org/whl/cu124
# transformers/accelerate + the DeepSpeed runtime deps (DeepSpeed is installed
# --no-deps below, so list its deps here: einops/hjson/msgpack/ninja/numpy/
# packaging/psutil/py-cpuinfo/pydantic/tqdm). The repo README's list was incomplete.
uv pip install -p "$PY" "transformers==4.51.3" accelerate datasets wandb setuptools wheel \
    einops hjson msgpack ninja numpy packaging psutil py-cpuinfo pydantic tqdm nvidia-ml-py

# DeepSpeed master (has DeepCompile). JIT-compiles ops at runtime, so the install is
# just the Python package. --no-deps per the repo README (deps installed above).
uv pip install -p "$PY" --no-deps "git+https://github.com/deepspeedai/DeepSpeed.git@master"

"$PY" -c "import torch,accelerate,transformers,deepspeed,datasets,huggingface_hub as h; \
print('torch',torch.__version__,'| transformers',transformers.__version__,'| deepspeed',deepspeed.__version__)"
echo "OK: source $VENV/bin/activate  before running fsdp/deepspeed/deepcompile cells"
