#!/usr/bin/env bash
# Build the TorchTitan benchmarking venv: ./.venv-torchtitan (Python 3.12).
# torch-only (no TE/apex/flash needed) — the smooth cross-framework path.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

VENV=".venv-torchtitan"
uv venv "$VENV" --python 3.12          # torchtitan main uses 3.12 typing features
PY="$repo_root/$VENV/bin/python"

uv pip install -p "$PY" "torch==2.6.*" --index-url https://download.pytorch.org/whl/cu124
uv pip install -p "$PY" setuptools wheel packaging
# Editable install pulls torchtitan's own requirements (datasets, tomli, etc.).
# NOTE: this also pulls a recent torch (overriding the 2.6 line above) — torchtitan
# main tracks recent torch; that's fine, this venv is isolated.
uv pip install -p "$PY" -e third_party/torchtitan

# Compat patch: torchtitan calls create_block_mask(separate_full_blocks=...), a kwarg
# only in a narrow torch nightly window. Make it drop unsupported kwargs (idempotent).
"$PY" scripts/patch_torchtitan.py

# torchtitan's qwen3_14b config uses the C4 dataset + the Qwen3-14B tokenizer.
# Qwen is NOT gated (no HF token). Paths are relative to the torchtitan root.
echo ">> downloading Qwen3-14B tokenizer assets"
( cd third_party/torchtitan && "$PY" scripts/download_hf_assets.py --repo_id Qwen/Qwen3-14B --assets tokenizer ) || \
  echo "WARN: tokenizer download failed — torchtitan will error until assets/hf/Qwen3-14B exists"

"$PY" -c "import torch; print('torch', torch.__version__)"
echo "OK: TorchTitan venv at $VENV"
