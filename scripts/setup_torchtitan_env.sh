#!/usr/bin/env bash
# Build the TorchTitan benchmarking venv: ./.venv-torchtitan (Python 3.12).
# torch-only (no TE/apex/flash needed) — the smooth cross-framework path.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

VENV=".venv-torchtitan"
uv venv "$VENV" --python 3.12          # torchtitan main uses 3.12 typing features
PY="$repo_root/$VENV/bin/python"

uv pip install -p "$PY" setuptools wheel packaging
# torchtitan REQUIRES a PyTorch NIGHTLY (per its README): it's PyTorch's reference
# codebase for the newest distributed/compile APIs (FSDP2, DTensor, FlexAttention
# create_block_mask kwargs, ...) that aren't in stable torch. cu130 per the README;
# matches this node's CUDA 13. torchdata nightly is required alongside nightly torch.
#
# The editable install pulls torchtitan's requirements (datasets/tyro/tokenizers/...)
# AND a *stable* torch via torchdata; we override torch + torchdata to nightly after.
TT_CUDA="${TORCHTITAN_NIGHTLY_CUDA:-cu130}"
uv pip install -p "$PY" -e third_party/torchtitan
uv pip install -p "$PY" --pre torchdata --index-url https://download.pytorch.org/whl/nightly/cpu
# Force the NIGHTLY torch: the steps above pull a *stable* torch that already satisfies
# the requirement, so uv won't upgrade it (the README uses pip --force-reinstall).
# Uninstall first, then install from the nightly index to get the real dev build.
uv pip uninstall -p "$PY" torch || true
uv pip install -p "$PY" --pre torch --index-url "https://download.pytorch.org/whl/nightly/${TT_CUDA}"

# Safety-net compat patch (idempotent; nightly should already accept the kwarg, but
# this keeps it robust if torch and the pinned torchtitan commit drift).
"$PY" scripts/patch_torchtitan.py

# torchtitan's qwen3_* configs use the C4 dataset + the per-model tokenizer at
# assets/hf/<repo>. Qwen is NOT gated (no HF token). Each qwen3_configs entry hard-codes
# its own hf_assets_path, so every benchmarked model needs its tokenizer (all identical
# Qwen tokenizers, but torchtitan looks them up by model dir). Paths relative to the
# torchtitan root.
for repo in Qwen/Qwen3-14B Qwen/Qwen3-32B Qwen/Qwen3-30B-A3B; do
  echo ">> downloading $repo tokenizer assets"
  ( cd third_party/torchtitan && "$PY" scripts/download_hf_assets.py --repo_id "$repo" --assets tokenizer ) || \
    echo "WARN: tokenizer download failed for $repo — torchtitan will error until assets/hf/${repo#*/} exists"
done

"$PY" -c "import torch; print('torch', torch.__version__)"
echo "OK: TorchTitan venv at $VENV"
