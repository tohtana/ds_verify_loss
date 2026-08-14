#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="nvidia/Alpamayo2-Super"
MODEL_REVISION="00554695e729a6ff0b6281fd2c81b18d06e33dbe"
OFFICIAL_SOURCE_REVISION="beb2977d9a7e9d66837d4a3ad5144ff59de37519"
DEEPSPEED_REVISION="79046032e5d6800a547348f6b0c7b3e1f112e5ce"
HARNESS_CANDIDATE_REVISION="${DS_VERIFY_LOSS_CANDIDATE_SHA:-unknown}"
HARNESS_CANDIDATE_DIFF_SHA256="${DS_VERIFY_LOSS_CANDIDATE_DIFF_SHA256:-unknown}"
MASTER_PORT="${MASTER_PORT:-29673}"
BENCHMARK_LANE="${ALPAMAYO2_BENCHMARK_LANE:-baseline}"

: "${ALPAMAYO2_SOURCE_REPO:?set ALPAMAYO2_SOURCE_REPO to the official source checkout}"
: "${DEEPSPEED_SOURCE_REPO:?set DEEPSPEED_SOURCE_REPO to the pinned DeepSpeed source checkout}"
: "${ALPAMAYO2_CACHE_ROOT:?set ALPAMAYO2_CACHE_ROOT to the immutable prepared shared asset root}"
: "${ALPAMAYO2_OUTPUT_ROOT:?set ALPAMAYO2_OUTPUT_ROOT to persistent benchmark storage}"
: "${ALPAMAYO2_LOCAL_STAGE_ROOT:?set ALPAMAYO2_LOCAL_STAGE_ROOT to a unique run-local path}"
: "${DS_VERIFY_LOSS_CANDIDATE_CONTENT_SHA256:?set the reviewed harness content digest}"

if [[ "${BENCHMARK_LANE}" != "baseline" && "${BENCHMARK_LANE}" != "deepspeed-deepcompile" ]]; then
    echo "ALPAMAYO2_BENCHMARK_LANE must be baseline or deepspeed-deepcompile" >&2
    exit 2
fi

if [[ "$(git -C "${ALPAMAYO2_SOURCE_REPO}" rev-parse HEAD)" != "${OFFICIAL_SOURCE_REVISION}" ]]; then
    echo "official Alpamayo2 source revision mismatch" >&2
    exit 2
fi
if [[ "$(git -C "${DEEPSPEED_SOURCE_REPO}" rev-parse HEAD)" != "${DEEPSPEED_REVISION}" ]]; then
    echo "DeepSpeed source revision mismatch" >&2
    exit 2
fi
if [[ -n "$(git -C "${DEEPSPEED_SOURCE_REPO}" status --porcelain)" ]]; then
    echo "DeepSpeed source checkout must be clean" >&2
    exit 2
fi
if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')" != "8" ]]; then
    echo "attempt 0 requires exactly 8 visible H100 GPUs" >&2
    exit 2
fi
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -vq 'H100'; then
    echo "attempt 0 requires H100 GPUs" >&2
    exit 2
fi

HARNESS_CONTENT_SHA256="$(python - <<'PY'
import hashlib
from pathlib import Path

files = [
    "README.md",
    "alpamayo2_benchmark.py",
    "configs/alpamayo2_zero3.json",
    "configs/alpamayo2_zero3_deepcompile.json",
    "scripts/run_alpamayo2_benchmark.sh",
    "scripts/stage_alpamayo2_assets.py",
    "tests/test_alpamayo2_benchmark_contract.py",
]
hasher = hashlib.sha256()
for name in sorted(files):
    digest = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    hasher.update(f"{name}\0{digest}\n".encode())
print(hasher.hexdigest())
PY
)"
if [[ "${HARNESS_CONTENT_SHA256}" != "${DS_VERIFY_LOSS_CANDIDATE_CONTENT_SHA256}" ]]; then
    echo "reviewed harness content digest mismatch" >&2
    exit 2
fi

RUN_ID="${ALPAMAYO2_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "ALPAMAYO2_RUN_ID must contain only letters, digits, dot, underscore, or hyphen" >&2
    exit 2
fi
LOCAL_ROOT="${ALPAMAYO2_LOCAL_STAGE_ROOT}"
if [[ "${LOCAL_ROOT%/}" != /mnt/local_storage/* ]]; then
    echo "ALPAMAYO2_LOCAL_STAGE_ROOT must be under /mnt/local_storage" >&2
    exit 2
fi
if [[ "${LOCAL_ROOT%/}" != */"${RUN_ID}" ]]; then
    echo "ALPAMAYO2_LOCAL_STAGE_ROOT must end with the current ALPAMAYO2_RUN_ID" >&2
    exit 2
fi
MODEL_PATH="${LOCAL_ROOT}/models/alpamayo2-super-${MODEL_REVISION}"
DATASET_PATH="${LOCAL_ROOT}/datasets/coco-val2017-1000-v1"
BATCH_CACHE="${LOCAL_ROOT}/batches/alpamayo2-super-coco1000-v1"
RUN_DIR="${ALPAMAYO2_OUTPUT_ROOT}/${RUN_ID}"
mkdir -p "${RUN_DIR}"

python scripts/stage_alpamayo2_assets.py \
    --shared-asset-root "${ALPAMAYO2_CACHE_ROOT}" \
    --local-root "${LOCAL_ROOT}" 2>&1 | tee "${RUN_DIR}/stage-assets.log"

export HF_HOME="${LOCAL_ROOT}/huggingface"
export TORCHINDUCTOR_CACHE_DIR="${LOCAL_ROOT}/torchinductor-cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export ALPAMAYO2_EFFECTIVE_LOCAL_ROOT="${LOCAL_ROOT}"
export ALPAMAYO2_EFFECTIVE_MODEL_PATH="${MODEL_PATH}"
export ALPAMAYO2_EFFECTIVE_DATASET_PATH="${DATASET_PATH}"
export ALPAMAYO2_EFFECTIVE_BATCH_CACHE="${BATCH_CACHE}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

# Install from run-local clones so the pinned source checkouts remain read-only.
LOCAL_SOURCE_ROOT="${LOCAL_ROOT}/sources"
mkdir -p "${LOCAL_SOURCE_ROOT}"
git clone --quiet --shared "${ALPAMAYO2_SOURCE_REPO}" "${LOCAL_SOURCE_ROOT}/alpamayo2"
git -C "${LOCAL_SOURCE_ROOT}/alpamayo2" checkout --quiet --detach "${OFFICIAL_SOURCE_REVISION}"
git clone --quiet --shared "${DEEPSPEED_SOURCE_REPO}" "${LOCAL_SOURCE_ROOT}/DeepSpeed"
git -C "${LOCAL_SOURCE_ROOT}/DeepSpeed" checkout --quiet --detach "${DEEPSPEED_REVISION}"
ALPAMAYO2_RUNTIME_SOURCE="${LOCAL_SOURCE_ROOT}/alpamayo2"
DEEPSPEED_RUNTIME_SOURCE="${LOCAL_SOURCE_ROOT}/DeepSpeed"

python -m pip install 'torchvision==0.25.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install --upgrade \
    'transformers==4.57.1' 'accelerate>=1.12.0,<2' 'huggingface-hub>=0.34.0,<1.0' \
    'hydra-core>=1.3.2' 'hydra-colorlog>=1.2.0' 'physical-ai-av>=0.2.0' \
    'av>=16.0.1' 'einops>=0.8.1' 'mediapy>=1.2.4' 'pillow>=12.0.0' 'scipy>=1.16.0' \
    hjson ninja nvidia-ml-py py-cpuinfo
python -m pip install --no-deps -e "${ALPAMAYO2_RUNTIME_SOURCE}"
python -m pip install --no-deps -e "${DEEPSPEED_RUNTIME_SOURCE}"
DEEPSPEED_IMPORT_REVISION="$(git -C "${DEEPSPEED_RUNTIME_SOURCE}" rev-parse --short HEAD)"
python - "${DEEPSPEED_REVISION}" "${DEEPSPEED_IMPORT_REVISION}" "${DEEPSPEED_RUNTIME_SOURCE}" <<'PY'
import pathlib
import sys

import deepspeed
from alpamayo2_benchmark import validate_deepspeed_import_identity

expected_revision = sys.argv[1]
expected_import_revision = sys.argv[2]
source_root = pathlib.Path(sys.argv[3])
validate_deepspeed_import_identity(
    expected_revision,
    expected_import_revision,
    source_root,
    getattr(deepspeed, "__git_hash__", None),
    pathlib.Path(deepspeed.__file__),
)
PY

python alpamayo2_benchmark.py \
    --prepare-batch \
    --model-path "${MODEL_PATH}" \
    --dataset-path "${DATASET_PATH}" \
    --batch-cache "${BATCH_CACHE}" 2>&1 | tee "${RUN_DIR}/prepare-batch.log"
PREPARE_STATUS=${PIPESTATUS[0]}
if [[ "${PREPARE_STATUS}" != "0" ]]; then
    exit "${PREPARE_STATUS}"
fi

python - <<PY > "${RUN_DIR}/setup-manifest.json"
import hashlib, json, os, platform, subprocess
from pathlib import Path

import deepspeed, torch, transformers

def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()
harness_files = [
    "README.md",
    "alpamayo2_benchmark.py",
    "configs/alpamayo2_zero3.json",
    "configs/alpamayo2_zero3_deepcompile.json",
    "scripts/run_alpamayo2_benchmark.sh",
    "scripts/stage_alpamayo2_assets.py",
    "tests/test_alpamayo2_benchmark_contract.py",
]
harness_file_sha256 = {
    name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in harness_files
}
content_hasher = hashlib.sha256()
for name, digest in sorted(harness_file_sha256.items()):
    content_hasher.update(f"{name}\0{digest}\n".encode())
print(json.dumps({
    "model_id": "${MODEL_ID}",
    "model_revision": "${MODEL_REVISION}",
    "official_source_revision": "${OFFICIAL_SOURCE_REVISION}",
    "deepspeed_revision": "${DEEPSPEED_REVISION}",
    "harness_candidate_revision": "${HARNESS_CANDIDATE_REVISION}",
    "harness_candidate_diff_sha256": "${HARNESS_CANDIDATE_DIFF_SHA256}",
    "harness_checkout_revision": git("rev-parse", "HEAD"),
    "harness_content_sha256": content_hasher.hexdigest(),
    "expected_harness_content_sha256": "${DS_VERIFY_LOSS_CANDIDATE_CONTENT_SHA256}",
    "benchmark_lane": "${BENCHMARK_LANE}",
    "harness_file_sha256": harness_file_sha256,
    "official_source_checkout_revision": git("-C", os.environ["ALPAMAYO2_SOURCE_REPO"], "rev-parse", "HEAD"),
    "deepspeed_source_checkout_revision": git("-C", os.environ["DEEPSPEED_SOURCE_REPO"], "rev-parse", "HEAD"),
    "deepspeed_import_revision": deepspeed.__git_hash__,
    "deepspeed_import_path": deepspeed.__file__,
    "local_asset_root": os.environ["ALPAMAYO2_EFFECTIVE_LOCAL_ROOT"],
    "local_model_path": os.environ["ALPAMAYO2_EFFECTIVE_MODEL_PATH"],
    "local_dataset_path": os.environ["ALPAMAYO2_EFFECTIVE_DATASET_PATH"],
    "local_batch_cache": os.environ["ALPAMAYO2_EFFECTIVE_BATCH_CACHE"],
    "python": platform.python_version(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "deepspeed": deepspeed.__version__,
    "cuda_runtime": torch.version.cuda,
    "gpu_inventory": subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines(),
    "topology": subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True).splitlines(),
}, indent=2))
PY

run_backend() {
    local backend="$1"
    local deepspeed_config="$2"
    local warmup_steps="$3"
    local measured_steps="$4"
    local -a extra=()
    if [[ -n "${deepspeed_config}" ]]; then
        extra=(--deepspeed-config "${deepspeed_config}")
    fi
    torchrun --nnodes=1 --node-rank=0 --nproc-per-node=8 \
        --master-addr=127.0.0.1 --master-port="${MASTER_PORT}" \
        alpamayo2_benchmark.py \
        --backend "${backend}" \
        --model-path "${MODEL_PATH}" \
        --batch-cache "${BATCH_CACHE}" \
        --output-dir "${RUN_DIR}" \
        --deepspeed-revision "${DEEPSPEED_REVISION}" \
        --learning-rate 1e-6 \
        --warmup-steps "${warmup_steps}" \
        --measured-steps "${measured_steps}" \
        "${extra[@]}" 2>&1 | tee "${RUN_DIR}/${backend}.log"
    return "${PIPESTATUS[0]}"
}

if [[ "${BENCHMARK_LANE}" == "deepspeed-deepcompile" ]]; then
    set +e
    run_backend deepspeed-deepcompile configs/alpamayo2_zero3_deepcompile.json 20 105
    DEEPCOMPILE_STATUS=$?
    set -e
    python - <<PY | tee "${RUN_DIR}/attempt-summary.json"
import json
print(json.dumps({
    "run_dir": "${RUN_DIR}",
    "benchmark_lane": "deepspeed-deepcompile",
    "deepspeed_deepcompile_exit_code": ${DEEPCOMPILE_STATUS},
}, indent=2))
PY
else
    set +e
    run_backend fsdp "" 1 124
    FSDP_STATUS=$?
    run_backend deepspeed configs/alpamayo2_zero3.json 1 124
    DEEPSPEED_STATUS=$?
    set -e
    python - <<PY | tee "${RUN_DIR}/attempt-summary.json"
import json
print(json.dumps({
    "run_dir": "${RUN_DIR}",
    "fsdp_exit_code": ${FSDP_STATUS},
    "deepspeed_exit_code": ${DEEPSPEED_STATUS},
}, indent=2))
PY
fi
find "${RUN_DIR}" -maxdepth 1 -type f -print -exec sha256sum {} \;
for artifact in "${RUN_DIR}"/*.json; do
    echo "ARTIFACT_JSON ${artifact}"
    cat "${artifact}"
done

# Preserve both rows as evidence, but never report a successful launcher when a row failed.
if [[ "${BENCHMARK_LANE}" == "deepspeed-deepcompile" ]]; then
    if [[ "${DEEPCOMPILE_STATUS}" != "0" ]]; then
        exit 1
    fi
elif [[ "${FSDP_STATUS}" != "0" || "${DEEPSPEED_STATUS}" != "0" ]]; then
    exit 1
fi
