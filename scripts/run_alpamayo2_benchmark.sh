#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="nvidia/Alpamayo2-Super"
MODEL_REVISION="00554695e729a6ff0b6281fd2c81b18d06e33dbe"
OFFICIAL_SOURCE_REVISION="beb2977d9a7e9d66837d4a3ad5144ff59de37519"
DEEPSPEED_REVISION="${DEEPSPEED_REVISION:-79046032e5d6800a547348f6b0c7b3e1f112e5ce}"
HARNESS_CANDIDATE_REVISION="${DS_VERIFY_LOSS_CANDIDATE_SHA:-unknown}"
MASTER_PORT="${MASTER_PORT:-29673}"

: "${ALPAMAYO2_SOURCE_REPO:?set ALPAMAYO2_SOURCE_REPO to the official source checkout}"
: "${ALPAMAYO2_CACHE_ROOT:?set ALPAMAYO2_CACHE_ROOT to storage sized for the 72 GB checkpoint}"
: "${ALPAMAYO2_OUTPUT_ROOT:?set ALPAMAYO2_OUTPUT_ROOT to persistent benchmark storage}"

if [[ "$(git -C "${ALPAMAYO2_SOURCE_REPO}" rev-parse HEAD)" != "${OFFICIAL_SOURCE_REVISION}" ]]; then
    echo "official Alpamayo2 source revision mismatch" >&2
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

mkdir -p "${ALPAMAYO2_CACHE_ROOT}" "${ALPAMAYO2_OUTPUT_ROOT}"
export HF_HOME="${ALPAMAYO2_CACHE_ROOT}/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false

python -m pip install 'torchvision==0.25.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install --upgrade \
    'transformers==4.57.1' 'accelerate>=1.12.0,<2' 'huggingface-hub>=0.34.0,<1.0' \
    'hydra-core>=1.3.2' 'hydra-colorlog>=1.2.0' 'physical-ai-av>=0.2.0' \
    'av>=16.0.1' 'einops>=0.8.1' 'mediapy>=1.2.4' 'pillow>=12.0.0' 'scipy>=1.16.0'
python -m pip install --no-deps -e "${ALPAMAYO2_SOURCE_REPO}"

MODEL_PATH="${ALPAMAYO2_CACHE_ROOT}/models/alpamayo2-super-${MODEL_REVISION}"
mkdir -p "${MODEL_PATH}"
hf download "${MODEL_ID}" --revision "${MODEL_REVISION}" --local-dir "${MODEL_PATH}"

RUN_ID="${ALPAMAYO2_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${ALPAMAYO2_OUTPUT_ROOT}/${RUN_ID}"
BATCH_CACHE="${ALPAMAYO2_CACHE_ROOT}/batches/alpamayo2-super-attempt0.pt"
mkdir -p "${RUN_DIR}" "$(dirname "${BATCH_CACHE}")"

python alpamayo2_benchmark.py \
    --prepare-batch \
    --model-path "${MODEL_PATH}" \
    --batch-cache "${BATCH_CACHE}" 2>&1 | tee "${RUN_DIR}/prepare-batch.log"
PREPARE_STATUS=${PIPESTATUS[0]}
if [[ "${PREPARE_STATUS}" != "0" ]]; then
    exit "${PREPARE_STATUS}"
fi

python - <<PY > "${RUN_DIR}/setup-manifest.json"
import json, os, platform, subprocess, torch, transformers, deepspeed
def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()
print(json.dumps({
    "model_id": "${MODEL_ID}",
    "model_revision": "${MODEL_REVISION}",
    "official_source_revision": "${OFFICIAL_SOURCE_REVISION}",
    "deepspeed_revision": "${DEEPSPEED_REVISION}",
    "harness_candidate_revision": "${HARNESS_CANDIDATE_REVISION}",
    "harness_checkout_revision": git("rev-parse", "HEAD"),
    "harness_effective_tree": git("write-tree"),
    "official_source_checkout_revision": git("-C", os.environ["ALPAMAYO2_SOURCE_REPO"], "rev-parse", "HEAD"),
    "deepspeed_checkout_revision": git("-C", os.path.join(os.environ["DEVDS_REPOS_DIR"], "DeepSpeed"), "rev-parse", "HEAD"),
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
    local -a extra=()
    if [[ "${backend}" == "deepspeed" ]]; then
        extra=(--deepspeed-config configs/alpamayo2_zero3.json)
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
        --warmup-steps 1 \
        --measured-steps 3 \
        "${extra[@]}" 2>&1 | tee "${RUN_DIR}/${backend}.log"
    return "${PIPESTATUS[0]}"
}

set +e
run_backend fsdp
FSDP_STATUS=$?
run_backend deepspeed
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
find "${RUN_DIR}" -maxdepth 1 -type f -print -exec sha256sum {} \;
for artifact in "${RUN_DIR}"/*.json; do
    echo "ARTIFACT_JSON ${artifact}"
    cat "${artifact}"
done

# Both exact-path attempts are evidence even when one or both fail.
exit 0
