#!/usr/bin/env bash
set -euo pipefail

NUM_NODES=${NUM_NODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}
NUM_PROCESSES=$((${NUM_NODES} * ${NGPUS_PER_NODE}))

BACKEND="deepspeed"
MODEL="meta-llama/Meta-Llama-3-8B"
ZERO_STAGE=3
COMPILE=0
FP16=0
PASSES="ALL"
EXTRA_OPTS=""

EAGER=0
DEEPCOMPILE=0
GRADIENT_ACCUMULATION_STEPS=1
ACTIVATION_CHECKPOINTING=0
BATCH_SIZE=1
SEQ_LENGTH=512
DEBUG_LOG=0
SYNC_BEFORE_REDUCE=0
SYNC_AFTER_REDUCE=0
SYNC_BEFORE_ALLGATHER=0
SYNC_AFTER_ALLGATHER=0
ZERO_OVERLAP_COMM=true
ZERO_CONTIGUOUS_GRADIENTS=""
ZERO_REDUCE_SCATTER=""
ZERO_ALLGATHER_PARTITIONS=""
ZERO_REDUCE_BUCKET_SIZE=0
ZERO_ALLGATHER_BUCKET_SIZE=0
ZERO_SUB_GROUP_SIZE=0
ZERO_STAGE3_PREFETCH_BUCKET_SIZE=0
ZERO_STAGE3_PARAM_PERSISTENCE_THRESHOLD=-1
ZERO_STAGE3_MAX_LIVE_PARAMETERS=0
ZERO_STAGE3_MAX_REUSE_DISTANCE=0
ZERO_STAGE3_OFFLOAD_PARAM_DEVICE=""
ZERO_STAGE3_OFFLOAD_PARAM_PIN_MEMORY=true
CHUNKED_CAUSAL_LM_LOSS_TOKENS=0
CHUNKED_CAUSAL_LM_LOSS_EMPTY_CACHE=0
CHUNKED_CAUSAL_LM_LOSS_DEVICE="cuda"

HOST_IP="127.0.0.1"
MACHINE_RANK=0
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-12345}

echo "NUM_NODES: ${NUM_NODES} NGPUS_PER_NODE: ${NGPUS_PER_NODE} NUM_PROCESSES: ${NUM_PROCESSES}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --host_ip)
            HOST_IP="$2"
            shift 2
            ;;
        --machine_rank)
            MACHINE_RANK="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --zero_stage|--zero-stage)
            ZERO_STAGE="$2"
            shift 2
            ;;
        --batch_size|--batch-size)
            BATCH_SIZE="$2"
            EXTRA_OPTS="${EXTRA_OPTS} --batch_size $2"
            shift 2
            ;;
        --seq_length|--seq-length)
            SEQ_LENGTH="$2"
            EXTRA_OPTS="${EXTRA_OPTS} --seq_length $2"
            shift 2
            ;;
        --gradient_accumulation_steps|--gradient-accumulation-steps)
            GRADIENT_ACCUMULATION_STEPS="$2"
            EXTRA_OPTS="${EXTRA_OPTS} --gradient_accumulation_steps $2"
            shift 2
            ;;
        --activation_checkpointing|--activation-checkpointing)
            ACTIVATION_CHECKPOINTING=1
            EXTRA_OPTS="${EXTRA_OPTS} --activation_checkpointing"
            shift
            ;;   
        --compile)
            COMPILE=1
            EXTRA_OPTS="${EXTRA_OPTS} $1"
            shift
            ;;
        --eager)
            EAGER=1
            EXTRA_OPTS="${EXTRA_OPTS} --backend eager"
            shift
            ;;
        --deepcompile)
            DEEPCOMPILE=1
            shift
            ;;
        --passes)
            PASSES="$2"
            EXTRA_OPTS="${EXTRA_OPTS} $1 $2"
            shift 2
            ;;
        --model|--model_name|--model-name)
            MODEL="$2"
            shift 2
            ;;
        --fp16)
            FP16=1
            shift
            ;;
        --bf16)
            FP16=0
            shift
            ;;
        --debug_log)
            DEBUG_LOG=1
            shift
            ;;
        --sync_before_reduce)
            SYNC_BEFORE_REDUCE=1
            shift
            ;;
        --sync_after_reduce)
            SYNC_AFTER_REDUCE=1
            shift
            ;;
        --sync_before_allgather)
            SYNC_BEFORE_ALLGATHER=1
            shift
            ;;
        --sync_after_allgather)
            SYNC_AFTER_ALLGATHER=1
            shift
            ;;
        --zero_overlap_comm|--zero-overlap-comm)
            ZERO_OVERLAP_COMM="$2"
            shift 2
            ;;
        --zero_contiguous_gradients|--zero-contiguous-gradients)
            ZERO_CONTIGUOUS_GRADIENTS="$2"
            shift 2
            ;;
        --zero_reduce_scatter|--zero-reduce-scatter)
            ZERO_REDUCE_SCATTER="$2"
            shift 2
            ;;
        --zero_allgather_partitions|--zero-allgather-partitions)
            ZERO_ALLGATHER_PARTITIONS="$2"
            shift 2
            ;;
        --zero_reduce_bucket_size|--zero-reduce-bucket-size)
            ZERO_REDUCE_BUCKET_SIZE="$2"
            shift 2
            ;;
        --zero_allgather_bucket_size|--zero-allgather-bucket-size)
            ZERO_ALLGATHER_BUCKET_SIZE="$2"
            shift 2
            ;;
        --zero_sub_group_size|--zero-sub-group-size)
            ZERO_SUB_GROUP_SIZE="$2"
            shift 2
            ;;
        --zero_stage3_prefetch_bucket_size|--zero-stage3-prefetch-bucket-size)
            ZERO_STAGE3_PREFETCH_BUCKET_SIZE="$2"
            shift 2
            ;;
        --zero_stage3_param_persistence_threshold|--zero-stage3-param-persistence-threshold)
            ZERO_STAGE3_PARAM_PERSISTENCE_THRESHOLD="$2"
            shift 2
            ;;
        --zero_stage3_max_live_parameters|--zero-stage3-max-live-parameters)
            ZERO_STAGE3_MAX_LIVE_PARAMETERS="$2"
            shift 2
            ;;
        --zero_stage3_max_reuse_distance|--zero-stage3-max-reuse-distance)
            ZERO_STAGE3_MAX_REUSE_DISTANCE="$2"
            shift 2
            ;;
        --zero_stage3_offload_param_device|--zero-stage3-offload-param-device)
            ZERO_STAGE3_OFFLOAD_PARAM_DEVICE="$2"
            shift 2
            ;;
        --zero_stage3_offload_param_pin_memory|--zero-stage3-offload-param-pin-memory)
            ZERO_STAGE3_OFFLOAD_PARAM_PIN_MEMORY="$2"
            shift 2
            ;;
        --no_zero_stage3_offload_param_pin_memory|--no-zero-stage3-offload-param-pin-memory)
            ZERO_STAGE3_OFFLOAD_PARAM_PIN_MEMORY=false
            shift
            ;;
        --chunked_causal_lm_loss_tokens|--chunked-causal-lm-loss-tokens)
            CHUNKED_CAUSAL_LM_LOSS_TOKENS="$2"
            EXTRA_OPTS="${EXTRA_OPTS} --chunked_causal_lm_loss_tokens $2"
            shift 2
            ;;
        --chunked_causal_lm_loss_empty_cache|--chunked-causal-lm-loss-empty-cache)
            CHUNKED_CAUSAL_LM_LOSS_EMPTY_CACHE=1
            EXTRA_OPTS="${EXTRA_OPTS} --chunked_causal_lm_loss_empty_cache"
            shift
            ;;
        --chunked_causal_lm_loss_device|--chunked-causal-lm-loss-device)
            CHUNKED_CAUSAL_LM_LOSS_DEVICE="$2"
            EXTRA_OPTS="${EXTRA_OPTS} --chunked_causal_lm_loss_device $2"
            shift 2
            ;;
        *)
            # Check if the next argument looks like a value (doesn't start with --)
            if [[ $# -gt 1 && ! "$2" =~ ^-- ]]; then
                EXTRA_OPTS="${EXTRA_OPTS} $1 $2"
                shift 2
            else
                EXTRA_OPTS="${EXTRA_OPTS} $1"
                shift
            fi
            ;;
    esac
done



export NCCL_DEBUG=WARN

CONFIG_TEMPLATE=configs/ds_config.yaml.template
if [ "${BACKEND}" == "fsdp" ]; then
    CONFIG_TEMPLATE=configs/fsdp_config.yaml.template
elif [ "${BACKEND}" == "ddp" ]; then
    CONFIG_TEMPLATE=configs/ddp_config.yaml.template
elif [ "${BACKEND}" == "singlegpu" ]; then
    CONFIG_TEMPLATE=configs/singlegpu_config.yaml.template
elif [ "${BACKEND}" != "deepspeed" ]; then
    echo "Invalid backend: ${BACKEND}"
    exit 1
fi

if [ "${BACKEND}" != "deepspeed" ]; then
    ZERO_STAGE=0
fi

echo "HOST_IP: ${HOST_IP}"
echo "MAIN_PROCESS_PORT: ${MAIN_PROCESS_PORT}"
echo "NUM_NODES: ${NUM_NODES}"
echo "NUM_PROCESSES: ${NUM_PROCESSES}"
echo "BACKEND: ${BACKEND}"
echo "ZERO_STAGE: ${ZERO_STAGE}"
echo "MODEL: ${MODEL}"
echo "GRADIENT_ACCUMULATION_STEPS: ${GRADIENT_ACCUMULATION_STEPS}"
echo "ZERO_STAGE3_OFFLOAD_PARAM_DEVICE: ${ZERO_STAGE3_OFFLOAD_PARAM_DEVICE:-none}"
echo "ZERO_STAGE3_OFFLOAD_PARAM_PIN_MEMORY: ${ZERO_STAGE3_OFFLOAD_PARAM_PIN_MEMORY}"
echo "CHUNKED_CAUSAL_LM_LOSS_TOKENS: ${CHUNKED_CAUSAL_LM_LOSS_TOKENS}"
echo "CHUNKED_CAUSAL_LM_LOSS_EMPTY_CACHE: ${CHUNKED_CAUSAL_LM_LOSS_EMPTY_CACHE}"
echo "CHUNKED_CAUSAL_LM_LOSS_DEVICE: ${CHUNKED_CAUSAL_LM_LOSS_DEVICE}"
echo "EXTRA_OPTS: ${EXTRA_OPTS}"

python generate_conf.py \
    --machine_rank ${MACHINE_RANK} \
    --num_machines ${NUM_NODES} \
    --num_processes ${NUM_PROCESSES} \
    --zero_stage ${ZERO_STAGE} \
    --template_file ${CONFIG_TEMPLATE} \
    --output_file configs/config.yaml

GAS_OPTS="--gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}"

if [ "${BACKEND}" == "deepspeed" ]; then
    DEEPCOMPILE_OPTS=""
    if [ "${DEEPCOMPILE}" == "1" ]; then
        DEEPCOMPILE_OPTS="--deepcompile"
    fi

    DEBUG_LOG_OPTS=""
    if [ "${DEBUG_LOG}" == "1" ]; then
        DEBUG_LOG_OPTS="--debug_log"
    fi

    SYNC_BEFORE_REDUCE_OPTS=""
    if [ "${SYNC_BEFORE_REDUCE}" == "1" ]; then
        SYNC_BEFORE_REDUCE_OPTS="--sync_before_reduce"
    fi
    
    SYNC_AFTER_REDUCE_OPTS=""
    if [ "${SYNC_AFTER_REDUCE}" == "1" ]; then
        SYNC_AFTER_REDUCE_OPTS="--sync_after_reduce"
    fi

    SYNC_BEFORE_ALLGATHER_OPTS=""
    if [ "${SYNC_BEFORE_ALLGATHER}" == "1" ]; then
        SYNC_BEFORE_ALLGATHER_OPTS="--sync_before_allgather"
    fi

    SYNC_AFTER_ALLGATHER_OPTS=""
    if [ "${SYNC_AFTER_ALLGATHER}" == "1" ]; then
        SYNC_AFTER_ALLGATHER_OPTS="--sync_after_allgather"
    fi

    FP16_OPTS=""
    if [ "${FP16}" == "1" ]; then
        FP16_OPTS="--fp16"
    fi

    ZERO_CONFIG_OPTS=(
        --zero_overlap_comm "${ZERO_OVERLAP_COMM}"
    )
    if [ -n "${ZERO_CONTIGUOUS_GRADIENTS}" ]; then
        ZERO_CONFIG_OPTS+=(--zero_contiguous_gradients "${ZERO_CONTIGUOUS_GRADIENTS}")
    fi
    if [ -n "${ZERO_REDUCE_SCATTER}" ]; then
        ZERO_CONFIG_OPTS+=(--zero_reduce_scatter "${ZERO_REDUCE_SCATTER}")
    fi
    if [ -n "${ZERO_ALLGATHER_PARTITIONS}" ]; then
        ZERO_CONFIG_OPTS+=(--zero_allgather_partitions "${ZERO_ALLGATHER_PARTITIONS}")
    fi
    if [ "${ZERO_REDUCE_BUCKET_SIZE}" != "0" ]; then
        ZERO_CONFIG_OPTS+=(--zero_reduce_bucket_size "${ZERO_REDUCE_BUCKET_SIZE}")
    fi
    if [ "${ZERO_ALLGATHER_BUCKET_SIZE}" != "0" ]; then
        ZERO_CONFIG_OPTS+=(--zero_allgather_bucket_size "${ZERO_ALLGATHER_BUCKET_SIZE}")
    fi
    if [ "${ZERO_SUB_GROUP_SIZE}" != "0" ]; then
        ZERO_CONFIG_OPTS+=(--zero_sub_group_size "${ZERO_SUB_GROUP_SIZE}")
    fi
    if [ "${ZERO_STAGE3_PREFETCH_BUCKET_SIZE}" != "0" ]; then
        ZERO_CONFIG_OPTS+=(--zero_stage3_prefetch_bucket_size "${ZERO_STAGE3_PREFETCH_BUCKET_SIZE}")
    fi
    if [ "${ZERO_STAGE3_PARAM_PERSISTENCE_THRESHOLD}" != "-1" ]; then
        ZERO_CONFIG_OPTS+=(--zero_stage3_param_persistence_threshold "${ZERO_STAGE3_PARAM_PERSISTENCE_THRESHOLD}")
    fi
    if [ "${ZERO_STAGE3_MAX_LIVE_PARAMETERS}" != "0" ]; then
        ZERO_CONFIG_OPTS+=(--zero_stage3_max_live_parameters "${ZERO_STAGE3_MAX_LIVE_PARAMETERS}")
    fi
    if [ "${ZERO_STAGE3_MAX_REUSE_DISTANCE}" != "0" ]; then
        ZERO_CONFIG_OPTS+=(--zero_stage3_max_reuse_distance "${ZERO_STAGE3_MAX_REUSE_DISTANCE}")
    fi
    if [ -n "${ZERO_STAGE3_OFFLOAD_PARAM_DEVICE}" ]; then
        ZERO_CONFIG_OPTS+=(
            --zero_stage3_offload_param_device "${ZERO_STAGE3_OFFLOAD_PARAM_DEVICE}"
            --zero_stage3_offload_param_pin_memory "${ZERO_STAGE3_OFFLOAD_PARAM_PIN_MEMORY}"
        )
    fi

    python generate_conf.py \
        --machine_rank ${MACHINE_RANK} \
        --num_machines ${NUM_NODES} \
        --num_processes ${NUM_PROCESSES} \
        --zero_stage ${ZERO_STAGE} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
        ${FP16_OPTS} \
        ${DEEPCOMPILE_OPTS} ${DEBUG_LOG_OPTS} \
        ${SYNC_BEFORE_REDUCE_OPTS} ${SYNC_AFTER_REDUCE_OPTS} \
        ${SYNC_BEFORE_ALLGATHER_OPTS} ${SYNC_AFTER_ALLGATHER_OPTS} \
        "${ZERO_CONFIG_OPTS[@]}" \
        --template_file configs/ds_config.json.template \
        --output_file configs/ds_config.json
fi

#replace , with _ in PASSES
PASSES=$(echo $PASSES | tr ',' '_')
LOG_DIR=logs
mkdir -p ${LOG_DIR}
LOG_FILE=${LOG_DIR}/debug_n${MACHINE_RANK}_${MODEL##*/}_${BACKEND}_np${NUM_PROCESSES}z${ZERO_STAGE}c${COMPILE}dc${DEEPCOMPILE}E${EAGER}b${BATCH_SIZE}seq${SEQ_LENGTH}g${GRADIENT_ACCUMULATION_STEPS}a${ACTIVATION_CHECKPOINTING}p${PASSES}.log
echo "Logging to ${LOG_FILE}"

set +e
accelerate launch --main_process_ip ${HOST_IP} --main_process_port ${MAIN_PROCESS_PORT} \
--num_machines ${NUM_NODES} --num_processes ${NUM_PROCESSES} --machine_rank ${MACHINE_RANK} \
--config_file configs/config.yaml \
verify_loss.py \
--model_name "${MODEL}" \
--zero_stage ${ZERO_STAGE} \
${GAS_OPTS} \
${EXTRA_OPTS} \
2>&1 | tee ${LOG_FILE}
status=${PIPESTATUS[0]}
set -e
exit "${status}"
