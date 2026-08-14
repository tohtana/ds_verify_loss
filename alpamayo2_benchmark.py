#!/usr/bin/env python3
"""Alpamayo2-Super single-node throughput benchmark for FSDP and ZeRO-3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import torch


MODEL_ID = "nvidia/Alpamayo2-Super"
MODEL_REVISION = "00554695e729a6ff0b6281fd2c81b18d06e33dbe"
OFFICIAL_SOURCE_REVISION = "beb2977d9a7e9d66837d4a3ad5144ff59de37519"
CAMERA_IDS = (0, 1, 2, 3, 5, 6)
LOCAL_STORAGE_ROOT = Path("/mnt/local_storage")
PREPARED_DATASET_KIND = "coco_val2017_alpamayo2_vlm_benchmark"
WORLD_SIZE = 8
SAMPLE_COUNT = 1_000
TOTAL_STEPS = SAMPLE_COUNT // WORLD_SIZE
WARMUP_STEPS = 1
MEASURED_STEPS = TOTAL_STEPS - WARMUP_STEPS
DEEPCOMPILE_BACKEND = "deepspeed-deepcompile"
DEEPCOMPILE_WARMUP_STEPS = 20
DEEPCOMPILE_MEASURED_STEPS = TOTAL_STEPS - DEEPCOMPILE_WARMUP_STEPS
DEEPCOMPILE_DEFAULT_SCHEDULE = {
    "source": "DeepSpeed init_z3 default schedule at the pinned revision",
    "custom_schedule": False,
    "transition_steps": [0, 5],
    "transitions": [
        {"global_step": 0, "passes": ["z3_gather_release"]},
        {
            "global_step": 5,
            "passes": ["z3_gather_release", "prefetch", "selective_gather"],
        },
    ],
}
BATCH_CACHE_SCHEMA = 1
TRAINING_SCOPE = (
    "vlm_train only; the approximately 32B VLM is trainable; the 2.3B action expert is "
    "disabled, not instantiated, and not loaded"
)
CURRENT_STAGE = "argument_validation"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256_path(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def is_deepspeed_backend(backend: str | None) -> bool:
    return backend in ("deepspeed", DEEPCOMPILE_BACKEND)


def timing_contract(warmup_steps: int, measured_steps: int) -> dict[str, Any]:
    total_steps = warmup_steps + measured_steps
    if warmup_steps <= 0 or measured_steps <= 0 or total_steps != TOTAL_STEPS:
        raise RuntimeError(
            "timing contract must contain positive warmup and measured ranges "
            f"covering exactly {TOTAL_STEPS} optimizer steps"
        )
    return {
        "warmup_steps": warmup_steps,
        "warmup_step_range": [0, warmup_steps - 1],
        "measured_steps": measured_steps,
        "measured_step_range": [warmup_steps, total_steps - 1],
        "total_optimizer_steps": total_steps,
    }


def deepcompile_record(
    args: argparse.Namespace,
    config: dict[str, Any] | None = None,
    active: bool | None = None,
) -> dict[str, Any]:
    if config is None:
        config = getattr(args, "deepcompile_deepspeed_config", None)
    compile_config = config.get("compile") if isinstance(config, dict) else None
    configured = (
        bool(compile_config.get("deepcompile", False))
        if isinstance(compile_config, dict)
        else None
    )
    if active is None:
        active = getattr(args, "deepcompile_active", None)
    requested = getattr(args, "backend", None) == DEEPCOMPILE_BACKEND
    return {
        "requested": requested,
        "configured": configured,
        "active": active,
        "engine_compile_called_after_initialize": getattr(
            args, "deepcompile_engine_compile_called", False
        ),
        "compile_config": compile_config,
        "default_zero3_schedule": (DEEPCOMPILE_DEFAULT_SCHEDULE if requested else None),
        **timing_contract(args.warmup_steps, args.measured_steps),
    }


def identity_record(args: argparse.Namespace) -> dict[str, Any]:
    record = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "official_source_revision": OFFICIAL_SOURCE_REVISION,
        "deepspeed_revision": getattr(args, "deepspeed_revision", None),
    }
    config_path = getattr(args, "deepspeed_config", None)
    if is_deepspeed_backend(getattr(args, "backend", None)) and config_path:
        path = Path(config_path)
        record["deepspeed_config"] = path.name
        if path.is_file():
            record["deepspeed_config_sha256"] = sha256_path(path)
    return record


def validate_deepspeed_import_identity(
    expected_full_revision: str,
    expected_import_revision: str,
    source_root: Path,
    imported_revision: str | None,
    imported_path: Path,
) -> None:
    """Validate the installed package against the pinned clean source checkout."""
    source_root = source_root.resolve()
    imported_path = imported_path.resolve()
    if not expected_full_revision.startswith(expected_import_revision):
        raise RuntimeError(
            "DeepSpeed short revision is not a prefix of the pinned full revision: "
            f"{expected_import_revision} versus {expected_full_revision}"
        )
    if imported_revision != expected_import_revision:
        raise RuntimeError(
            "imported DeepSpeed revision mismatch: "
            f"expected {expected_import_revision}, got {imported_revision}"
        )
    if source_root not in imported_path.parents:
        raise RuntimeError(
            f"DeepSpeed import does not resolve under {source_root}: {imported_path}"
        )


def trajectory_from_prepared_spec(spec: dict[str, Any]) -> dict[str, torch.Tensor]:
    if spec.get("kind") != "deterministic_synthetic_v1":
        raise RuntimeError(f"unsupported prepared trajectory kind: {spec.get('kind')}")
    history_points = int(spec.get("history_points", -1))
    future_points = int(spec.get("future_points", -1))
    if (history_points, future_points) != (16, 64):
        raise RuntimeError(
            f"prepared trajectory shape mismatch: history={history_points}, future={future_points}"
        )
    speed = float(spec["speed_mps"])
    lateral_amplitude = float(spec["lateral_amplitude_m"])
    yaw_rate = float(spec["yaw_rate_rad_s"])

    def poses(times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xyz = torch.stack(
            (
                speed * times,
                lateral_amplitude * torch.sin(times),
                torch.zeros_like(times),
            ),
            dim=-1,
        )
        yaw = yaw_rate * times
        cosine = torch.cos(yaw)
        sine = torch.sin(yaw)
        rotations = torch.zeros((len(times), 3, 3), dtype=torch.float32)
        rotations[:, 0, 0] = cosine
        rotations[:, 0, 1] = -sine
        rotations[:, 1, 0] = sine
        rotations[:, 1, 1] = cosine
        rotations[:, 2, 2] = 1.0
        return xyz, rotations

    history_xyz, history_rot = poses(torch.arange(-15, 1, dtype=torch.float32) / 10)
    future_xyz, future_rot = poses(torch.arange(1, 65, dtype=torch.float32) / 10)
    return {
        "ego_history_xyz": history_xyz.view(1, 1, 16, 3),
        "ego_history_rot": history_rot.view(1, 1, 16, 3, 3),
        "ego_future_xyz": future_xyz.view(1, 1, 64, 3),
        "ego_future_rot": future_rot.view(1, 1, 64, 3, 3),
    }


def load_prepared_records(
    dataset_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    from scripts.stage_alpamayo2_assets import load_json, load_jsonl, validate_dataset

    dataset_dir = dataset_dir.resolve()
    manifest = load_json(dataset_dir / "dataset-manifest.json")
    validate_dataset(dataset_dir, manifest, verify_hashes=False)
    samples_name = Path(manifest["training_samples_manifest"]).name
    selected_name = Path(manifest["selected_images_manifest"]).name
    samples = load_jsonl(dataset_dir / samples_name)
    selected_list = load_jsonl(dataset_dir / selected_name)
    selected = {record["relative_path"]: record for record in selected_list}
    if len(selected) != len(selected_list):
        raise RuntimeError("selected-images manifest contains duplicate paths")
    return manifest, samples, selected


def prepared_coco_data(
    dataset_dir: Path,
    sample_index: int,
    prepared: tuple[list[dict[str, Any]], dict[str, dict[str, Any]]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    from PIL import Image

    dataset_dir = dataset_dir.resolve()
    if prepared is None:
        _, samples, selected_records = load_prepared_records(dataset_dir)
    else:
        samples, selected_records = prepared
    if sample_index < 0 or sample_index >= len(samples):
        raise RuntimeError(f"prepared sample index is out of range: {sample_index}")
    sample = samples[sample_index]
    if sample.get("sample_index") != sample_index:
        raise RuntimeError(
            f"prepared sample order mismatch: expected {sample_index}, "
            f"got {sample.get('sample_index')}"
        )

    camera_frames = []
    image_records = []
    for camera in sample["cameras"]:
        frames = []
        for relative_path in camera["frames"]:
            path = dataset_dir / relative_path
            image = np.asarray(Image.open(path).convert("RGB")).copy()
            frames.append(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        camera_frames.append(frames)
        selected = selected_records[camera["relative_path"]]
        image_records.append(
            {
                "camera_id": camera["camera_id"],
                "image_id": camera["image_id"],
                "relative_path": camera["relative_path"],
                "sha256": selected["sha256"],
            }
        )

    # The processor accepts different source resolutions, but stacking does not.
    height = min(int(frame.shape[-2]) for frames in camera_frames for frame in frames)
    width = min(int(frame.shape[-1]) for frames in camera_frames for frame in frames)
    image_frames = torch.stack(
        [
            torch.stack([frame[..., :height, :width] for frame in frames])
            for frames in camera_frames
        ]
    )
    data = {
        "image_frames": image_frames,
        "camera_indices": torch.tensor(CAMERA_IDS, dtype=torch.int64),
        "cot": sample["cot"],
        **trajectory_from_prepared_spec(sample["trajectory"]),
    }
    provenance = {
        "kind": "prepared_coco_val2017_deterministic_sample",
        "dataset_kind": PREPARED_DATASET_KIND,
        "sample_index": sample_index,
        "source_record_sha256": canonical_json_sha256(sample),
        "images": image_records,
        "trajectory": sample["trajectory"],
        "deviation": (
            "Prepared COCO validation images are repeated across four historical frames; "
            "egomotion and future trajectories use the prepared deterministic descriptor."
        ),
    }
    return data, provenance


def require_local_read_path(path: str | Path, description: str) -> Path:
    resolved = Path(path).resolve()
    local_root = LOCAL_STORAGE_ROOT.resolve()
    if resolved != local_root and local_root not in resolved.parents:
        raise RuntimeError(f"{description} must be under {local_root}: {resolved}")
    return resolved


def load_bound_config(model_path: str) -> Any:
    from alpamayo2_super.config import Alpamayo2SuperConfig

    config = Alpamayo2SuperConfig.from_pretrained(model_path, local_files_only=True)
    # The release config stores empty path fields; official tokenization and model
    # construction resolve their tokenizer/processor through these fields.
    config._name_or_path = model_path
    config.vlm_name_or_path = model_path
    # The training forward only needs the VLM and trajectory tokenizers. Disabling
    # the expert before construction prevents allocating or loading its 2.3B weights.
    config.enable_expert = False
    config.cotrain_expert_vlm = False
    return config


def build_preprocessor(model_path: str) -> tuple[Any, Any]:
    from alpamayo2_super.config import build_alpamayo2_super_tokenizer
    from alpamayo2_super.helper import get_processor

    config = load_bound_config(model_path)
    tokenizer = build_alpamayo2_super_tokenizer(
        model_path, config.history_vocab_size, config.future_vocab_size
    )
    return config, get_processor(tokenizer, config)


def tokenize_batch(data: dict[str, Any], config: Any, processor: Any) -> dict[str, Any]:
    from alpamayo2_super.chat_template.conversation import build_conversation

    messages = build_conversation(
        data=data,
        num_tokens_per_history_traj=config.tokens_per_history_traj,
        num_tokens_per_future_traj=config.tokens_per_future_traj,
        components_order=["image", "traj_history", "prompt", "cot", "traj_future"],
        components_prompt=[],
        generation_mode=False,
        include_camera_ids=config.include_camera_ids,
        camera_ids=data["camera_indices"],
        include_frame_nums=config.frame_label == "frame_num",
    )
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, add_vision_id=False
    )
    images = data["image_frames"].flatten(0, 1).float() / 255.0
    tokenized_data = dict(
        processor(
            text=text,
            images=images,
            videos=None,
            padding=False,
            return_tensors="pt",
            do_rescale=False,
        )
    )
    trajectory_keys = (
        "ego_history_xyz",
        "ego_history_rot",
        "ego_future_xyz",
        "ego_future_rot",
    )
    return {
        "tokenized_data": tokenized_data,
        "traj_data": {key: data[key] for key in trajectory_keys},
    }


def validate_batch_records(records: Any) -> list[dict[str, Any]]:
    if not isinstance(records, list) or len(records) != SAMPLE_COUNT:
        raise RuntimeError(f"batch cache must contain exactly {SAMPLE_COUNT} records")
    if [record.get("sample_index") for record in records] != list(range(SAMPLE_COUNT)):
        raise RuntimeError(
            "batch cache sample records are missing, duplicated, or out of order"
        )
    expected_paths = [f"samples/sample-{index:04d}.pt" for index in range(SAMPLE_COUNT)]
    if [record.get("relative_path") for record in records] != expected_paths:
        raise RuntimeError("batch cache paths are missing, duplicated, or out of order")
    source_digests = [record.get("source_record_sha256") for record in records]
    if any(not isinstance(value, str) or len(value) != 64 for value in source_digests):
        raise RuntimeError("batch cache source record digest is missing or invalid")
    if len(set(source_digests)) != SAMPLE_COUNT:
        raise RuntimeError(
            "batch cache does not represent 1000 distinct source records"
        )
    if any(
        not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64
        for record in records
    ):
        raise RuntimeError("batch cache sample digest is missing or invalid")
    return records


def cache_file(batch_root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"unsafe batch cache path: {relative_path}")
    path = batch_root / relative
    resolved = path.resolve()
    if (
        batch_root.resolve() not in resolved.parents
        or not path.is_file()
        or path.is_symlink()
    ):
        raise RuntimeError(f"batch cache file is missing or unsafe: {path}")
    return path


def load_batch_cache_manifest(batch_root: Path) -> dict[str, Any]:
    manifest_path = batch_root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RuntimeError(
            f"batch cache manifest is missing or unsafe: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": BATCH_CACHE_SCHEMA,
        "dataset_kind": PREPARED_DATASET_KIND,
        "sample_count": SAMPLE_COUNT,
        "world_size": WORLD_SIZE,
        "total_steps": TOTAL_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "measured_steps": MEASURED_STEPS,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise RuntimeError(
                f"batch cache {field} mismatch: expected {value!r}, got {manifest.get(field)!r}"
            )
    records = validate_batch_records(manifest.get("records"))
    for record in records:
        cache_file(batch_root, record["relative_path"])
    return manifest


def load_cached_sample(
    batch_root: Path, record: dict[str, Any], expected_index: int
) -> dict[str, Any]:
    if record.get("sample_index") != expected_index:
        raise RuntimeError(
            f"runtime sample order mismatch: expected {expected_index}, "
            f"got {record.get('sample_index')}"
        )
    path = cache_file(batch_root, record["relative_path"])
    if sha256_path(path) != record["sha256"]:
        raise RuntimeError(f"batch cache hash mismatch for sample {expected_index}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("sample_index") != expected_index:
        raise RuntimeError(
            f"batch cache payload index mismatch for sample {expected_index}"
        )
    provenance = payload.get("provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("sample_index") != expected_index
    ):
        raise RuntimeError(
            f"batch cache provenance mismatch for sample {expected_index}"
        )
    if not isinstance(payload.get("tokenized_data"), dict) or not isinstance(
        payload.get("traj_data"), dict
    ):
        raise RuntimeError(
            f"batch cache payload is incomplete for sample {expected_index}"
        )
    return payload


def prepare_batch(args: argparse.Namespace) -> None:
    global CURRENT_STAGE
    CURRENT_STAGE = "data_preparation"
    model_path = require_local_read_path(args.model_path, "model path")
    dataset_path = require_local_read_path(args.dataset_path, "dataset path")
    batch_root = require_local_read_path(args.batch_cache, "batch cache")
    if batch_root.exists():
        raise RuntimeError(f"batch cache root already exists: {batch_root}")
    batch_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{batch_root.name}.prepare-", dir=batch_root.parent)
    )
    try:
        dataset_manifest, samples, selected = load_prepared_records(dataset_path)
        config, processor = build_preprocessor(str(model_path))
        records = []
        samples_dir = temporary / "samples"
        samples_dir.mkdir()
        for sample_index in range(SAMPLE_COUNT):
            data, provenance = prepared_coco_data(
                dataset_path, sample_index, prepared=(samples, selected)
            )
            if tuple(data["image_frames"].shape[:2]) != (6, 4):
                raise RuntimeError(
                    f"sample {sample_index} does not contain six cameras x four frames"
                )
            payload = tokenize_batch(data, config, processor)
            payload["sample_index"] = sample_index
            payload["provenance"] = {
                **provenance,
                "camera_ids": list(CAMERA_IDS),
                "frames_per_camera": 4,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "official_source_revision": OFFICIAL_SOURCE_REVISION,
                "preparation_outside_timed_window": True,
            }
            relative_path = f"samples/sample-{sample_index:04d}.pt"
            sample_path = temporary / relative_path
            torch.save(payload, sample_path)
            records.append(
                {
                    "sample_index": sample_index,
                    "relative_path": relative_path,
                    "size": sample_path.stat().st_size,
                    "sha256": sha256_path(sample_path),
                    "source_record_sha256": provenance["source_record_sha256"],
                }
            )
        validate_batch_records(records)
        manifest = {
            "schema_version": BATCH_CACHE_SCHEMA,
            "dataset_kind": PREPARED_DATASET_KIND,
            "dataset_manifest_sha256": sha256_path(
                dataset_path / "dataset-manifest.json"
            ),
            "sample_count": SAMPLE_COUNT,
            "world_size": WORLD_SIZE,
            "total_steps": TOTAL_STEPS,
            "warmup_steps": WARMUP_STEPS,
            "measured_steps": MEASURED_STEPS,
            "sample_order": "rank r at global step s consumes sample 8*s+r",
            "records": records,
        }
        atomic_json(temporary / "manifest.json", manifest)
        load_batch_cache_manifest(temporary)
        temporary.rename(batch_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    manifest_path = batch_root / "manifest.json"
    print(
        json.dumps(
            {
                "batch_cache": str(batch_root),
                "manifest_sha256": sha256_path(manifest_path),
                "sample_count": SAMPLE_COUNT,
                "dataset_sample_count": dataset_manifest["sample_count"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def distributed_context() -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 8 or torch.cuda.device_count() != 8:
        raise RuntimeError(
            f"exact benchmark contract requires world_size=8 and 8 visible GPUs; "
            f"got world_size={world_size}, visible_gpus={torch.cuda.device_count()}"
        )
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def logical_parameter_numel(parameter: torch.nn.Parameter) -> int:
    """Count an unpartitioned parameter or its ZeRO-3 logical shape."""
    zero3_numel = getattr(parameter, "ds_numel", None)
    return int(zero3_numel) if zero3_numel is not None else parameter.numel()


def set_training_scope(model: torch.nn.Module) -> tuple[int, int]:
    if not hasattr(model, "vlm"):
        raise RuntimeError("checkpoint wrapper does not expose the VLM")
    if hasattr(model, "expert"):
        raise RuntimeError("vlm_train must not instantiate or load the action expert")
    model.vlm.requires_grad_(True)
    total = sum(logical_parameter_numel(parameter) for parameter in model.parameters())
    trainable = sum(
        logical_parameter_numel(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if total <= 0 or trainable != total:
        raise RuntimeError(
            f"vlm_train parameter scope mismatch: trainable={trainable}, total={total}"
        )
    return trainable, total


def cast_trajectory_to_float32(value: Any) -> Any:
    """Keep the official trajectory tokenizers on their required FP32 path."""
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and value.dtype != torch.float32:
            return value.to(dtype=torch.float32)
        return value
    if isinstance(value, dict):
        return {key: cast_trajectory_to_float32(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cast_trajectory_to_float32(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cast_trajectory_to_float32(item) for item in value)
    return value


def fp32_trajectory_forward_pre_hook(
    module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Undo FSDP's root-input BF16 cast only for trajectory tokenization.

    Alpamayo's action-space utilities explicitly disable CUDA autocast because
    their Cholesky solve requires FP32. Root FSDP mixed precision casts keyword
    inputs before invoking the wrapped module, so this hook runs inside FSDP and
    restores only ``traj_data``. The VLM forward remains under BF16 autocast.
    """
    del module
    if "traj_data" not in kwargs:
        raise RuntimeError("Alpamayo forward requires traj_data as a keyword input")
    updated = dict(kwargs)
    updated["traj_data"] = cast_trajectory_to_float32(kwargs["traj_data"])
    return args, updated


def install_fp32_trajectory_input_hook(model: torch.nn.Module) -> None:
    model.register_forward_pre_hook(fp32_trajectory_forward_pre_hook, with_kwargs=True)


def load_fsdp(
    args: argparse.Namespace, rank: int, device: torch.device
) -> tuple[Any, Any, dict]:
    from accelerate import init_empty_weights
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer
    import functools

    config = load_bound_config(args.model_path)
    config.vlm_config._attn_implementation = "sdpa"
    if rank == 0:
        model = Alpamayo2Super.from_pretrained(
            args.model_path,
            config=config,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
    else:
        with init_empty_weights():
            model = Alpamayo2Super(config)
    trainable, total = set_training_scope(model)
    install_fp32_trajectory_input_hook(model)
    model.vlm.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls={Qwen3VLTextDecoderLayer}
    )
    mixed = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed,
        device_id=device,
        use_orig_params=True,
        sync_module_states=True,
        param_init_fn=lambda module: module.to_empty(device=device, recurse=False),
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    effective = {
        "backend": "fsdp",
        "sharding_strategy": "FULL_SHARD",
        "wrap_layer": "Qwen3VLTextDecoderLayer",
        "mixed_precision": "BF16 parameters/reductions/buffers",
        "autocast_dtype": "bfloat16",
        "trajectory_tokenization_dtype": "float32",
        "activation_checkpointing": True,
        "activation_checkpointing_use_reentrant": False,
        "use_orig_params": True,
        "gradient_clearing": "optimizer.zero_grad(set_to_none=True)",
        "vlm_train_only": True,
        "action_expert_loaded": False,
        "trainable_params": trainable,
        "total_params": total,
    }
    return model, optimizer, effective


def load_deepspeed(args: argparse.Namespace) -> tuple[Any, Any, dict]:
    global CURRENT_STAGE
    import deepspeed
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super
    from transformers.integrations import HfDeepSpeedConfig

    deepspeed.init_distributed(dist_backend="nccl")
    deepspeed_world_size = deepspeed.comm.get_world_size()
    if deepspeed_world_size != WORLD_SIZE:
        raise RuntimeError(
            "DeepSpeed communication world size must be initialized before ZeRO-3 "
            f"model construction; got {deepspeed_world_size}, expected {WORLD_SIZE}"
        )
    config = json.loads(Path(args.deepspeed_config).read_text(encoding="utf-8"))
    args.deepcompile_deepspeed_config = config
    args.deepcompile_engine_compile_called = False
    args.deepcompile_active = False
    requested = args.backend == DEEPCOMPILE_BACKEND
    compile_config = config.get("compile", {})
    if not isinstance(compile_config, dict):
        raise RuntimeError("DeepSpeed compile config must be a JSON object")
    configured = bool(compile_config.get("deepcompile", False))
    if configured != requested:
        raise RuntimeError(
            "DeepCompile lane/config mismatch: "
            f"requested={requested}, configured={configured}"
        )
    hf_zero3 = HfDeepSpeedConfig(config)
    model_config = load_bound_config(args.model_path)
    model_config.vlm_config._attn_implementation = "sdpa"
    model = Alpamayo2Super.from_pretrained(
        args.model_path,
        config=model_config,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    trainable, total = set_training_scope(model)
    install_fp32_trajectory_input_hook(model)
    model.vlm.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=config,
    )
    if requested:
        CURRENT_STAGE = "deepcompile_engine_compile"
        args.deepcompile_engine_compile_called = True
        engine.compile()
        CURRENT_STAGE = "deepcompile_activation_validation"
        args.deepcompile_active = bool(engine.is_deepcompile_active())
        if not args.deepcompile_active:
            raise RuntimeError(
                "DeepCompile was requested but is not active after engine.compile()"
            )
    elif bool(engine.is_deepcompile_active()):
        raise RuntimeError("ordinary ZeRO-3 unexpectedly activated DeepCompile")
    CURRENT_STAGE = "model_and_backend_initialization"
    effective_gradient_clipping = float(engine.gradient_clipping())
    if effective_gradient_clipping != 0.0:
        raise RuntimeError(
            "DeepSpeed gradient clipping must be disabled to match FSDP; "
            f"got {effective_gradient_clipping}"
        )
    effective = {
        "backend": args.backend,
        "zero_stage": config["zero_optimization"]["stage"],
        "gradient_clipping": effective_gradient_clipping,
        "bf16": config["bf16"],
        "torch_autocast": config["torch_autocast"],
        "activation_checkpointing": True,
        "activation_checkpointing_use_reentrant": False,
        "autocast_dtype": "bfloat16",
        "trajectory_tokenization_dtype": "float32",
        "deepspeed_world_size_at_zero3_construction": deepspeed_world_size,
        "gradient_clearing": "DeepSpeedEngine.step",
        "vlm_train_only": True,
        "action_expert_loaded": False,
        "trainable_params": trainable,
        "total_params": total,
        "deepcompile": deepcompile_record(
            args, config=config, active=args.deepcompile_active
        ),
    }
    del hf_zero3
    return engine, optimizer, effective


def move_tensors(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=False)
    if isinstance(value, dict):
        return {key: move_tensors(item, device) for key, item in value.items()}
    return value


def software_record() -> dict[str, Any]:
    record = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
    }
    try:
        record["deepspeed"] = __import__("deepspeed").__version__
    except ImportError:
        record["deepspeed"] = None
    try:
        record["nvidia_smi"] = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
            )
            .strip()
            .splitlines()
        )
    except Exception as error:
        record["nvidia_smi_error"] = str(error)
    return record


def summarize_times(step_times: list[float], world_size: int) -> dict[str, float]:
    mean = statistics.fmean(step_times)
    return {
        "mean_step_seconds": mean,
        "median_step_seconds": statistics.median(step_times),
        "samples_per_second": world_size / mean,
    }


def finite_loss_flag(loss: torch.Tensor) -> torch.Tensor:
    """Return a scalar integer flag suitable for a distributed MIN reduction."""
    return torch.isfinite(loss.detach()).all().to(dtype=torch.int32)


def sample_index_for(global_step: int, rank: int, world_size: int = WORLD_SIZE) -> int:
    if world_size != WORLD_SIZE:
        raise RuntimeError(
            f"sample mapping requires world_size={WORLD_SIZE}, got {world_size}"
        )
    if global_step < 0 or global_step >= TOTAL_STEPS:
        raise RuntimeError(f"global step is out of range: {global_step}")
    if rank < 0 or rank >= world_size:
        raise RuntimeError(f"rank is out of range: {rank}")
    return world_size * global_step + rank


def validate_consumed_samples(
    consumed_by_rank: list[list[int]], world_size: int
) -> list[int]:
    if world_size != WORLD_SIZE or len(consumed_by_rank) != WORLD_SIZE:
        raise RuntimeError("sample consumption proof requires exactly eight ranks")
    if any(len(indices) != TOTAL_STEPS for indices in consumed_by_rank):
        raise RuntimeError("a rank did not consume exactly 125 samples")
    global_order = [
        consumed_by_rank[rank][step]
        for step in range(TOTAL_STEPS)
        for rank in range(world_size)
    ]
    if global_order != list(range(SAMPLE_COUNT)):
        raise RuntimeError("sample consumption is missing, duplicated, or out of order")
    return global_order


def run_benchmark(args: argparse.Namespace) -> None:
    global CURRENT_STAGE
    rank, world_size, _, device = distributed_context()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    CURRENT_STAGE = "batch_cache_manifest_load"
    model_path = require_local_read_path(args.model_path, "model path")
    batch_root = require_local_read_path(args.batch_cache, "batch cache")
    args.model_path = str(model_path)
    args.batch_cache = str(batch_root)
    batch_manifest = load_batch_cache_manifest(batch_root)
    batch_manifest_sha256 = sha256_path(batch_root / "manifest.json")
    batch_records = batch_manifest["records"]
    CURRENT_STAGE = "model_and_backend_initialization"
    if args.backend == "fsdp":
        model, optimizer, effective = load_fsdp(args, rank, device)
    else:
        model, optimizer, effective = load_deepspeed(args)
    ranges = timing_contract(args.warmup_steps, args.measured_steps)
    effective.update(
        {
            "world_size": world_size,
            "micro_batch_per_rank": 1,
            "gradient_accumulation_steps": 1,
            "optimizer": "torch.optim.AdamW",
            "learning_rate": args.learning_rate,
            "attention": "sdpa",
            "training_scope": TRAINING_SCOPE,
            "sample_count": SAMPLE_COUNT,
            "total_optimizer_steps": TOTAL_STEPS,
            "warmup_step_index": 0,
            "warmup_step_range": ranges["warmup_step_range"],
            "measured_step_range": ranges["measured_step_range"],
            "batch_cache_manifest_sha256": batch_manifest_sha256,
        }
    )
    if rank == 0:
        print("EFFECTIVE_CONFIG " + json.dumps(effective, sort_keys=True), flush=True)
    model.train()
    measured_times: list[float] = []
    loss_observations: list[dict[str, Any]] = []
    consumed_samples: list[int] = []
    total_steps = args.warmup_steps + args.measured_steps
    for index in range(total_steps):
        sample_index = sample_index_for(index, rank, world_size)
        CURRENT_STAGE = f"sample_{sample_index}_cache_load_and_device_transfer"
        cache_error = None
        try:
            batch = load_cached_sample(
                batch_root, batch_records[sample_index], sample_index
            )
            inputs = move_tensors(
                {
                    "tokenized_data": batch["tokenized_data"],
                    "traj_data": batch["traj_data"],
                },
                device,
            )
            del batch
        except Exception as error:
            cache_error = f"{type(error).__name__}: {error}"
            inputs = None
        cache_ready = torch.tensor(
            cache_error is None, dtype=torch.int32, device=device
        )
        torch.distributed.all_reduce(cache_ready, op=torch.distributed.ReduceOp.MIN)
        if int(cache_ready.item()) != 1:
            cache_errors: list[str | None] = [None] * world_size
            torch.distributed.all_gather_object(cache_errors, cache_error)
            raise RuntimeError(
                f"sample cache preparation failed across ranks: {cache_errors}"
            )
        del cache_ready
        consumed_samples.append(sample_index)
        warmup = index < args.warmup_steps
        CURRENT_STAGE = "warmup_train_step" if warmup else "measured_train_step"
        torch.distributed.barrier()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(**inputs).loss
        if is_deepspeed_backend(args.backend):
            model.backward(loss)
            model.step()
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        local_elapsed = torch.tensor(
            time.perf_counter() - start, dtype=torch.float64, device=device
        )
        torch.distributed.all_reduce(local_elapsed, op=torch.distributed.ReduceOp.MAX)
        loss_scalar = loss.detach().float()
        if loss_scalar.numel() != 1:
            raise RuntimeError(
                f"expected a scalar loss, got shape {tuple(loss_scalar.shape)}"
            )
        loss_finite = finite_loss_flag(loss_scalar)
        torch.distributed.all_reduce(loss_finite, op=torch.distributed.ReduceOp.MIN)
        if int(loss_finite.item()) != 1:
            raise FloatingPointError("non-finite loss observed on at least one rank")
        global_mean_loss = loss_scalar.clone()
        torch.distributed.all_reduce(
            global_mean_loss, op=torch.distributed.ReduceOp.SUM
        )
        global_mean_loss /= world_size
        elapsed = float(local_elapsed.item())
        observation = {
            "global_step": index,
            "warmup": warmup,
            "global_sample_indices": [
                WORLD_SIZE * index,
                WORLD_SIZE * index + WORLD_SIZE - 1,
            ],
            "global_mean_loss": float(global_mean_loss.item()),
        }
        loss_observations.append(observation)
        if index == args.warmup_steps - 1:
            torch.cuda.reset_peak_memory_stats(device)
        elif not warmup:
            measured_times.append(elapsed)
        if rank == 0:
            print(
                f"TRAIN_STEP global_step={index} warmup={str(warmup).lower()} "
                f"samples={WORLD_SIZE * index}-{WORLD_SIZE * index + WORLD_SIZE - 1} "
                f"max_rank_seconds={elapsed:.6f} "
                f"global_mean_loss={observation['global_mean_loss']:.8g}",
                flush=True,
            )
        del global_mean_loss, inputs, local_elapsed, loss, loss_finite, loss_scalar
    consumed_by_rank: list[list[int] | None] = [None] * world_size
    torch.distributed.all_gather_object(consumed_by_rank, consumed_samples)
    if any(indices is None for indices in consumed_by_rank):
        raise RuntimeError("sample consumption proof is incomplete")
    global_sample_order = validate_consumed_samples(consumed_by_rank, world_size)
    memory = {
        "rank": rank,
        "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    memories: list[dict[str, int] | None] = [None] * world_size
    torch.distributed.all_gather_object(memories, memory)
    if rank == 0:
        result = {
            "status": "success",
            "backend": args.backend,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "official_source_revision": OFFICIAL_SOURCE_REVISION,
            "deepspeed_revision": args.deepspeed_revision,
            "identities": identity_record(args),
            "effective_config": effective,
            "deepcompile": deepcompile_record(args),
            "data": {
                "kind": PREPARED_DATASET_KIND,
                "sample_count": SAMPLE_COUNT,
                "distinct_sample_records": SAMPLE_COUNT,
                "batch_cache_manifest_sha256": batch_manifest_sha256,
                "sample_mapping": "rank r at global step s consumes sample 8*s+r",
                "sample_order_sha256": canonical_json_sha256(global_sample_order),
                "sample_indices_by_rank": consumed_by_rank,
            },
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.measured_steps,
            "total_optimizer_steps": total_steps,
            "warmup_step_index": 0,
            "warmup_step_range": ranges["warmup_step_range"],
            "measured_step_indices": ranges["measured_step_range"],
            "straggler_critical_step_seconds": measured_times,
            **summarize_times(measured_times, world_size),
            "loss_observations_not_training_quality_evaluation": loss_observations,
            "memory_per_rank": memories,
            "global_max_allocated_bytes": max(
                item["max_allocated_bytes"] for item in memories
            ),
            "global_max_reserved_bytes": max(
                item["max_reserved_bytes"] for item in memories
            ),
            "software": software_record(),
        }
        atomic_json(output_dir / f"{args.backend}.json", result)
        print("BENCHMARK_RESULT " + json.dumps(result, sort_keys=True), flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-batch", action="store_true")
    parser.add_argument("--backend", choices=("fsdp", "deepspeed", DEEPCOMPILE_BACKEND))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path")
    parser.add_argument("--batch-cache", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--deepspeed-config")
    parser.add_argument("--deepspeed-revision", default="unknown")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measured-steps", type=int, default=MEASURED_STEPS)
    args = parser.parse_args(argv)
    if args.backend == DEEPCOMPILE_BACKEND and not args.deepspeed_config:
        args.deepspeed_config = "configs/alpamayo2_zero3_deepcompile.json"
    elif args.backend == "deepspeed" and not args.deepspeed_config:
        args.deepspeed_config = "configs/alpamayo2_zero3.json"
    if args.prepare_batch:
        if args.backend or args.output_dir:
            parser.error(
                "--prepare-batch cannot be combined with --backend/--output-dir"
            )
        if not args.dataset_path:
            parser.error("--prepare-batch requires --dataset-path")
    elif not args.backend or not args.output_dir:
        parser.error("benchmark mode requires --backend and --output-dir")
    elif args.dataset_path:
        parser.error(
            "benchmark mode does not read --dataset-path; it uses the cached batch"
        )
    expected = (
        (DEEPCOMPILE_WARMUP_STEPS, DEEPCOMPILE_MEASURED_STEPS)
        if args.backend == DEEPCOMPILE_BACKEND
        else (WARMUP_STEPS, MEASURED_STEPS)
    )
    if (args.warmup_steps, args.measured_steps) != expected:
        parser.error(
            f"the {args.backend or 'batch-preparation'} contract requires exactly "
            f"{expected[0]} warmup and {expected[1]} measured steps"
        )
    return args


def main(argv: list[str] | None = None) -> None:
    global CURRENT_STAGE
    args = parse_args(argv)
    try:
        if args.prepare_batch:
            prepare_batch(args)
        else:
            run_benchmark(args)
    except BaseException as error:
        failure = {
            "status": "failure",
            "backend": getattr(args, "backend", None),
            "stage": CURRENT_STAGE,
            "rank": int(os.environ.get("RANK", "0")),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "identities": identity_record(args),
            "deepcompile": deepcompile_record(args),
        }
        output_dir = getattr(args, "output_dir", None)
        if output_dir:
            atomic_json(
                Path(output_dir) / f"{args.backend}-rank{failure['rank']}-failure.json",
                failure,
            )
        print(
            "BENCHMARK_FAILURE " + json.dumps(failure, sort_keys=True),
            file=__import__("sys").stderr,
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
