#!/usr/bin/env python3
"""Alpamayo2-Super single-node throughput benchmark for FSDP and ZeRO-3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Any

import torch


MODEL_ID = "nvidia/Alpamayo2-Super"
MODEL_REVISION = "00554695e729a6ff0b6281fd2c81b18d06e33dbe"
OFFICIAL_SOURCE_REVISION = "beb2977d9a7e9d66837d4a3ad5144ff59de37519"
CAMERA_IDS = (0, 1, 2, 3, 5, 6)
CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US = 5_100_000
COCO_URLS = (
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "http://images.cocodataset.org/val2017/000000397133.jpg",
    "http://images.cocodataset.org/val2017/000000252219.jpg",
    "http://images.cocodataset.org/val2017/000000087038.jpg",
    "http://images.cocodataset.org/val2017/000000174482.jpg",
    "http://images.cocodataset.org/val2017/000000403385.jpg",
)
TRAINING_SCOPE = (
    "full official checkpoint loaded; 32B VLM trainable; 2.3B diffusion expert frozen and "
    "not invoked by Alpamayo2Super.forward"
)
CURRENT_STAGE = "argument_validation"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synthetic_trajectory() -> dict[str, torch.Tensor]:
    history_x = torch.linspace(-1.5, 0.0, 16)
    future_x = torch.linspace(0.1, 6.4, 64)
    history_xyz = torch.stack((history_x, torch.zeros(16), torch.zeros(16)), dim=-1)
    future_xyz = torch.stack((future_x, torch.zeros(64), torch.zeros(64)), dim=-1)
    return {
        "ego_history_xyz": history_xyz.view(1, 1, 16, 3),
        "ego_history_rot": torch.eye(3).view(1, 1, 1, 3, 3).repeat(1, 1, 16, 1, 1),
        "ego_future_xyz": future_xyz.view(1, 1, 64, 3),
        "ego_future_rot": torch.eye(3).view(1, 1, 1, 3, 3).repeat(1, 1, 64, 1, 1),
    }


def fallback_data(cache_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    from PIL import Image

    image_dir = cache_dir / "coco-val2017"
    image_dir.mkdir(parents=True, exist_ok=True)
    images = []
    records = []
    for url in COCO_URLS:
        path = image_dir / Path(url).name
        if not path.exists():
            request = urllib.request.Request(url, headers={"User-Agent": "ds-verify-loss/benchmark"})
            with urllib.request.urlopen(request, timeout=60) as response:
                path.write_bytes(response.read())
        image = np.asarray(Image.open(path).convert("RGB")).copy()
        tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        images.append(tensor.unsqueeze(0).repeat(4, 1, 1, 1))
        records.append({"url": url, "sha256": sha256_file(path)})

    # The processor accepts different source resolutions, but stacking does not.
    height = min(int(image.shape[-2]) for image in images)
    width = min(int(image.shape[-1]) for image in images)
    images = [image[..., :height, :width] for image in images]
    trajectory = synthetic_trajectory()
    data = {
        "image_frames": torch.stack(images),
        "camera_indices": torch.tensor(CAMERA_IDS, dtype=torch.int64),
        "cot": "The road scene is visible from six cameras. Continue forward cautiously.",
        **trajectory,
    }
    provenance = {
        "kind": "deterministic_public_real_image_fallback",
        "public_sources": records,
        "deviation": (
            "Six public COCO validation images are each repeated across four historical frames; "
            "egomotion and future trajectories are deterministic synthetic tensors."
        ),
    }
    return data, provenance


def official_data() -> tuple[dict[str, Any], dict[str, Any]]:
    from alpamayo2_super.input_profiles import select_task_input
    from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset

    source = load_physical_aiavdataset(CLIP_ID, t0_us=T0_US, num_frames=4)
    data = select_task_input(source, "trajectory")
    data["cot"] = "The road scene is visible from six cameras. Continue forward cautiously."
    return data, {
        "kind": "physical-ai-av-validation-sample",
        "clip_id": CLIP_ID,
        "t0_us": T0_US,
        "synthetic_fields": ["cot"],
    }


def tokenize_batch(data: dict[str, Any], model_path: str) -> dict[str, Any]:
    from alpamayo2_super.chat_template.conversation import build_conversation
    from alpamayo2_super.config import Alpamayo2SuperConfig, build_alpamayo2_super_tokenizer
    from alpamayo2_super.helper import get_processor

    config = Alpamayo2SuperConfig.from_pretrained(model_path, local_files_only=True)
    tokenizer = build_alpamayo2_super_tokenizer(
        model_path, config.history_vocab_size, config.future_vocab_size
    )
    processor = get_processor(tokenizer, config)
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
    trajectory_keys = ("ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot")
    return {
        "tokenized_data": tokenized_data,
        "traj_data": {key: data[key] for key in trajectory_keys},
    }


def prepare_batch(args: argparse.Namespace) -> None:
    global CURRENT_STAGE
    CURRENT_STAGE = "data_preparation"
    cache_dir = Path(args.batch_cache).parent
    try:
        data, provenance = official_data()
    except Exception as error:  # Gated access is optional; the fallback is part of the contract.
        data, provenance = fallback_data(cache_dir)
        provenance["official_sample_error"] = f"{type(error).__name__}: {error}"
    if tuple(data["image_frames"].shape[:2]) != (6, 4):
        raise ValueError(f"expected six cameras x four frames, got {data['image_frames'].shape}")
    payload = tokenize_batch(data, args.model_path)
    payload["provenance"] = {
        **provenance,
        "camera_ids": list(CAMERA_IDS),
        "frames_per_camera": 4,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "official_source_revision": OFFICIAL_SOURCE_REVISION,
        "preparation_outside_timed_window": True,
    }
    batch_path = Path(args.batch_cache)
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, batch_path)
    atomic_json(batch_path.with_suffix(".provenance.json"), payload["provenance"])
    print(json.dumps({"batch_cache": str(batch_path), "provenance": payload["provenance"]}))


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


def set_training_scope(model: torch.nn.Module) -> tuple[int, int]:
    if not hasattr(model, "vlm") or not hasattr(model, "expert"):
        raise RuntimeError("checkpoint must expose both the VLM and diffusion expert")
    model.vlm.requires_grad_(True)
    model.expert.requires_grad_(False)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return trainable, total


def load_fsdp(args: argparse.Namespace, rank: int, device: torch.device) -> tuple[Any, Any, dict]:
    from accelerate import init_empty_weights
    from alpamayo2_super.config import Alpamayo2SuperConfig
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer
    import functools

    if rank == 0:
        model = Alpamayo2Super.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
    else:
        config = Alpamayo2SuperConfig.from_pretrained(args.model_path, local_files_only=True)
        config.vlm_config._attn_implementation = "sdpa"
        with init_empty_weights():
            model = Alpamayo2Super(config)
    trainable, total = set_training_scope(model)
    model.vlm.gradient_checkpointing_enable()
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls={Qwen3VLTextDecoderLayer}
    )
    mixed = MixedPrecision(
        param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16
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
        "use_orig_params": True,
        "trainable_params": trainable,
        "total_params": total,
    }
    return model, optimizer, effective


def load_deepspeed(args: argparse.Namespace) -> tuple[Any, Any, dict]:
    import deepspeed
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super
    from transformers.integrations import HfDeepSpeedConfig

    config = json.loads(Path(args.deepspeed_config).read_text(encoding="utf-8"))
    hf_zero3 = HfDeepSpeedConfig(config)
    model = Alpamayo2Super.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    trainable, total = set_training_scope(model)
    model.vlm.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=config,
    )
    effective = {
        "backend": "deepspeed",
        "zero_stage": config["zero_optimization"]["stage"],
        "bf16": config["bf16"],
        "torch_autocast": config["torch_autocast"],
        "activation_checkpointing": True,
        "autocast_dtype": "bfloat16",
        "trainable_params": trainable,
        "total_params": total,
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
        record["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()
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


def run_benchmark(args: argparse.Namespace) -> None:
    global CURRENT_STAGE
    rank, world_size, _, device = distributed_context()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    CURRENT_STAGE = "cached_batch_load"
    batch = torch.load(args.batch_cache, map_location="cpu", weights_only=False)
    inputs = move_tensors(
        {"tokenized_data": batch["tokenized_data"], "traj_data": batch["traj_data"]}, device
    )
    CURRENT_STAGE = "model_and_backend_initialization"
    if args.backend == "fsdp":
        model, optimizer, effective = load_fsdp(args, rank, device)
    else:
        model, optimizer, effective = load_deepspeed(args)
    effective.update(
        {
            "world_size": world_size,
            "micro_batch_per_rank": 1,
            "gradient_accumulation_steps": 1,
            "optimizer": "torch.optim.AdamW",
            "learning_rate": args.learning_rate,
            "attention": "sdpa",
            "training_scope": TRAINING_SCOPE,
        }
    )
    if rank == 0:
        print("EFFECTIVE_CONFIG " + json.dumps(effective, sort_keys=True), flush=True)
    model.train()
    measured_times = []
    losses = []
    total_steps = args.warmup_steps + args.measured_steps
    CURRENT_STAGE = "warmup_train_step"
    for index in range(total_steps):
        torch.distributed.barrier()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(**inputs)
            loss = output.loss
        if args.backend == "deepspeed":
            model.backward(loss)
            model.step()
            model.zero_grad()
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        local_elapsed = torch.tensor(time.perf_counter() - start, dtype=torch.float64, device=device)
        torch.distributed.all_reduce(local_elapsed, op=torch.distributed.ReduceOp.MAX)
        if index + 1 == args.warmup_steps:
            torch.cuda.reset_peak_memory_stats(device)
            CURRENT_STAGE = "measured_train_step"
        elif index >= args.warmup_steps:
            measured_times.append(float(local_elapsed.item()))
            losses.append(float(loss.detach().float().item()))
            if rank == 0:
                print(
                    f"MEASURED_STEP {index - args.warmup_steps + 1} "
                    f"max_rank_seconds={measured_times[-1]:.6f}",
                    flush=True,
                )
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
            "effective_config": effective,
            "data": batch["provenance"],
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.measured_steps,
            "straggler_critical_step_seconds": measured_times,
            **summarize_times(measured_times, world_size),
            "loss_observations_not_quality_evaluation": losses,
            "memory_per_rank": memories,
            "global_max_allocated_bytes": max(item["max_allocated_bytes"] for item in memories),
            "global_max_reserved_bytes": max(item["max_reserved_bytes"] for item in memories),
            "software": software_record(),
        }
        atomic_json(output_dir / f"{args.backend}.json", result)
        print("BENCHMARK_RESULT " + json.dumps(result, sort_keys=True), flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-batch", action="store_true")
    parser.add_argument("--backend", choices=("fsdp", "deepspeed"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--batch-cache", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--deepspeed-config", default="configs/alpamayo2_zero3.json")
    parser.add_argument("--deepspeed-revision", default="unknown")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measured-steps", type=int, default=3)
    args = parser.parse_args(argv)
    if args.prepare_batch:
        if args.backend or args.output_dir:
            parser.error("--prepare-batch cannot be combined with --backend/--output-dir")
    elif not args.backend or not args.output_dir:
        parser.error("benchmark mode requires --backend and --output-dir")
    if args.warmup_steps != 1 or args.measured_steps != 3:
        parser.error("attempt 0 requires exactly 1 warmup and 3 measured steps")
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
        }
        output_dir = getattr(args, "output_dir", None)
        if output_dir:
            atomic_json(Path(output_dir) / f"{args.backend}-rank{failure['rank']}-failure.json", failure)
        print("BENCHMARK_FAILURE " + json.dumps(failure, sort_keys=True), file=__import__("sys").stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
