import json
import inspect
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import alpamayo2_benchmark as benchmark
from scripts import stage_alpamayo2_assets as staging


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def prepared_asset_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path]:
    from PIL import Image

    shared_mount = tmp_path / "cluster_storage"
    local_mount = tmp_path / "local_storage"
    source_root = shared_mount / "prepared"
    local_root = local_mount / "run-0"
    model_dir = source_root / "models" / staging.MODEL_DIRECTORY
    dataset_dir = source_root / "datasets" / staging.DATASET_DIRECTORY
    model_dir.mkdir(parents=True)
    (dataset_dir / "raw").mkdir(parents=True)
    (dataset_dir / "val2017").mkdir()

    monkeypatch.setattr(staging, "SHARED_STORAGE_ROOT", shared_mount)
    monkeypatch.setattr(staging, "LOCAL_STORAGE_ROOT", local_mount)
    monkeypatch.setattr(staging, "LOCAL_FREE_SPACE_MARGIN_BYTES", 0)
    monkeypatch.setattr(staging, "AVAILABLE_IMAGE_COUNT", 7)
    monkeypatch.setattr(staging, "SELECTED_IMAGE_COUNT", 6)
    monkeypatch.setattr(staging, "SAMPLE_COUNT", 1)
    monkeypatch.setattr(benchmark, "LOCAL_STORAGE_ROOT", local_mount)

    shard_names = []
    for index in range(1, staging.MODEL_SHARD_COUNT + 1):
        name = f"model-{index:05d}-of-{staging.MODEL_SHARD_COUNT:05d}.safetensors"
        (model_dir / name).write_bytes(f"shard-{index}".encode())
        shard_names.append(name)
    write_json(model_dir / "config.json", {"model_type": "alpamayo2_super"})
    write_json(model_dir / "tokenizer.json", {"version": "1.0"})
    write_json(
        model_dir / "tokenizer_config.json", {"tokenizer_class": "Qwen2TokenizerFast"}
    )
    write_json(
        model_dir / "preprocessor_config.json", {"processor_class": "Qwen3VLProcessor"}
    )
    write_json(
        model_dir / staging.MODEL_INDEX,
        {
            "metadata": {"total_size": staging.MODEL_TOTAL_SIZE},
            "weight_map": {
                f"parameter.{index}": name for index, name in enumerate(shard_names)
            },
        },
    )
    model_files = []
    for name in [*shard_names, "config.json", staging.MODEL_INDEX]:
        path = model_dir / name
        model_files.append(
            {"path": name, "size": path.stat().st_size, "sha256": staging.sha256(path)}
        )
    model_manifest = {
        "schema_version": 1,
        "kind": "huggingface_snapshot",
        "repo_id": staging.MODEL_ID,
        "revision": staging.MODEL_REVISION,
        "path": str(model_dir),
        "safetensors_index_total_size": staging.MODEL_TOTAL_SIZE,
        "shard_count": staging.MODEL_SHARD_COUNT,
        "files": model_files,
        "verified_at_unix": 1,
    }
    write_json(model_dir / ".complete.json", model_manifest)

    image_records = []
    for index in range(7):
        filename = f"{index:012d}.jpg"
        image_path = dataset_dir / "val2017" / filename
        Image.new("RGB", (12 + index, 10 + index), (index, index + 1, index + 2)).save(
            image_path
        )
        if index < 6:
            image_records.append(
                {
                    "selection_index": index,
                    "image_id": index,
                    "file_name": filename,
                    "relative_path": f"val2017/{filename}",
                    "width": 12 + index,
                    "height": 10 + index,
                    "sha256": staging.sha256(image_path),
                    "captions": [f"image {index}"],
                }
            )
    selected_path = dataset_dir / "selected-images-1000.jsonl"
    samples_path = dataset_dir / "training-samples-1000.jsonl"
    write_jsonl(selected_path, image_records)
    sample = {
        "sample_index": 0,
        "cameras": [
            {
                "camera_id": camera_id,
                "selection_index": position,
                "image_id": position,
                "relative_path": image_records[position]["relative_path"],
                "frames": [image_records[position]["relative_path"]]
                * staging.FRAMES_PER_CAMERA,
            }
            for position, camera_id in enumerate(staging.CAMERA_IDS)
        ],
        "cot": "deterministic prepared sample",
        "trajectory": {
            "kind": "deterministic_synthetic_v1",
            "seed": 0,
            "history_points": 16,
            "future_points": 64,
            "speed_mps": 3.0,
            "lateral_amplitude_m": -0.16,
            "yaw_rate_rad_s": -0.01,
        },
    }
    write_jsonl(samples_path, [sample])
    sources = []
    for name, payload in (
        ("val2017.zip", b"images"),
        ("annotations_trainval2017.zip", b"annotations"),
    ):
        path = dataset_dir / "raw" / name
        path.write_bytes(payload)
        sources.append(
            {
                "url": f"http://example.invalid/{name}",
                "path": str(path),
                "size": len(payload),
                "sha256": staging.sha256(path),
            }
        )
    dataset_manifest = {
        "schema_version": 1,
        "kind": staging.DATASET_KIND,
        "path": str(dataset_dir),
        "sources": sources,
        "available_image_count": 7,
        "selected_image_count": 6,
        "sample_count": 1,
        "camera_ids": list(staging.CAMERA_IDS),
        "frames_per_camera": staging.FRAMES_PER_CAMERA,
        "selected_images_manifest": str(selected_path),
        "training_samples_manifest": str(samples_path),
        "global_batch_8_steps_per_epoch": 0,
    }
    write_json(dataset_dir / "dataset-manifest.json", dataset_manifest)
    write_json(
        source_root / "asset-preparation-manifest.json",
        {
            "schema_version": 1,
            "status": "complete",
            "cache_root": str(source_root),
            "model": model_manifest,
            "dataset": dataset_manifest,
        },
    )
    return source_root, local_root, local_mount


def test_identity_and_shape_contract() -> None:
    assert benchmark.MODEL_REVISION == "00554695e729a6ff0b6281fd2c81b18d06e33dbe"
    assert (
        benchmark.OFFICIAL_SOURCE_REVISION == "beb2977d9a7e9d66837d4a3ad5144ff59de37519"
    )
    assert benchmark.CAMERA_IDS == (0, 1, 2, 3, 5, 6)


def test_prepared_assets_are_staged_and_sample_reads_locally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, local_root, _ = prepared_asset_fixture(tmp_path, monkeypatch)
    source_hashes = {
        str(path.relative_to(source_root)): staging.sha256(path)
        for path in source_root.rglob("*")
        if path.is_file()
    }
    result = staging.stage_prepared_assets(source_root, local_root)
    assert result["status"] == "complete"
    assert result["local_root"] == str(local_root)
    assert (local_root / "asset-preparation-manifest.json").is_file()
    data, provenance = benchmark.prepared_coco_data(
        local_root / "datasets" / staging.DATASET_DIRECTORY, 0
    )
    assert data["image_frames"].shape == (6, 4, 3, 10, 12)
    assert data["camera_indices"].tolist() == list(staging.CAMERA_IDS)
    assert data["ego_history_xyz"].shape == (1, 1, 16, 3)
    assert data["ego_future_rot"].shape == (1, 1, 64, 3, 3)
    assert provenance["kind"] == "prepared_coco_val2017_deterministic_sample"
    assert provenance["sample_index"] == 0
    assert source_hashes == {
        str(path.relative_to(source_root)): staging.sha256(path)
        for path in source_root.rglob("*")
        if path.is_file()
    }


def test_staging_fails_closed_on_manifest_or_capacity_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, local_root, _ = prepared_asset_fixture(tmp_path, monkeypatch)
    marker_path = source_root / "models" / staging.MODEL_DIRECTORY / ".complete.json"
    overall_path = source_root / "asset-preparation-manifest.json"
    marker = json.loads(marker_path.read_text())
    marker["shard_count"] = staging.MODEL_SHARD_COUNT - 1
    write_json(marker_path, marker)
    overall = json.loads(overall_path.read_text())
    overall["model"] = marker
    write_json(overall_path, overall)
    with pytest.raises(RuntimeError, match="model manifest shard_count mismatch"):
        staging.stage_prepared_assets(source_root, local_root)

    marker["shard_count"] = staging.MODEL_SHARD_COUNT
    marker["files"][0]["sha256"] = "0" * 64
    write_json(marker_path, marker)
    overall["model"] = marker
    write_json(overall_path, overall)
    with pytest.raises(RuntimeError, match="model file hash mismatch after copy"):
        staging.stage_prepared_assets(source_root, local_root)

    marker["files"][0]["sha256"] = staging.sha256(
        source_root / "models" / staging.MODEL_DIRECTORY / marker["files"][0]["path"]
    )
    write_json(marker_path, marker)
    overall["model"] = marker
    write_json(overall_path, overall)
    monkeypatch.setattr(staging.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(RuntimeError, match="insufficient local free space"):
        staging.stage_prepared_assets(source_root, local_root)


def test_preprocessing_writes_an_ordered_local_per_sample_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, local_root, _ = prepared_asset_fixture(tmp_path, monkeypatch)
    staging.stage_prepared_assets(source_root, local_root)
    monkeypatch.setattr(benchmark, "SAMPLE_COUNT", 1)
    monkeypatch.setattr(benchmark, "TOTAL_STEPS", 1)
    monkeypatch.setattr(benchmark, "MEASURED_STEPS", 0)
    monkeypatch.setattr(benchmark, "build_preprocessor", lambda _: (object(), object()))
    monkeypatch.setattr(
        benchmark,
        "tokenize_batch",
        lambda data, config, processor: {
            "tokenized_data": {"input_ids": torch.tensor([[1, 2]])},
            "traj_data": {"ego_history_xyz": data["ego_history_xyz"]},
        },
    )
    batch_root = local_root / "batches" / "fixture"
    args = SimpleNamespace(
        model_path=str(local_root / "models" / staging.MODEL_DIRECTORY),
        dataset_path=str(local_root / "datasets" / staging.DATASET_DIRECTORY),
        batch_cache=str(batch_root),
    )
    benchmark.prepare_batch(args)
    manifest = benchmark.load_batch_cache_manifest(batch_root)
    assert [record["sample_index"] for record in manifest["records"]] == [0]
    payload = benchmark.load_cached_sample(batch_root, manifest["records"][0], 0)
    assert payload["sample_index"] == 0
    assert payload["tokenized_data"]["input_ids"].tolist() == [[1, 2]]


def test_zero3_precision_contract() -> None:
    config = json.loads((ROOT / "configs/alpamayo2_zero3.json").read_text())
    deepcompile = json.loads(
        (ROOT / "configs/alpamayo2_zero3_deepcompile.json").read_text()
    )
    assert config["zero_optimization"]["stage"] == 3
    assert config["train_batch_size"] == 8
    assert config["train_micro_batch_size_per_gpu"] == 1
    assert config["gradient_accumulation_steps"] == 1
    assert config["gradient_clipping"] == 0.0
    assert config["bf16"] == {
        "enabled": True,
        "bf16_master_weights_and_grads": True,
        "bf16_optimizer_states": True,
    }
    assert config["torch_autocast"] == {"enabled": True, "dtype": "bfloat16"}
    assert "steps_per_print" not in config
    assert "offload_optimizer" not in config["zero_optimization"]
    assert "offload_param" not in config["zero_optimization"]
    assert deepcompile.pop("compile") == {"deepcompile": True}
    assert deepcompile == config


def test_attempt_zero_step_counts_are_fixed() -> None:
    args = benchmark.parse_args(
        [
            "--backend",
            "fsdp",
            "--model-path",
            "/checkpoint",
            "--batch-cache",
            "/batch.pt",
            "--output-dir",
            "/output",
        ]
    )
    assert (args.warmup_steps, args.measured_steps) == (1, 124)
    with pytest.raises(SystemExit):
        benchmark.parse_args(
            [
                "--backend",
                "fsdp",
                "--model-path",
                "/checkpoint",
                "--batch-cache",
                "/batch.pt",
                "--output-dir",
                "/output",
                "--measured-steps",
                "123",
            ]
        )
    deepcompile = benchmark.parse_args(
        [
            "--backend",
            "deepspeed-deepcompile",
            "--model-path",
            "/checkpoint",
            "--batch-cache",
            "/batch.pt",
            "--output-dir",
            "/output",
            "--warmup-steps",
            "20",
            "--measured-steps",
            "105",
        ]
    )
    assert (deepcompile.warmup_steps, deepcompile.measured_steps) == (20, 105)
    assert deepcompile.deepspeed_config == ("configs/alpamayo2_zero3_deepcompile.json")
    assert benchmark.timing_contract(20, 105) == {
        "warmup_steps": 20,
        "warmup_step_range": [0, 19],
        "measured_steps": 105,
        "measured_step_range": [20, 124],
        "total_optimizer_steps": 125,
    }
    with pytest.raises(SystemExit):
        benchmark.parse_args(
            [
                "--backend",
                "deepspeed-deepcompile",
                "--model-path",
                "/checkpoint",
                "--batch-cache",
                "/batch.pt",
                "--output-dir",
                "/output",
                "--warmup-steps",
                "6",
                "--measured-steps",
                "119",
            ]
        )


def test_deepspeed_import_identity_accepts_pinned_short_hash(tmp_path: Path) -> None:
    full_revision = "f406908b9281607acaea52a3408162955d320856"
    source_root = tmp_path / "DeepSpeed"
    imported_path = source_root / "deepspeed" / "__init__.py"
    benchmark.validate_deepspeed_import_identity(
        full_revision,
        "f406908b",
        source_root,
        "f406908b",
        imported_path,
    )
    with pytest.raises(RuntimeError, match="imported DeepSpeed revision mismatch"):
        benchmark.validate_deepspeed_import_identity(
            full_revision,
            "f406908b",
            source_root,
            full_revision,
            imported_path,
        )


def test_straggler_critical_summary_uses_global_batch() -> None:
    summary = benchmark.summarize_times([2.0, 1.0, 3.0], world_size=8)
    assert summary["mean_step_seconds"] == 2.0
    assert summary["median_step_seconds"] == 2.0
    assert summary["samples_per_second"] == 4.0


def test_non_finite_loss_flags_fail_closed() -> None:
    assert benchmark.finite_loss_flag(torch.tensor(1.0)).item() == 1
    assert benchmark.finite_loss_flag(torch.tensor(float("nan"))).item() == 0
    assert benchmark.finite_loss_flag(torch.tensor(float("inf"))).item() == 0
    assert benchmark.finite_loss_flag(torch.tensor(float("-inf"))).item() == 0


def test_one_epoch_sample_mapping_is_exact_and_fail_closed() -> None:
    assert benchmark.WORLD_SIZE == 8
    assert benchmark.SAMPLE_COUNT == 1_000
    assert benchmark.TOTAL_STEPS == 125
    assert benchmark.WARMUP_STEPS == 1
    assert benchmark.MEASURED_STEPS == 124
    assert benchmark.DEEPCOMPILE_WARMUP_STEPS == 20
    assert benchmark.DEEPCOMPILE_MEASURED_STEPS == 105
    consumed = [
        [
            benchmark.sample_index_for(step, rank)
            for step in range(benchmark.TOTAL_STEPS)
        ]
        for rank in range(benchmark.WORLD_SIZE)
    ]
    assert benchmark.validate_consumed_samples(consumed, benchmark.WORLD_SIZE) == list(
        range(1000)
    )
    consumed[7][-1] = 998
    with pytest.raises(RuntimeError, match="missing, duplicated, or out of order"):
        benchmark.validate_consumed_samples(consumed, benchmark.WORLD_SIZE)
    with pytest.raises(RuntimeError, match="world_size=8"):
        benchmark.sample_index_for(0, 0, world_size=4)


def test_batch_manifest_records_reject_missing_duplicates_and_reordering() -> None:
    records = [
        {
            "sample_index": index,
            "relative_path": f"samples/sample-{index:04d}.pt",
            "sha256": "a" * 64,
            "source_record_sha256": f"{index:064x}",
        }
        for index in range(benchmark.SAMPLE_COUNT)
    ]
    assert benchmark.validate_batch_records(records) is records
    with pytest.raises(RuntimeError, match="exactly 1000"):
        benchmark.validate_batch_records(records[:-1])
    reordered = records.copy()
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(RuntimeError, match="missing, duplicated, or out of order"):
        benchmark.validate_batch_records(reordered)
    duplicate_source = [dict(record) for record in records]
    duplicate_source[1]["source_record_sha256"] = duplicate_source[0][
        "source_record_sha256"
    ]
    with pytest.raises(RuntimeError, match="1000 distinct source records"):
        benchmark.validate_batch_records(duplicate_source)


def test_vlm_train_scope_rejects_an_instantiated_expert() -> None:
    class VlmOnly(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vlm = torch.nn.Linear(2, 2)

    model = VlmOnly()
    trainable, total = benchmark.set_training_scope(model)
    assert trainable == total == 6
    model.expert = torch.nn.Linear(1, 1)
    with pytest.raises(RuntimeError, match="must not instantiate or load"):
        benchmark.set_training_scope(model)


def test_vlm_train_scope_counts_zero3_logical_parameters() -> None:
    class Zero3Vlm(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(0))
            self.weight.ds_numel = 6

    model = torch.nn.Module()
    model.vlm = Zero3Vlm()
    trainable, total = benchmark.set_training_scope(model)
    assert (trainable, total) == (6, 6)


def test_trajectory_hook_restores_only_trajectory_floats_to_fp32() -> None:
    model = torch.nn.Linear(1, 1)
    args = (torch.tensor([1]),)
    pixel_values = torch.ones(2, dtype=torch.bfloat16)
    integer_ids = torch.tensor([2, 3])
    kwargs = {
        "tokenized_data": {"pixel_values": pixel_values},
        "traj_data": {
            "ego_history_xyz": torch.ones(2, dtype=torch.bfloat16),
            "nested": [torch.ones(1, dtype=torch.float16), integer_ids],
        },
    }
    returned_args, returned_kwargs = benchmark.fp32_trajectory_forward_pre_hook(
        model, args, kwargs
    )
    assert returned_args == args
    assert returned_kwargs["traj_data"]["ego_history_xyz"].dtype == torch.float32
    assert returned_kwargs["traj_data"]["nested"][0].dtype == torch.float32
    assert returned_kwargs["traj_data"]["nested"][1] is integer_ids
    assert returned_kwargs["tokenized_data"]["pixel_values"] is pixel_values
    assert kwargs["traj_data"]["ego_history_xyz"].dtype == torch.bfloat16
    with pytest.raises(RuntimeError, match="requires traj_data"):
        benchmark.fp32_trajectory_forward_pre_hook(model, (), {})


def test_deepspeed_communication_initializes_before_zero3_construction() -> None:
    source = inspect.getsource(benchmark.load_deepspeed)
    init_position = source.index('deepspeed.init_distributed(dist_backend="nccl")')
    world_size_position = source.index("deepspeed.comm.get_world_size()")
    zero3_position = source.index("HfDeepSpeedConfig(config)")
    model_position = source.index("Alpamayo2Super.from_pretrained")
    initialize_position = source.index("deepspeed.initialize(")
    compile_position = source.index("engine.compile()")
    active_position = source.index("engine.is_deepcompile_active()", compile_position)
    assert init_position < world_size_position < zero3_position < model_position
    assert model_position < initialize_position < compile_position < active_position
    assert "deepspeed_world_size != WORLD_SIZE" in source
    assert "engine.compile(schedule" not in source
    assert "DeepCompile was requested but is not active" in source


def test_deepcompile_result_contract_records_default_schedule_and_state() -> None:
    args = benchmark.parse_args(
        [
            "--backend",
            "deepspeed-deepcompile",
            "--model-path",
            "/checkpoint",
            "--batch-cache",
            "/batch.pt",
            "--output-dir",
            "/output",
            "--warmup-steps",
            "20",
            "--measured-steps",
            "105",
        ]
    )
    args.deepcompile_engine_compile_called = True
    record = benchmark.deepcompile_record(
        args, config={"compile": {"deepcompile": True}}, active=True
    )
    assert record["requested"] is True
    assert record["configured"] is True
    assert record["active"] is True
    assert record["compile_config"] == {"deepcompile": True}
    assert record["engine_compile_called_after_initialize"] is True
    assert record["default_zero3_schedule"] == {
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
    assert record["warmup_step_range"] == [0, 19]
    assert record["measured_step_range"] == [20, 124]


def test_public_launcher_guards_exact_hardware_and_port() -> None:
    launcher = (ROOT / "scripts/run_alpamayo2_benchmark.sh").read_text()
    assert '!= "8"' in launcher
    assert "grep -vq 'H100'" in launcher
    assert "29673" in launcher
    assert "--nnodes=1" in launcher
    assert "--nproc-per-node=8" in launcher
    assert "run_backend fsdp" in launcher
    assert "run_backend deepspeed" in launcher
    assert "DEEPSPEED_SOURCE_REPO:?" in launcher
    assert "deepspeed.__git_hash__" in launcher
    assert "validate_deepspeed_import_identity" in launcher
    assert (
        'if [[ "${FSDP_STATUS}" != "0" || "${DEEPSPEED_STATUS}" != "0" ]]; then'
        in launcher
    )
    assert "ALPAMAYO2_LOCAL_STAGE_ROOT:?" in launcher
    assert "DS_VERIFY_LOSS_CANDIDATE_CONTENT_SHA256:?" in launcher
    assert 'LOCAL_ROOT="${ALPAMAYO2_LOCAL_STAGE_ROOT}"' in launcher
    assert "python scripts/stage_alpamayo2_assets.py" in launcher
    assert launcher.index("reviewed harness content digest mismatch") < launcher.index(
        "python scripts/stage_alpamayo2_assets.py"
    )
    assert '--dataset-path "${DATASET_PATH}"' in launcher
    assert 'BATCH_CACHE="${LOCAL_ROOT}/batches/alpamayo2-super-coco1000-v1"' in launcher
    assert 'BENCHMARK_LANE="${ALPAMAYO2_BENCHMARK_LANE:-baseline}"' in launcher
    assert 'run_backend fsdp "" 1 124' in launcher
    assert "run_backend deepspeed configs/alpamayo2_zero3.json 1 124" in launcher
    assert (
        "run_backend deepspeed-deepcompile "
        "configs/alpamayo2_zero3_deepcompile.json 20 105" in launcher
    )
    assert launcher.count('"configs/alpamayo2_zero3_deepcompile.json",') == 2
    assert "HF_HUB_OFFLINE=1" in launcher
    assert 'TORCHINDUCTOR_CACHE_DIR="${LOCAL_ROOT}/torchinductor-cache"' in launcher
    assert "TRANSFORMERS_OFFLINE=1" in launcher
    assert "hf download" not in launcher
    assert 'git clone --quiet --shared "${ALPAMAYO2_SOURCE_REPO}"' in launcher
    assert 'git clone --quiet --shared "${DEEPSPEED_SOURCE_REPO}"' in launcher
    assert 'pip install --no-deps -e "${ALPAMAYO2_RUNTIME_SOURCE}"' in launcher
    assert 'pip install --no-deps -e "${DEEPSPEED_RUNTIME_SOURCE}"' in launcher
    assert 'pip install --no-deps -e "${ALPAMAYO2_SOURCE_REPO}"' not in launcher
    assert 'pip install --no-deps -e "${DEEPSPEED_SOURCE_REPO}"' not in launcher
    after_staging = launcher.split(
        'export HF_HOME="${LOCAL_ROOT}/huggingface"', maxsplit=1
    )[1]
    assert "${ALPAMAYO2_CACHE_ROOT}" not in after_staging


def test_prepare_batch_requires_the_staged_dataset_path() -> None:
    with pytest.raises(SystemExit):
        benchmark.parse_args(
            [
                "--prepare-batch",
                "--model-path",
                "/mnt/local_storage/model",
                "--batch-cache",
                "/mnt/local_storage/batches",
            ]
        )
    args = benchmark.parse_args(
        [
            "--prepare-batch",
            "--model-path",
            "/mnt/local_storage/model",
            "--dataset-path",
            "/mnt/local_storage/dataset",
            "--batch-cache",
            "/mnt/local_storage/batches",
        ]
    )
    assert args.dataset_path == "/mnt/local_storage/dataset"


def test_effective_backend_policy_is_explicit() -> None:
    source = (ROOT / "alpamayo2_benchmark.py").read_text()
    assert '"sharding_strategy": "FULL_SHARD"' in source
    assert '"wrap_layer": "Qwen3VLTextDecoderLayer"' in source
    assert 'torch.autocast(device_type="cuda", dtype=torch.bfloat16)' in source
    assert "model.vlm.requires_grad_(True)" in source
    assert "model.expert.requires_grad_(False)" not in source
    assert "config.enable_expert = False" in source
    assert "not instantiated, and not loaded" in source
    assert "config._name_or_path = model_path" in source
    assert "config.vlm_name_or_path = model_path" in source
    assert source.count("load_bound_config(args.model_path)") == 2
    assert "effective_gradient_clipping = float(engine.gradient_clipping())" in source
    assert '"gradient_clipping": effective_gradient_clipping' in source
    assert "loss = model(**inputs).loss" in source
    assert "output = model(**inputs)" not in source
    assert "model.zero_grad()" not in source
    assert '"gradient_clearing": "DeepSpeedEngine.step"' in source
    assert "del global_mean_loss, inputs, local_elapsed, loss," in source
    assert source.count('"action_expert_loaded": False') == 2
    assert source.count('"activation_checkpointing": True') == 2
    assert source.count('"activation_checkpointing_use_reentrant": False') == 2
    assert source.count('gradient_checkpointing_kwargs={"use_reentrant": False}') == 2
    assert source.count('"trajectory_tokenization_dtype": "float32"') == 2
    assert source.count("install_fp32_trajectory_input_hook(model)") == 2


def test_cache_and_loss_collectives_are_outside_the_train_step_timer() -> None:
    source = inspect.getsource(benchmark.run_benchmark)
    load_position = source.index("load_cached_sample")
    transfer_position = source.index("move_tensors")
    barrier_position = source.index("torch.distributed.barrier()")
    start_position = source.index("start = time.perf_counter()")
    elapsed_position = source.index("local_elapsed = torch.tensor")
    loss_reduce_position = source.index("global_mean_loss = loss_scalar.clone()")
    assert load_position < barrier_position
    assert transfer_position < barrier_position < start_position
    assert elapsed_position < loss_reduce_position
    assert "warmup = index < args.warmup_steps" in source
    assert '"warmup": warmup' in source
    assert "index == args.warmup_steps - 1" in source
    assert "loss_observations.append(observation)" in source
