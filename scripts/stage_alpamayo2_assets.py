#!/usr/bin/env python3
"""Stage the pinned Alpamayo2-Super benchmark assets onto node-local storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any


MODEL_ID = "nvidia/Alpamayo2-Super"
MODEL_REVISION = "00554695e729a6ff0b6281fd2c81b18d06e33dbe"
MODEL_DIRECTORY = f"alpamayo2-super-{MODEL_REVISION}"
MODEL_INDEX = "model.safetensors.index.json"
MODEL_TOTAL_SIZE = 71_627_868_644
MODEL_SHARD_COUNT = 15
MODEL_RUNTIME_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
)

DATASET_DIRECTORY = "coco-val2017-1000-v1"
DATASET_KIND = "coco_val2017_alpamayo2_vlm_benchmark"
AVAILABLE_IMAGE_COUNT = 5_000
SELECTED_IMAGE_COUNT = 1_000
SAMPLE_COUNT = 1_000
CAMERA_IDS = (0, 1, 2, 3, 5, 6)
FRAMES_PER_CAMERA = 4

SHARED_STORAGE_ROOT = Path("/mnt/cluster_storage")
LOCAL_STORAGE_ROOT = Path("/mnt/local_storage")
LOCAL_FREE_SPACE_MARGIN_BYTES = 5 * 1024**3


def sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"required JSON file is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"required JSONL file is missing or unsafe: {path}")
    records = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RuntimeError(f"expected an object at {path}:{line_number}")
                records.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSONL file {path}: {error}") from error
    return records


def contained_file(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"unsafe asset path in manifest: {relative_path}")
    candidate = root / relative
    if not candidate.is_file() or candidate.is_symlink():
        raise RuntimeError(f"required asset file is missing or unsafe: {candidate}")
    return candidate


def directory_file_sizes(root: Path) -> dict[str, int]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"required asset directory is missing or unsafe: {root}")
    result = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"asset tree contains a symlink: {path}")
        if path.is_file():
            result[str(path.relative_to(root))] = path.stat().st_size
        elif not path.is_dir():
            raise RuntimeError(f"asset tree contains a non-file entry: {path}")
    return result


def validate_model(
    model_dir: Path, manifest: dict[str, Any], verify_hashes: bool
) -> None:
    expected = {
        "schema_version": 1,
        "kind": "huggingface_snapshot",
        "repo_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "safetensors_index_total_size": MODEL_TOTAL_SIZE,
        "shard_count": MODEL_SHARD_COUNT,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise RuntimeError(
                f"model manifest {field} mismatch: expected {value!r}, got {manifest.get(field)!r}"
            )

    index_path = contained_file(model_dir, MODEL_INDEX)
    for runtime_file in MODEL_RUNTIME_FILES:
        contained_file(model_dir, runtime_file)
    index = load_json(index_path)
    indexed_total = index.get("metadata", {}).get("total_size")
    if indexed_total != MODEL_TOTAL_SIZE:
        raise RuntimeError(
            f"model index total_size mismatch: expected {MODEL_TOTAL_SIZE}, got {indexed_total}"
        )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("model index has no weight_map")
    if any(not isinstance(name, str) for name in weight_map.values()):
        raise RuntimeError("model index weight_map contains a non-string shard name")
    shard_names = sorted(set(weight_map.values()))
    if len(shard_names) != MODEL_SHARD_COUNT:
        raise RuntimeError(
            f"model shard count mismatch: expected {MODEL_SHARD_COUNT}, got {len(shard_names)}"
        )

    file_records = manifest.get("files")
    if not isinstance(file_records, list):
        raise RuntimeError("model manifest files must be a list")
    records_by_path = {}
    for record in file_records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise RuntimeError("invalid model manifest file record")
        if record["path"] in records_by_path:
            raise RuntimeError(
                f"duplicate model manifest file record: {record['path']}"
            )
        records_by_path[record["path"]] = record
    expected_files = set(shard_names) | {"config.json", MODEL_INDEX}
    if set(records_by_path) != expected_files:
        raise RuntimeError(
            "model manifest does not record exactly the shards, config, and index"
        )
    for relative_path, record in records_by_path.items():
        path = contained_file(model_dir, relative_path)
        if path.stat().st_size != record.get("size"):
            raise RuntimeError(f"model file size mismatch: {relative_path}")
        expected_hash = record.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise RuntimeError(
                f"model file hash is missing or invalid: {relative_path}"
            )
        if verify_hashes and sha256(path) != expected_hash:
            raise RuntimeError(f"model file hash mismatch after copy: {relative_path}")


def validate_dataset(
    dataset_dir: Path, manifest: dict[str, Any], verify_hashes: bool
) -> None:
    expected = {
        "schema_version": 1,
        "kind": DATASET_KIND,
        "available_image_count": AVAILABLE_IMAGE_COUNT,
        "selected_image_count": SELECTED_IMAGE_COUNT,
        "sample_count": SAMPLE_COUNT,
        "camera_ids": list(CAMERA_IDS),
        "frames_per_camera": FRAMES_PER_CAMERA,
        "global_batch_8_steps_per_epoch": SAMPLE_COUNT // 8,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise RuntimeError(
                f"dataset manifest {field} mismatch: expected {value!r}, got {manifest.get(field)!r}"
            )

    images_dir = dataset_dir / "val2017"
    image_paths = sorted(images_dir.glob("*.jpg")) if images_dir.is_dir() else []
    if len(image_paths) != AVAILABLE_IMAGE_COUNT:
        raise RuntimeError(
            f"available COCO image count mismatch: expected {AVAILABLE_IMAGE_COUNT}, got {len(image_paths)}"
        )
    if any(path.is_symlink() or not path.is_file() for path in image_paths):
        raise RuntimeError("COCO image tree contains an unsafe entry")

    selected_name = Path(str(manifest.get("selected_images_manifest", ""))).name
    samples_name = Path(str(manifest.get("training_samples_manifest", ""))).name
    if selected_name != "selected-images-1000.jsonl":
        raise RuntimeError("unexpected selected-images manifest name")
    if samples_name != "training-samples-1000.jsonl":
        raise RuntimeError("unexpected training-samples manifest name")
    selected_records = load_jsonl(dataset_dir / selected_name)
    sample_records = load_jsonl(dataset_dir / samples_name)
    if [record.get("selection_index") for record in selected_records] != list(
        range(SELECTED_IMAGE_COUNT)
    ):
        raise RuntimeError("selected-images manifest indices/count are incomplete")
    if [record.get("sample_index") for record in sample_records] != list(
        range(SAMPLE_COUNT)
    ):
        raise RuntimeError("training-samples manifest indices/count are incomplete")
    sample_digests = {
        hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        for record in sample_records
    }
    if len(sample_digests) != SAMPLE_COUNT:
        raise RuntimeError("training-samples manifest records are not distinct")

    selected_paths = set()
    for record in selected_records:
        relative_path = record.get("relative_path")
        if not isinstance(relative_path, str):
            raise RuntimeError("selected COCO record has no relative_path")
        path = contained_file(dataset_dir, relative_path)
        selected_paths.add(relative_path)
        expected_hash = record.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise RuntimeError(f"selected COCO image hash is invalid: {relative_path}")
        if verify_hashes and sha256(path) != expected_hash:
            raise RuntimeError(
                f"selected COCO image hash mismatch after copy: {relative_path}"
            )
    if len(selected_paths) != SELECTED_IMAGE_COUNT:
        raise RuntimeError("selected-images manifest contains duplicate image paths")

    for expected_index, sample in enumerate(sample_records):
        cameras = sample.get("cameras")
        if not isinstance(cameras, list) or [
            camera.get("camera_id") for camera in cameras
        ] != list(CAMERA_IDS):
            raise RuntimeError(
                f"training sample {expected_index} has the wrong camera mapping"
            )
        for camera in cameras:
            relative_path = camera.get("relative_path")
            frames = camera.get("frames")
            if relative_path not in selected_paths:
                raise RuntimeError(
                    f"training sample {expected_index} references an unselected image"
                )
            if frames != [relative_path] * FRAMES_PER_CAMERA:
                raise RuntimeError(
                    f"training sample {expected_index} has the wrong frame mapping"
                )
        trajectory = sample.get("trajectory")
        if not isinstance(sample.get("cot"), str) or not sample["cot"]:
            raise RuntimeError(f"training sample {expected_index} has no CoT text")
        if (
            not isinstance(trajectory, dict)
            or trajectory.get("kind") != "deterministic_synthetic_v1"
        ):
            raise RuntimeError(
                f"training sample {expected_index} has the wrong trajectory kind"
            )
        if trajectory.get("seed") != expected_index:
            raise RuntimeError(
                f"training sample {expected_index} has the wrong trajectory seed"
            )
        if (trajectory.get("history_points"), trajectory.get("future_points")) != (
            16,
            64,
        ):
            raise RuntimeError(
                f"training sample {expected_index} has the wrong trajectory shape"
            )
        if any(
            not isinstance(trajectory.get(field), (int, float))
            for field in ("speed_mps", "lateral_amplitude_m", "yaw_rate_rad_s")
        ):
            raise RuntimeError(
                f"training sample {expected_index} has invalid trajectory values"
            )

    sources = manifest.get("sources")
    if not isinstance(sources, list) or len(sources) != 2:
        raise RuntimeError("dataset manifest must record both COCO source archives")
    expected_archives = {"val2017.zip", "annotations_trainval2017.zip"}
    if {
        Path(str(record.get("path", ""))).name for record in sources
    } != expected_archives:
        raise RuntimeError("dataset manifest records unexpected source archives")
    for record in sources:
        archive_name = Path(str(record["path"])).name
        path = contained_file(dataset_dir, f"raw/{archive_name}")
        if path.stat().st_size != record.get("size"):
            raise RuntimeError(f"dataset archive size mismatch: {archive_name}")
        expected_hash = record.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise RuntimeError(f"dataset archive hash is invalid: {archive_name}")
        if verify_hashes and sha256(path) != expected_hash:
            raise RuntimeError(
                f"dataset archive hash mismatch after copy: {archive_name}"
            )


def validate_prepared_root(
    root: Path, *, verify_hashes: bool, require_declared_root: bool
) -> tuple[Path, Path]:
    overall = load_json(root / "asset-preparation-manifest.json")
    if overall.get("schema_version") != 1 or overall.get("status") != "complete":
        raise RuntimeError(
            "overall asset-preparation manifest is missing or incomplete"
        )
    if (
        require_declared_root
        and Path(str(overall.get("cache_root", ""))).resolve() != root.resolve()
    ):
        raise RuntimeError("overall asset-preparation manifest cache_root mismatch")

    model_dir = root / "models" / MODEL_DIRECTORY
    dataset_dir = root / "datasets" / DATASET_DIRECTORY
    model_manifest = load_json(model_dir / ".complete.json")
    dataset_manifest = load_json(dataset_dir / "dataset-manifest.json")
    if overall.get("model") != model_manifest:
        raise RuntimeError("overall and model completion manifests disagree")
    if overall.get("dataset") != dataset_manifest:
        raise RuntimeError("overall and dataset manifests disagree")
    if require_declared_root:
        declared_paths = {
            "model": (model_manifest.get("path"), model_dir),
            "dataset": (dataset_manifest.get("path"), dataset_dir),
            "selected images": (
                dataset_manifest.get("selected_images_manifest"),
                dataset_dir / "selected-images-1000.jsonl",
            ),
            "training samples": (
                dataset_manifest.get("training_samples_manifest"),
                dataset_dir / "training-samples-1000.jsonl",
            ),
        }
        for description, (declared, expected_path) in declared_paths.items():
            if Path(str(declared or "")).resolve() != expected_path.resolve():
                raise RuntimeError(f"source manifest {description} path mismatch")
        declared_archives = {
            Path(str(record.get("path", ""))).resolve()
            for record in dataset_manifest.get("sources", [])
            if isinstance(record, dict)
        }
        expected_archives = {
            (dataset_dir / "raw" / "val2017.zip").resolve(),
            (dataset_dir / "raw" / "annotations_trainval2017.zip").resolve(),
        }
        if declared_archives != expected_archives:
            raise RuntimeError("source manifest archive paths mismatch")
    validate_model(model_dir, model_manifest, verify_hashes)
    validate_dataset(dataset_dir, dataset_manifest, verify_hashes)
    return model_dir, dataset_dir


def require_storage_location(path: Path, storage_root: Path, description: str) -> Path:
    resolved = path.resolve()
    resolved_storage = storage_root.resolve()
    if resolved != resolved_storage and resolved_storage not in resolved.parents:
        raise RuntimeError(
            f"{description} must be under {resolved_storage}: {resolved}"
        )
    return resolved


def stage_prepared_assets(shared_asset_root: Path, local_root: Path) -> dict[str, Any]:
    shared_asset_root = require_storage_location(
        shared_asset_root, SHARED_STORAGE_ROOT, "shared asset root"
    )
    local_root = require_storage_location(
        local_root, LOCAL_STORAGE_ROOT, "local stage root"
    )
    if local_root.exists():
        raise RuntimeError(
            f"local stage root already exists; use a unique run path: {local_root}"
        )

    source_model, source_dataset = validate_prepared_root(
        shared_asset_root, verify_hashes=False, require_declared_root=True
    )
    source_model_sizes = directory_file_sizes(source_model)
    source_dataset_sizes = directory_file_sizes(source_dataset)
    required_bytes = sum(
        (
            sum(source_model_sizes.values()),
            sum(source_dataset_sizes.values()),
            (shared_asset_root / "asset-preparation-manifest.json").stat().st_size,
        )
    )
    local_root.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(local_root.parent).free
    minimum_free = required_bytes + LOCAL_FREE_SPACE_MARGIN_BYTES
    if free_bytes < minimum_free:
        raise RuntimeError(
            "insufficient local free space: "
            f"need at least {minimum_free} bytes, found {free_bytes} bytes"
        )

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{local_root.name}.stage-", dir=local_root.parent)
    )
    try:
        shutil.copy2(
            shared_asset_root / "asset-preparation-manifest.json",
            temporary / "asset-preparation-manifest.json",
        )
        (temporary / "models").mkdir()
        (temporary / "datasets").mkdir()
        shutil.copytree(source_model, temporary / "models" / MODEL_DIRECTORY)
        shutil.copytree(source_dataset, temporary / "datasets" / DATASET_DIRECTORY)

        copied_model = temporary / "models" / MODEL_DIRECTORY
        copied_dataset = temporary / "datasets" / DATASET_DIRECTORY
        if directory_file_sizes(copied_model) != source_model_sizes:
            raise RuntimeError("model copy is partial or has unexpected files")
        if directory_file_sizes(copied_dataset) != source_dataset_sizes:
            raise RuntimeError("dataset copy is partial or has unexpected files")
        validate_prepared_root(
            temporary, verify_hashes=True, require_declared_root=False
        )
        temporary.rename(local_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "status": "complete",
        "local_root": str(local_root),
        "model_path": str(local_root / "models" / MODEL_DIRECTORY),
        "dataset_path": str(local_root / "datasets" / DATASET_DIRECTORY),
        "required_bytes": required_bytes,
        "free_bytes_before_copy": free_bytes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-asset-root", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = stage_prepared_assets(args.shared_asset_root, args.local_root)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
