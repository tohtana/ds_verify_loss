import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import alpamayo2_benchmark as benchmark


ROOT = Path(__file__).resolve().parents[1]


def test_identity_and_shape_contract() -> None:
    assert benchmark.MODEL_REVISION == "00554695e729a6ff0b6281fd2c81b18d06e33dbe"
    assert benchmark.OFFICIAL_SOURCE_REVISION == "beb2977d9a7e9d66837d4a3ad5144ff59de37519"
    assert benchmark.CAMERA_IDS == (0, 1, 2, 3, 5, 6)
    trajectory = benchmark.synthetic_trajectory()
    assert trajectory["ego_history_xyz"].shape == (1, 1, 16, 3)
    assert trajectory["ego_history_rot"].shape == (1, 1, 16, 3, 3)
    assert trajectory["ego_future_xyz"].shape == (1, 1, 64, 3)
    assert trajectory["ego_future_rot"].shape == (1, 1, 64, 3, 3)
    assert torch.isfinite(trajectory["ego_future_xyz"]).all()


def test_zero3_precision_contract() -> None:
    config = json.loads((ROOT / "configs/alpamayo2_zero3.json").read_text())
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
    assert "offload_optimizer" not in config["zero_optimization"]
    assert "offload_param" not in config["zero_optimization"]


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
    assert (args.warmup_steps, args.measured_steps) == (1, 3)
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
                "4",
            ]
        )


def test_deepspeed_import_identity_accepts_pinned_short_hash(tmp_path: Path) -> None:
    full_revision = "79046032e5d6800a547348f6b0c7b3e1f112e5ce"
    source_root = tmp_path / "DeepSpeed"
    imported_path = source_root / "deepspeed" / "__init__.py"
    benchmark.validate_deepspeed_import_identity(
        full_revision,
        "79046032",
        source_root,
        "79046032",
        imported_path,
    )
    with pytest.raises(RuntimeError, match="imported DeepSpeed revision mismatch"):
        benchmark.validate_deepspeed_import_identity(
            full_revision,
            "79046032",
            source_root,
            full_revision,
            imported_path,
        )


def test_straggler_critical_summary_uses_global_batch() -> None:
    summary = benchmark.summarize_times([2.0, 1.0, 3.0], world_size=8)
    assert summary["mean_step_seconds"] == 2.0
    assert summary["median_step_seconds"] == 2.0
    assert summary["samples_per_second"] == 4.0


def test_public_launcher_guards_exact_hardware_and_port() -> None:
    launcher = (ROOT / "scripts/run_alpamayo2_benchmark.sh").read_text()
    assert '!= "8"' in launcher
    assert "grep -vq 'H100'" in launcher
    assert "29673" in launcher
    assert "--nnodes=1" in launcher
    assert "--nproc-per-node=8" in launcher
    assert "run_backend fsdp" in launcher
    assert "run_backend deepspeed" in launcher
    assert "DEVDS_REPOS_DIR" not in launcher
    assert 'DEEPSPEED_SOURCE_REPO:?' in launcher
    assert 'pip install --no-deps -e "${DEEPSPEED_SOURCE_REPO}"' in launcher
    assert 'deepspeed.__git_hash__' in launcher
    assert 'validate_deepspeed_import_identity' in launcher
    assert 'if [[ "${FSDP_STATUS}" != "0" || "${DEEPSPEED_STATUS}" != "0" ]]; then' in launcher


def test_effective_backend_policy_is_explicit() -> None:
    source = (ROOT / "alpamayo2_benchmark.py").read_text()
    assert '"sharding_strategy": "FULL_SHARD"' in source
    assert '"wrap_layer": "Qwen3VLTextDecoderLayer"' in source
    assert "torch.autocast(device_type=\"cuda\", dtype=torch.bfloat16)" in source
    assert "model.vlm.requires_grad_(True)" in source
    assert "model.expert.requires_grad_(False)" in source
    assert "not invoked by Alpamayo2Super.forward" in source
    assert "config._name_or_path = model_path" in source
    assert "config.vlm_name_or_path = model_path" in source
    assert source.count("load_bound_config(args.model_path)") == 2
    assert 'effective_gradient_clipping = float(engine.gradient_clipping())' in source
    assert '"gradient_clipping": effective_gradient_clipping' in source
