# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import codex_agent_wrapper
import generate_conf


def _load_verify_loss_with_fake_runtime(monkeypatch):
    accelerator_calls = []

    class FakeAccelerator:
        def __init__(self, **kwargs):
            accelerator_calls.append(kwargs)

    class FakeInitProcessGroupKwargs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_accelerate = ModuleType("accelerate")
    fake_accelerate.Accelerator = FakeAccelerator
    fake_accelerate_utils = ModuleType("accelerate.utils")
    fake_accelerate_utils.InitProcessGroupKwargs = FakeInitProcessGroupKwargs

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoConfig = object()
    fake_transformers.AutoModelForCausalLM = object()
    fake_transformers.enable_full_determinism = lambda *_args, **_kwargs: None
    fake_transformers.set_seed = lambda *_args, **_kwargs: None

    fake_data_utils = ModuleType("data_utils")
    fake_data_utils.get_tokenizer = lambda *_args, **_kwargs: None
    fake_data_utils.load_and_prepare_dataset = lambda *_args, **_kwargs: None

    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "accelerate", fake_accelerate)
    monkeypatch.setitem(sys.modules, "accelerate.utils", fake_accelerate_utils)
    monkeypatch.setitem(sys.modules, "wandb", ModuleType("wandb"))
    monkeypatch.setitem(sys.modules, "data_utils", fake_data_utils)

    module_name = f"verify_loss_test_{id(monkeypatch)}"
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / "verify_loss.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, accelerator_calls


def _fake_codex(tmp_path):
    executable = tmp_path / "fake-codex"
    executable.write_text(
        """#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import sys

request = sys.stdin.read()
prompt_path = None
for line in request.splitlines():
    if line.startswith("PROMPT_PATH="):
        prompt_path = json.loads(line.removeprefix("PROMPT_PATH="))
        break

report = {"selected": "final", "request": request, "prompt_path": prompt_path}
if prompt_path is not None:
    prompt_bytes = Path(prompt_path).read_bytes()
    report["prompt_size"] = len(prompt_bytes)
    report["prompt_sha256"] = hashlib.sha256(prompt_bytes).hexdigest()

print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": '{"selected":"first"}'}}))
print("non-json diagnostic output")
print(json.dumps({"type": "item.completed", "item": {"type": "assistant_message", "text": json.dumps(report)}}))
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def _run_wrapper(tmp_path, stdin):
    env = os.environ.copy()
    env["CODEX_BIN"] = str(_fake_codex(tmp_path))
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "codex_agent_wrapper.py")],
        cwd=REPO_ROOT,
        env=env,
        stdin=stdin,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_launcher_with_fake_runtime(tmp_path, *args):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_python = bin_dir / "python"
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)
    fake_accelerate = bin_dir / "accelerate"
    fake_accelerate.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$LAUNCHER_ARGS_FILE\"\n",
        encoding="utf-8",
    )
    fake_accelerate.chmod(0o755)

    env = os.environ.copy()
    env["NGPUS_PER_NODE"] = "1"
    env["LAUNCHER_ARGS_FILE"] = str(tmp_path / "accelerate-args.txt")
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    proc = subprocess.run(
        ["bash", str(REPO_ROOT / "run.sh"), *args],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    args_file = Path(env["LAUNCHER_ARGS_FILE"])
    launched_args = args_file.read_text(encoding="utf-8").splitlines() if args_file.exists() else []
    return proc, launched_args


def _config_args(tmp_path, strategy="baseline", backend=None):
    return SimpleNamespace(machine_rank=0,
                           num_machines=1,
                           num_processes=8,
                           zero_stage=3,
                           fp16=False,
                           gradient_accumulation_steps=1,
                           deepcompile=True,
                           debug_log=False,
                           sync_before_reduce=False,
                           sync_after_reduce=False,
                           sync_before_allgather=False,
                           sync_after_allgather=False,
                           zero3_tuning_strategy=strategy,
                           agent_backend=backend,
                           agent_architecture="graph_agent",
                           agent_max_iterations=3,
                           agent_max_retries_per_iteration=1,
                           agent_timeout_sec=300,
                           template_file=REPO_ROOT / "configs" / "ds_config.json.template",
                           output_file=tmp_path / "ds_config.json")


def test_generate_config_defaults_to_graph_agent(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["generate_conf.py"])
    assert generate_conf.get_args().agent_architecture == "graph_agent"


def test_baseline_config_retains_known_good_contract_without_candidate_only_keys(tmp_path):
    args = _config_args(tmp_path)
    generate_conf.main(args)

    config = json.loads(args.output_file.read_text(encoding="utf-8"))
    compile_config = config["compile"]

    assert config["bf16"]["bf16_master_weights_and_grads"] is True
    assert config["torch_autocast"] == {"enabled": True, "dtype": "bfloat16"}
    assert compile_config["passes"] == ["z3"]
    assert not any(key.startswith("agent_") for key in compile_config)
    assert "zero3_tuning_strategy" not in compile_config


def test_agent_config_emits_single_graph_agent_command(tmp_path):
    args = _config_args(tmp_path, strategy="agent", backend="codex")
    generate_conf.main(args)

    config = json.loads(args.output_file.read_text(encoding="utf-8"))
    compile_config = config["compile"]
    expected_command = [generate_conf.sys.executable, str((REPO_ROOT / "codex_agent_wrapper.py").resolve())]

    assert compile_config["passes"] == ["z3"]
    assert compile_config["zero3_tuning_strategy"] == "agent"
    assert compile_config["agent_architecture"] == "graph_agent"
    assert compile_config["agent_command"] == expected_command
    assert "agent_evaluator_command" not in compile_config
    assert "agent_optimizer_command" not in compile_config
    assert compile_config["agent_max_iterations"] == 3
    assert compile_config["agent_max_retries_per_iteration"] == 1
    assert compile_config["agent_timeout_sec"] == 300


def test_agent_strategy_requires_backend(tmp_path):
    args = _config_args(tmp_path, strategy="agent")

    try:
        generate_conf.main(args)
    except ValueError as exc:
        assert "--agent_backend is required" in str(exc)
    else:
        raise AssertionError("agent strategy unexpectedly accepted without an agent backend")


def test_codex_wrapper_uses_required_model_reasoning_and_persistent_binary(monkeypatch):
    monkeypatch.delenv("CODEX_BIN", raising=False)
    command = codex_agent_wrapper._build_command()

    assert command[0] == str(codex_agent_wrapper.DEFAULT_CODEX_BIN.resolve())
    assert command[1:5] == ["exec", "--json", "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox"]
    assert command[5:] == ["-m", "gpt-5.6-sol", "-c", 'model_reasoning_effort="xhigh"']


def test_codex_wrapper_discovers_nested_and_shallow_workspace_layouts(tmp_path, monkeypatch):
    nested_root = tmp_path / "nested-workspace"
    nested_wrapper = nested_root / "work" / "qwen-agent-harness" / "client" / "codex_agent_wrapper.py"
    nested_codex = nested_root / "tools" / "codex-cli" / codex_agent_wrapper.CODEX_BINARY_RELATIVE_PATH
    nested_wrapper.parent.mkdir(parents=True)
    nested_codex.parent.mkdir(parents=True)
    nested_codex.write_text("#!/bin/sh\n", encoding="utf-8")
    nested_codex.chmod(0o755)

    monkeypatch.setattr(codex_agent_wrapper, "__file__", str(nested_wrapper))
    assert codex_agent_wrapper._workspace_default_codex_bin() == nested_codex

    shallow_root = tmp_path / "shallow-workspace"
    shallow_wrapper = shallow_root / "work" / "client" / "codex_agent_wrapper.py"
    shallow_codex = shallow_root / "tools" / "codex-cli" / codex_agent_wrapper.CODEX_BINARY_RELATIVE_PATH
    shallow_wrapper.parent.mkdir(parents=True)
    shallow_codex.parent.mkdir(parents=True)
    shallow_codex.write_text("#!/bin/sh\n", encoding="utf-8")
    shallow_codex.chmod(0o755)

    monkeypatch.setattr(codex_agent_wrapper, "__file__", str(shallow_wrapper))
    assert codex_agent_wrapper._workspace_default_codex_bin() == shallow_codex


def test_codex_wrapper_accepts_explicit_executable_override(tmp_path, monkeypatch):
    executable = tmp_path / "codex"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("CODEX_BIN", str(executable))

    assert codex_agent_wrapper._build_command()[0] == str(executable.resolve())


def test_codex_wrapper_extracts_last_assistant_message():
    output = "\n".join([
        json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "first"}}),
        "not json",
        json.dumps({"type": "item.completed", "item": {"type": "assistant_message", "text": '{"ok":true}'}}),
    ])
    assert codex_agent_wrapper._extract_final_text(output) == '{"ok":true}'


def test_codex_wrapper_keeps_small_prompt_inline_and_extracts_final_jsonl(tmp_path):
    prompt = '{"request":"small prompt"}\n'
    env = os.environ.copy()
    env["CODEX_BIN"] = str(_fake_codex(tmp_path))
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "codex_agent_wrapper.py")],
        cwd=REPO_ROOT,
        env=env,
        input=prompt,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    response = json.loads(proc.stdout)
    assert response == {"selected": "final", "request": prompt, "prompt_path": None}


def test_codex_wrapper_hands_oversized_prompt_to_codex_by_absolute_file(tmp_path):
    prompt = b'{"complete_graph":"' + (b"x" * codex_agent_wrapper.CODEX_MAX_INLINE_CHARS) + b'"}\n'
    prompt_file = tmp_path / "graph_agent_prompt.txt"
    prompt_file.write_bytes(prompt)

    with prompt_file.open("r", encoding="utf-8") as prompt_stdin:
        proc = _run_wrapper(tmp_path, prompt_stdin)

    assert proc.returncode == 0, proc.stderr
    response = json.loads(proc.stdout)
    request = response["request"]
    referenced_path = Path(response["prompt_path"])

    assert response["selected"] == "final"
    assert len(request) < codex_agent_wrapper.CODEX_MAX_INLINE_CHARS
    assert prompt.decode("utf-8") not in request
    assert "Return only the single schema-valid JSON object" in request
    assert referenced_path.is_absolute()
    assert referenced_path == prompt_file.resolve()
    assert referenced_path.read_bytes() == prompt
    assert response["prompt_size"] == len(prompt)
    assert response["prompt_sha256"] == hashlib.sha256(prompt).hexdigest()


def test_agent_matrix_dry_run_starts_with_known_success_cell(tmp_path):
    command = [
        "bash",
        str(REPO_ROOT / "scripts" / "run_deepcompile_repro_matrix.sh"),
        "--results-root",
        str(tmp_path),
        "--nproc",
        "8",
        "--frameworks",
        "deepcompile",
        "--mbs",
        "1",
        "--seqs",
        "1024",
        "--zero3-tuning-strategy",
        "agent",
        "--agent-backend",
        "codex",
        "--agent-timeout-sec",
        "1800",
        "--dry-run",
    ]
    proc = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=True, env=os.environ.copy())

    command_files = list(tmp_path.glob("*/deepcompile-mb1-seq1024/command.txt"))
    assert len(command_files) == 1
    rendered = command_files[0].read_text(encoding="utf-8")
    assert "Qwen/Qwen3-14B" in rendered
    assert "--batch-size 1" in rendered
    assert "--seq-length 1024" in rendered
    assert "--warmup_step 10" in rendered
    assert "--bench_step 30" in rendered
    assert "--seed 42" in rendered
    assert "--zero3-tuning-strategy agent" in rendered
    assert "--agent-backend codex" in rendered
    assert "--agent-timeout-sec 1800" in rendered
    assert ">>> [1] deepcompile-mb1-seq1024" in proc.stdout


def test_run_launcher_exposes_agent_cli_without_graph_operation_or_pass_bans():
    launcher = (REPO_ROOT / "run.sh").read_text(encoding="utf-8")

    assert 'AGENT_ARCHITECTURE="graph_agent"' in launcher
    assert "--zero3_tuning_strategy|--zero3-tuning-strategy" in launcher
    assert "--agent_backend|--agent-backend" in launcher
    assert "--agent_max_retries_per_iteration|--agent-max-retries-per-iteration" in launcher
    assert "--agent_timeout_sec ${AGENT_TIMEOUT_SEC} --agent_max_retries_per_iteration ${AGENT_MAX_RETRIES_PER_ITERATION}" in launcher
    assert "Agent tuning currently requires --zero_stage 3" not in launcher
    assert "Agent tuning conflicts with --passes" not in launcher


def test_run_launcher_forwards_timeout_inputs_only_in_agent_mode(tmp_path):
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    baseline_proc, baseline_args = _run_launcher_with_fake_runtime(baseline_dir)
    assert baseline_proc.returncode == 0, baseline_proc.stderr
    assert "--agent_timeout_sec" not in baseline_args
    assert "--agent_max_retries_per_iteration" not in baseline_args

    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    agent_proc, agent_args = _run_launcher_with_fake_runtime(
        agent_dir,
        "--zero3-tuning-strategy",
        "agent",
        "--agent-backend",
        "codex",
        "--agent-timeout-sec",
        "1800",
        "--agent-max-retries-per-iteration",
        "1",
    )
    assert agent_proc.returncode == 0, agent_proc.stderr
    assert agent_args[agent_args.index("--agent_timeout_sec") + 1] == "1800"
    assert agent_args[agent_args.index("--agent_max_retries_per_iteration") + 1] == "1"


def test_run_launcher_rejects_invalid_agent_retry_count(tmp_path):
    proc, launched_args = _run_launcher_with_fake_runtime(
        tmp_path,
        "--zero3-tuning-strategy",
        "agent",
        "--agent-backend",
        "codex",
        "--agent-max-retries-per-iteration",
        "-1",
    )
    assert proc.returncode == 2
    assert "must be a nonnegative integer" in proc.stderr
    assert launched_args == []


def test_agent_timeout_wires_accelerate_handler_without_changing_default(monkeypatch):
    verify_loss, accelerator_calls = _load_verify_loss_with_fake_runtime(monkeypatch)

    verify_loss.create_accelerator(
        gradient_accumulation_steps=2,
        agent_timeout_sec=None,
        agent_max_retries_per_iteration=None,
    )
    assert accelerator_calls[-1] == {"gradient_accumulation_steps": 2}

    verify_loss.create_accelerator(
        gradient_accumulation_steps=2,
        agent_timeout_sec=1800,
        agent_max_retries_per_iteration=1,
    )
    agent_kwargs = accelerator_calls[-1]
    assert agent_kwargs["gradient_accumulation_steps"] == 2
    assert len(agent_kwargs["kwargs_handlers"]) == 1
    assert agent_kwargs["kwargs_handlers"][0].kwargs == {"timeout": verify_loss.timedelta(seconds=3900)}
    assert verify_loss.agent_process_group_timeout(1800, 0) == verify_loss.timedelta(seconds=1980)


def test_agent_timeout_rejects_invalid_values(monkeypatch):
    verify_loss, _ = _load_verify_loss_with_fake_runtime(monkeypatch)

    for timeout_sec, retry_count, expected_message in (
        (0, 1, "--agent-timeout-sec must be greater than zero"),
        (1800, -1, "--agent-max-retries-per-iteration must be nonnegative"),
    ):
        try:
            verify_loss.agent_process_group_timeout(timeout_sec, retry_count)
        except ValueError as exc:
            assert str(exc) == expected_message
        else:
            raise AssertionError("invalid agent timeout inputs were accepted")


def test_known_good_dataset_metrics_and_matrix_files_are_retained():
    verify_loss = (REPO_ROOT / "verify_loss.py").read_text(encoding="utf-8")
    data_utils = (REPO_ROOT / "data_utils.py").read_text(encoding="utf-8")

    assert "rank_step_times" in verify_loss
    assert "rank_memory_stats" in verify_loss
    assert "metrics_output" in verify_loss
    assert "SyntheticTokenDataset" in data_utils
    assert (REPO_ROOT / "scripts" / "summarize_repro_matrix.py").is_file()
