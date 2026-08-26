# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import ast
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import codex_agent_wrapper
import generate_conf

from deepspeed.compile.config import CompileConfig


init_z3_module = importlib.import_module("deepspeed.compile.init_z3")


def _load_make_schedule():
    source_path = Path(__file__).resolve().parents[1] / "verify_loss.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "make_schedule")
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"List": List}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["make_schedule"]


def _config_args(tmp_path):
    return SimpleNamespace(machine_rank=0,
                           num_machines=1,
                           num_processes=2,
                           zero_stage=3,
                           fp16=False,
                           gradient_accumulation_steps=1,
                           deepcompile=True,
                           debug_log=False,
                           sync_before_reduce=False,
                           sync_after_reduce=False,
                           sync_before_allgather=False,
                           sync_after_allgather=False,
                           zero3_tuning_strategy="agent",
                           agent_backend="codex",
                           agent_architecture="two_agent",
                           agent_max_iterations=3,
                           agent_max_retries_per_iteration=1,
                           agent_timeout_sec=300,
                           template_file=Path(generate_conf.__file__).resolve().parent / "configs" /
                           "ds_config.json.template",
                           output_file=tmp_path / "ds_config.json")


def test_codex_command_uses_durable_binary_and_required_model(monkeypatch):
    monkeypatch.delenv("CODEX_BIN", raising=False)
    monkeypatch.delenv("CODEX_AGENT_MODEL", raising=False)

    command = codex_agent_wrapper._build_command()

    assert command[0] == str(codex_agent_wrapper.DEFAULT_CODEX_BIN.resolve())
    assert command[-4:] == ["-m", "gpt-5.6-sol", "-c", 'model_reasoning_effort="xhigh"']
    assert "codex" not in command[:1]


def test_codex_command_accepts_only_an_explicit_executable_override(tmp_path, monkeypatch):
    executable = tmp_path / "codex"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("CODEX_BIN", str(executable))

    assert codex_agent_wrapper._build_command()[0] == str(executable.resolve())


def test_two_agent_config_sets_both_role_commands(tmp_path):
    args = _config_args(tmp_path)

    generate_conf.main(args)

    config = json.loads(args.output_file.read_text(encoding="utf-8"))["compile"]
    expected = [generate_conf.sys.executable, str(Path(codex_agent_wrapper.__file__).resolve())]
    assert config["agent_command"] == expected
    assert config["agent_evaluator_command"] == expected
    assert config["agent_optimizer_command"] == expected
    assert config["agent_max_retries_per_iteration"] == 1


def test_launcher_does_not_add_zero_stage_or_custom_pass_agent_bans():
    launcher = (Path(generate_conf.__file__).resolve().parent / "run.sh").read_text(encoding="utf-8")

    assert "Agent tuning currently requires --zero_stage 3" not in launcher
    assert "Agent tuning conflicts with --passes" not in launcher


def test_companion_pass_schedules_are_preserved_and_composed_with_agent():
    make_schedule = _load_make_schedule()
    explicit = make_schedule(["prefetch", "selective_gather"], warmup=5)
    original = [(step, list(passes)) for step, passes in explicit]

    assert init_z3_module._compose_agent_schedule(explicit, CompileConfig()) is explicit
    assert explicit == original

    config = CompileConfig(zero3_tuning_strategy="agent", agent_command=["agent"])
    composed = init_z3_module._compose_agent_schedule(explicit, config)
    assert composed[-1][1][:-1] == original[-1][1]
    assert composed[-1][1][-1] is init_z3_module.agent_optimization_loop

    no_warmup = make_schedule(["offload_adam_states_sync"], warmup=5)
    composed_no_warmup = init_z3_module._compose_agent_schedule(no_warmup, config)
    assert composed_no_warmup[0] == no_warmup[0]
    assert composed_no_warmup[-1][0] == init_z3_module.WARMUP
    assert composed_no_warmup[-1][1][:-1] == no_warmup[0][1]
    assert composed_no_warmup[-1][1][-1] is init_z3_module.agent_optimization_loop
