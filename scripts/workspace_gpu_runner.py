#!/usr/bin/env python3
"""Run a shell command on an Anyscale Workspace GPU worker through Ray."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_env(values: list[str]) -> dict[str, str]:
    env = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"expected KEY=VALUE for --env, got {item!r}")
        key, value = item.split("=", 1)
        env[key] = value
    return env


def tail_lines(text: str, limit: int = 200) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-limit:])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--cwd", required=True, help="Working directory visible to the GPU worker.")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--env", action="append", default=[], help="Environment variable in KEY=VALUE form.")
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--command-log", type=Path)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command and not args.inventory_only:
        parser.error("expected command after --")

    import ray

    ray.init(address="auto")
    nodes = [
        {
            "node_id": node.get("NodeID"),
            "alive": node.get("Alive"),
            "resources": node.get("Resources", {}),
            "node_manager_address": node.get("NodeManagerAddress"),
        }
        for node in ray.nodes()
    ]

    @ray.remote(num_gpus=args.num_gpus)
    def run_on_worker(payload: dict[str, Any]) -> dict[str, Any]:
        import ray

        env = os.environ.copy()
        env.update(payload["env"])
        cwd = payload["cwd"]
        command = payload["command"]
        command_log = payload.get("command_log")

        started_at = time.time()
        inventory = {
            "hostname": socket.gethostname(),
            "cwd": cwd,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "ray_gpu_ids": [str(item) for item in ray.get_gpu_ids()],
        }
        try:
            inventory["nvidia_smi"] = subprocess.check_output(
                ["nvidia-smi"], text=True, stderr=subprocess.STDOUT, timeout=30
            )
        except Exception as exc:
            inventory["nvidia_smi"] = f"{type(exc).__name__}: {exc}"

        if payload["inventory_only"]:
            return {
                "status": "success",
                "return_code": 0,
                "inventory": inventory,
                "duration_sec": time.time() - started_at,
            }

        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=payload["timeout_seconds"],
                check=False,
            )
            output = completed.stdout or ""
            if command_log:
                Path(command_log).parent.mkdir(parents=True, exist_ok=True)
                Path(command_log).write_text(output)
            return {
                "status": "success" if completed.returncode == 0 else "failure",
                "return_code": completed.returncode,
                "command": command,
                "inventory": inventory,
                "duration_sec": time.time() - started_at,
                "stdout_tail": tail_lines(output),
            }
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
            if command_log:
                Path(command_log).parent.mkdir(parents=True, exist_ok=True)
                Path(command_log).write_text(output)
            return {
                "status": "timeout",
                "return_code": 124,
                "command": command,
                "inventory": inventory,
                "duration_sec": time.time() - started_at,
                "stdout_tail": tail_lines(output),
                "error": f"timed out after {payload['timeout_seconds']} seconds",
            }

    payload = {
        "cwd": args.cwd,
        "command": command,
        "timeout_seconds": args.timeout_seconds,
        "env": parse_env(args.env),
        "command_log": str(args.command_log) if args.command_log else None,
        "inventory_only": args.inventory_only,
    }
    result = ray.get(run_on_worker.remote(payload))
    result["ray_nodes"] = nodes

    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(json.dumps(result, indent=2, sort_keys=True))
    return int(result.get("return_code", 1))


if __name__ == "__main__":
    raise SystemExit(main())
