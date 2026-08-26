#!/usr/bin/env python3

import json
import os
from pathlib import Path
import subprocess
import sys


CODEX_MODEL = "gpt-5.6-sol"
CODEX_REASONING_CONFIG = 'model_reasoning_effort="xhigh"'
DEFAULT_CODEX_BIN = (Path(__file__).resolve().parent.parent / ".codex-cli" / "node_modules" / "@openai" /
                     "codex-linux-x64" / "vendor" / "x86_64-unknown-linux-musl" / "bin" / "codex")


def _iter_json_lines(raw_output):
    for line in raw_output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _extract_final_text(raw_output):
    final_text = None
    for obj in _iter_json_lines(raw_output):
        if not isinstance(obj, dict):
            continue
        if obj.get("type") != "item.completed":
            continue

        item = obj.get("item")
        if not isinstance(item, dict):
            continue
        if item.get("type") not in ("agent_message", "assistant_message"):
            continue

        text = item.get("text")
        if isinstance(text, str):
            final_text = text

    if final_text is None:
        raise ValueError("Codex CLI output did not contain a final assistant message")
    return final_text


def _resolve_codex_bin():
    configured_path = os.environ.get("CODEX_BIN")
    candidate = Path(configured_path).expanduser() if configured_path else DEFAULT_CODEX_BIN
    candidate = candidate.resolve()
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        source = "CODEX_BIN" if configured_path else "the persistent Codex installation"
        raise FileNotFoundError(f"Unable to resolve an executable Codex binary from {source}: {candidate}")
    return str(candidate)


def _build_command():
    command = [
        _resolve_codex_bin(),
        "--dangerously-bypass-approvals-and-sandbox",
        "exec",
        "--json",
        "-m",
        CODEX_MODEL,
        "-c",
        CODEX_REASONING_CONFIG,
    ]

    profile = os.environ.get("CODEX_AGENT_PROFILE")
    if profile:
        command.extend(["-p", profile])

    return command


def main():
    prompt = sys.stdin.read()
    proc = subprocess.run(_build_command(), input=prompt, text=True, capture_output=True, check=False)

    if proc.returncode != 0:
        if proc.stdout:
            sys.stderr.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        return proc.returncode

    try:
        final_text = _extract_final_text(proc.stdout)
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        if proc.stdout:
            sys.stderr.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        return 1

    sys.stdout.write(final_text)
    if final_text and not final_text.endswith("\n"):
        sys.stdout.write("\n")
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
