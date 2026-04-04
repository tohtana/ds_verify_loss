#!/usr/bin/env python3

import json
import os
import subprocess
import sys


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


def _build_command():
    command = [
        os.environ.get("CODEX_BIN", "codex"),
        "--dangerously-bypass-approvals-and-sandbox",
        "exec",
        "--json",
    ]

    model = os.environ.get("CODEX_AGENT_MODEL")
    if model:
        command.extend(["-m", model])

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
