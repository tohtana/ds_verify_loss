#!/usr/bin/env python3

"""Adapt Codex CLI JSONL output to DeepCompile's JSON-over-stdio agents."""

import json
import os
from pathlib import Path
import stat
import subprocess
import sys


CODEX_MODEL = "gpt-5.6-sol"
CODEX_REASONING_CONFIG = 'model_reasoning_effort="xhigh"'
CODEX_MAX_INLINE_CHARS = 1_048_576
CODEX_BINARY_RELATIVE_PATH = (Path("node_modules") / "@openai" / "codex-linux-x64" / "vendor" /
                              "x86_64-unknown-linux-musl" / "bin" / "codex")


def _workspace_default_codex_bin() -> Path:
    wrapper_path = Path(__file__).resolve()
    candidates = [
        wrapper_path.parents[parent_index] / "tools" / "codex-cli" / CODEX_BINARY_RELATIVE_PATH
        for parent_index in (3, 2)
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    for candidate in candidates:
        if candidate.parents[len(CODEX_BINARY_RELATIVE_PATH.parts) - 1].is_dir():
            return candidate
    return candidates[0]


DEFAULT_CODEX_BIN = _workspace_default_codex_bin()


def _stdin_file_path():
    fd = sys.stdin.fileno()
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        raise ValueError("oversized Codex prompts require regular-file stdin")

    path = Path(f"/proc/self/fd/{fd}").resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"oversized Codex prompt is not available as a file: {path}")
    return path


def _read_codex_input():
    prompt = sys.stdin.read(CODEX_MAX_INLINE_CHARS + 1)
    if len(prompt) <= CODEX_MAX_INLINE_CHARS:
        return prompt

    prompt_path = _stdin_file_path()
    return (
        "Read the complete DeepCompile agent prompt from the absolute UTF-8 file below and follow it.\n"
        f"PROMPT_PATH={json.dumps(str(prompt_path))}\n"
        "Return only the single schema-valid JSON object required by its response_contract; no prose or Markdown.\n"
    )


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
        if not isinstance(obj, dict) or obj.get("type") != "item.completed":
            continue

        item = obj.get("item")
        if not isinstance(item, dict) or item.get("type") not in ("agent_message", "assistant_message"):
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
        source = "CODEX_BIN" if configured_path else "the persistent workspace Codex installation"
        raise FileNotFoundError(f"Unable to resolve an executable Codex binary from {source}: {candidate}")
    return str(candidate)


def _build_command():
    command = [
        _resolve_codex_bin(),
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
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
    try:
        codex_input = _read_codex_input()
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"Unable to prepare Codex prompt: {exc}\n")
        return 1

    proc = subprocess.run(_build_command(), input=codex_input, text=True, capture_output=True, check=False)

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
