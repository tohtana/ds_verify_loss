#!/usr/bin/env python3
"""Generate one-file diagnostic reports for ds_verify_loss runs."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROFILE_FIELDS = [
    ("gpu_idle", "GPU idle"),
    ("rank_imbalance", "Rank imbalance"),
    ("visible_nccl_collective_cost", "Visible NCCL/collective cost"),
    ("input_h2d_wait", "Input/H2D wait"),
    ("optimizer_tail", "Optimizer tail"),
    ("allocator_or_memory_pressure", "Allocator or memory pressure"),
    ("compile_graph_break_signal", "Compile/graph-break signal"),
]


def read_text(path: Path | None) -> str:
    if not path or not path.exists():
        return ""
    return path.read_text(errors="replace")


def read_json(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(errors="replace"))
    except json.JSONDecodeError as exc:
        return {"_parse_error": f"{type(exc).__name__}: {exc}"}


def find_error_summary(log_text: str) -> str | None:
    if not log_text:
        return None
    patterns = [
        r"Traceback \(most recent call last\):(?P<body>.*?)(?=\n\S|\Z)",
        r"RuntimeError: (?P<body>.*)",
        r"ValueError: (?P<body>.*)",
        r"ChildFailedError:(?P<body>.*?)(?=\n\S|\Z)",
    ]
    for pattern in patterns:
        match = re.search(pattern, log_text, flags=re.DOTALL)
        if match:
            body = match.group(0).strip()
            lines = [line.rstrip() for line in body.splitlines() if line.strip()]
            return "\n".join(lines[-24:])
    return None


def fmt_value(value: Any, unit: str = "") -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:.4f}{unit}"
    return f"{value}{unit}"


def bytes_to_gib(value: Any) -> str:
    if value is None:
        return "not available"
    try:
        return f"{float(value) / (1024 ** 3):.2f} GiB"
    except (TypeError, ValueError):
        return "not available"


def write_report(
    results_dir: Path,
    exit_code: int,
    command: str,
    metrics: dict[str, Any],
    profile: dict[str, Any],
    environment: dict[str, Any],
    log_text: str,
    next_command: str,
) -> dict[str, Any]:
    metrics_success = metrics.get("success")
    status = "success" if exit_code == 0 and metrics_success is not False else "failure"
    if exit_code != 0:
        status = "failure"

    error_summary = None
    if status != "success":
        metric_error = metrics.get("error_summary")
        if isinstance(metric_error, dict):
            error_summary = metric_error.get("traceback_tail") or metric_error.get("message")
        elif metric_error:
            error_summary = str(metric_error)
        error_summary = error_summary or find_error_summary(log_text) or "No traceback summary found in captured log."

    report_path = results_dir / "report.md"
    with report_path.open("w") as f:
        f.write("# ds_verify_loss Diagnostic Run Report\n\n")
        f.write(f"- Generated: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"- Status: `{status}`\n")
        f.write(f"- Exit code: `{exit_code}`\n")
        f.write(f"- Results directory: `{results_dir}`\n\n")

        f.write("## Command\n\n")
        f.write("```bash\n")
        f.write(command.strip() + "\n")
        f.write("```\n\n")

        f.write("## Environment\n\n")
        for key in ["hostname", "python", "torch", "deepspeed", "accelerate", "transformers", "cuda_available", "cuda_device_count"]:
            if key in environment:
                f.write(f"- {key}: `{environment[key]}`\n")
        if environment.get("nvidia_smi"):
            f.write("\n```text\n")
            f.write(str(environment["nvidia_smi"]).strip()[:4000] + "\n")
            f.write("```\n")
        f.write("\n")

        f.write("## Metrics\n\n")
        f.write("| Field | Value |\n")
        f.write("| --- | --- |\n")
        f.write(f"| avg step time | {fmt_value(metrics.get('avg_step_time_sec'), ' sec')} |\n")
        f.write(f"| samples/s | {fmt_value(metrics.get('samples_per_second'))} |\n")
        f.write(f"| tokens/s | {fmt_value(metrics.get('tokens_per_second'))} |\n")
        f.write(f"| max memory allocated | {bytes_to_gib(metrics.get('cuda_max_memory_allocated_bytes'))} |\n")
        f.write(f"| memory reserved | {bytes_to_gib(metrics.get('cuda_memory_reserved_bytes'))} |\n")
        f.write(f"| max memory reserved | {bytes_to_gib(metrics.get('cuda_max_memory_reserved_bytes'))} |\n")
        f.write(f"| measured steps | {fmt_value(metrics.get('measured_steps'))} |\n")
        f.write(f"| global samples/step | {fmt_value(metrics.get('global_samples_per_step'))} |\n")
        f.write(f"| global tokens/step | {fmt_value(metrics.get('global_tokens_per_step'))} |\n\n")

        f.write("## Profile Summary\n\n")
        f.write("| Signal | Rating |\n")
        f.write("| --- | --- |\n")
        for key, label in PROFILE_FIELDS:
            f.write(f"| {label} | `{profile.get(key, 'unknown')}` |\n")
        notes = profile.get("notes") or []
        if notes:
            f.write("\nNotes:\n")
            for note in notes:
                f.write(f"- {note}\n")
        f.write("\n")

        if status != "success":
            f.write("## Error Summary\n\n")
            f.write("```text\n")
            f.write((error_summary or "").strip()[:6000] + "\n")
            f.write("```\n\n")

        f.write("## Known Limitations\n\n")
        f.write("- This report summarizes a short diagnostic run, not the full optimization matrix.\n")
        f.write("- The profile classifier is intentionally coarse and should be treated as a triage hint.\n")
        f.write("- Rank imbalance is based on gathered per-rank step timing, not a full distributed trace.\n")
        f.write("- GPU idle, H2D, optimizer, and NCCL ratings come from rank-0 profiler events when profiling is enabled.\n\n")

        f.write("## Next Suggested Command\n\n")
        f.write("```bash\n")
        f.write(next_command.strip() + "\n")
        f.write("```\n")

    summary = {
        "status": status,
        "exit_code": exit_code,
        "report_path": str(report_path),
        "metrics_path": str(results_dir / "metrics.json"),
        "profile_summary": profile,
        "error_summary": error_summary,
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument("--command-file", type=Path)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--metrics-file", type=Path)
    parser.add_argument("--profile-summary-file", type=Path)
    parser.add_argument("--environment-file", type=Path)
    parser.add_argument("--next-command", default="")
    args = parser.parse_args()

    results_dir = args.results_dir
    metrics_file = args.metrics_file or results_dir / "metrics.json"
    profile_file = args.profile_summary_file or results_dir / "profile_summary.json"
    environment_file = args.environment_file or results_dir / "environment.json"
    log_file = args.log_file or results_dir / "train.log"
    command_file = args.command_file or results_dir / "command.txt"

    metrics = read_json(metrics_file)
    profile = read_json(profile_file) or metrics.get("profile_summary") or {}
    environment = read_json(environment_file)
    log_text = read_text(log_file)
    command = read_text(command_file)
    next_command = args.next_command or command

    summary = write_report(
        results_dir=results_dir,
        exit_code=args.exit_code,
        command=command,
        metrics=metrics,
        profile=profile,
        environment=environment,
        log_text=log_text,
        next_command=next_command,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
