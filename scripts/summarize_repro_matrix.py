#!/usr/bin/env python3
"""Summarize ds_verify_loss reproduction matrix runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


FRAMEWORKS = ["fsdp", "deepspeed", "deepcompile"]


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(errors="replace"))
    except json.JSONDecodeError:
        return {}


def bytes_to_gib(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value) / (1024**3)
    except (TypeError, ValueError):
        return None


def classify_status(cell_dir: Path, metrics: dict[str, Any], summary: dict[str, Any]) -> str:
    if metrics.get("success") is True and summary.get("status") == "success":
        return "ok"
    error = metrics.get("error_summary") or {}
    log_tail = ""
    log_path = cell_dir / "train.log"
    if log_path.is_file():
        log_tail = log_path.read_text(errors="replace")[-8000:]
    text = " ".join(
        str(x)
        for x in [
            error.get("type"),
            error.get("message"),
            error.get("traceback_tail"),
            log_tail,
        ]
        if x
    )
    lowered = text.lower()
    if "out of memory" in lowered or "cuda oom" in lowered:
        return "oom"
    if "device_time" in text:
        return "error:device_time"
    if summary.get("status"):
        return str(summary["status"])
    return "missing"


def parse_cell_name(name: str) -> tuple[str, int, int] | None:
    match = re.fullmatch(r"(fsdp|deepspeed|deepcompile)-mb(\d+)-seq(\d+)", name)
    if not match:
        return None
    return match.group(1), int(match.group(2)), int(match.group(3))


def load_cells(runs_root: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    cells: dict[tuple[str, int, int], dict[str, Any]] = {}
    for cell_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        parsed = parse_cell_name(cell_dir.name)
        if not parsed:
            continue
        metrics = read_json(cell_dir / "metrics.json")
        summary = read_json(cell_dir / "summary.json")
        env = read_json(cell_dir / "environment.json")
        framework, mb, seq = parsed
        cells[(framework, mb, seq)] = {
            "framework": framework,
            "mb": mb,
            "seq": seq,
            "status": classify_status(cell_dir, metrics, summary),
            "avg_step_time_sec": metrics.get("avg_step_time_sec"),
            "samples_per_second": metrics.get("samples_per_second"),
            "tokens_per_second": metrics.get("tokens_per_second"),
            "peak_alloc_gib": bytes_to_gib(
                metrics.get("cuda_max_memory_allocated_cross_rank_bytes", metrics.get("cuda_max_memory_allocated_bytes"))
            ),
            "peak_reserved_gib": bytes_to_gib(
                metrics.get("cuda_max_memory_reserved_cross_rank_bytes", metrics.get("cuda_max_memory_reserved_bytes"))
            ),
            "measured_steps": metrics.get("measured_steps"),
            "torch": env.get("torch"),
            "deepspeed": env.get("deepspeed"),
            "transformers": env.get("transformers"),
            "path": str(cell_dir),
        }
    return cells


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def axes(cells: dict[tuple[str, int, int], dict[str, Any]]) -> tuple[list[int], list[int]]:
    return (
        sorted({key[1] for key in cells}),
        sorted({key[2] for key in cells}),
    )


def metric_cell(
    cells: dict[tuple[str, int, int], dict[str, Any]],
    framework: str,
    mb: int,
    seq: int,
    metric: str,
) -> str:
    cell = cells.get((framework, mb, seq))
    if not cell:
        return "MISSING"
    if cell["status"] != "ok":
        return cell["status"]
    digits = 2 if metric.endswith("_gib") else 4
    return fmt(cell.get(metric), digits)


def write_tables(cells: dict[tuple[str, int, int], dict[str, Any]], out: Path) -> None:
    mbs, seqs = axes(cells)
    lines = ["# Reproduction Matrix Summary", ""]
    for framework in FRAMEWORKS:
        lines.extend([f"## {framework}", ""])
        for metric, title in [
            ("avg_step_time_sec", "average measured step time (s)"),
            ("peak_alloc_gib", "cross-rank peak allocated (GiB)"),
            ("peak_reserved_gib", "cross-rank peak reserved (GiB)"),
        ]:
            lines.extend([f"### {title}", ""])
            lines.append("| mb \\ seq | " + " | ".join(str(seq) for seq in seqs) + " |")
            lines.append("| --- | " + " | ".join("---" for _ in seqs) + " |")
            for mb in mbs:
                row = [metric_cell(cells, framework, mb, seq, metric) for seq in seqs]
                lines.append(f"| {mb} | " + " | ".join(row) + " |")
            lines.append("")
    out.write_text("\n".join(lines) + "\n")


def write_long_csv(cells: dict[tuple[str, int, int], dict[str, Any]], out: Path) -> None:
    fields = [
        "framework",
        "mb",
        "seq",
        "status",
        "avg_step_time_sec",
        "samples_per_second",
        "tokens_per_second",
        "peak_alloc_gib",
        "peak_reserved_gib",
        "measured_steps",
        "torch",
        "deepspeed",
        "transformers",
        "path",
    ]
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for cell in sorted(cells.values(), key=lambda x: (x["framework"], x["mb"], x["seq"])):
            writer.writerow({field: cell.get(field) for field in fields})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    out_dir = args.out_dir or args.runs_root
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = load_cells(args.runs_root)
    if not cells:
        raise SystemExit(f"no matrix cells found under {args.runs_root}")
    write_tables(cells, out_dir / "matrix-summary.md")
    write_long_csv(cells, out_dir / "matrix-long.csv")
    ok = sum(1 for cell in cells.values() if cell["status"] == "ok")
    print(f"wrote {out_dir / 'matrix-summary.md'} ({ok}/{len(cells)} ok)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
