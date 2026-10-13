# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Output formats.

text  for people at a terminal
json  every number
"""
from __future__ import annotations

import json

import slo_config as C


def report_text(current, summary: dict) -> str:
    """Human summary for `report`: worst files against each target."""
    files = current["files"]
    lines = [f"{len(files)} files measured.", ""]
    lines.append("Files over each target today:")
    for metric, (limit, direction) in C.TARGETS.items():
        if metric not in summary:
            continue
        op = "<=" if direction == "max" else ">="
        lines.append(f"  {metric:15} target {op} {limit:<6} over: {summary[metric]}")
    for metric, label in (
        ("sloc", "Biggest files (SLOC)"),
        ("raw_per_100", "Most raw IR ops per 100 SLOC"),
        ("longest_fn", "Longest functions"),
        ("overlap_max", "Most copy-paste overlap"),
    ):
        lines += ["", label + ":"]
        for r in sorted(files.values(), key=lambda r: r[metric] or 0, reverse=True)[:8]:
            extra = (
                f"  {r['longest_fn_name']}"
                if metric == "longest_fn"
                else f"  with {r['overlap_with']}" if metric == "overlap_max" else ""
            )
            lines.append(f"  {r[metric]:>8}  {r['path']}{extra}")
    return "\n".join(lines)


def to_json(obj) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, default=sorted)
