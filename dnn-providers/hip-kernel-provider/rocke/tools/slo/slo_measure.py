# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Measure a whole tree: per-file metrics plus copy-paste overlap between files.

A file that does not parse becomes an error record (check id SLO000), not a crash,
so one bad file never hides the numbers for the rest.
"""
from __future__ import annotations

from pathlib import Path

import slo_config as C
import slo_metrics as M
import slo_overlap as O


def measure(
    root: Path, files: list[str] | None = None, coverage: dict | None = None
) -> dict:
    """Every number for every in-scope file. `coverage` is slo_coverage.run() output."""
    files = files if files is not None else M.scope_files(root)
    helper_modules = M.helper_modules_of(root)
    sources, results, errors = {}, {}, []
    for rel in files:
        src = (root / rel).read_text(encoding="utf-8", errors="replace")
        try:
            results[rel] = M.analyze_source(src, rel, helper_modules)
            sources[rel] = src
        except SyntaxError as e:
            errors.append(
                {
                    "check": "SLO000",
                    "kind": "error",
                    "path": rel,
                    "line": e.lineno or 1,
                    "metric": "",
                    "old": None,
                    "new": None,
                    "target": None,
                    "enforce": True,
                    "message": f"cannot parse: {e.msg}",
                }
            )
    pairs = O.pairs(sources, {p: r["sloc"] for p, r in results.items()})
    worst = O.worst_per_file(pairs)
    for rel, r in results.items():
        w = worst.get(rel)
        r["overlap_max"] = w["overlap"] if w else 0.0
        r["overlap_with"] = w["with"] if w else ""
        if coverage is not None:
            c = coverage["files"].get(rel)
            r["cov_file"] = c["cov_file"] if c else None
            r["cov_body"] = c["cov_body"] if c else None
            r["cov_missing"] = c["missing_body"] if c else None
    return {
        "files": results,
        "pairs": pairs,
        "errors": errors,
        "coverage_ran": coverage is not None,
        "harness_ok": coverage["harness_ok"] if coverage else None,
    }


def over_target_summary(current: dict) -> dict[str, int]:
    """How many files are over each target today (old or new). For the report."""
    out = {}
    for metric, (limit, direction) in C.TARGETS.items():
        vals = [r.get(metric) for r in current["files"].values()]
        out[metric] = sum(
            1
            for v in vals
            if v is not None and (v > limit if direction == "max" else v < limit)
        )
    return out
