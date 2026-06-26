# Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Combined quality + AI-friendliness report for tensilelite (Tensile/ tree).

Folds two measurements into one check:
  * complexity / size  — function CCN and file NLOC violator counts (via lizard)
  * AI-friendliness     — 21 AST readability signals (via llm_readability_report)

Report-only by default: it prints current values against the targets in
``quality_targets.json`` and ALWAYS exits 0, so it never blocks a commit while
the targets are still being agreed. Targets start at the current measurement
("hold the line") and are tightened deliberately over time.

Usage:
    python check_quality.py             # print the report (exit 0)
    python check_quality.py --update     # set targets to the current measurement
    python check_quality.py --enforce    # exit 1 if any metric is over target (opt-in)

See BASELINE.md for the signal definitions and layout notes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lizard

# scripts/quality/ -> scripts/ -> tensilelite/
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
METRICS_DIR = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "Tensile"
TARGETS_PATH = METRICS_DIR / "quality_targets.json"

sys.path.insert(0, str(METRICS_DIR))
from llm_readability_report import (  # noqa: E402
    LOC_HIGH,
    NESTING_THRESHOLD,
    SKIP_DIRS,
    INIT_LOC_THRESHOLD,
    INIT_REEXPORT_THRESHOLD,
    scan,
)

CCN_LOW = 12
CCN_HIGH = 20
NLOC_LOW = 509
NLOC_HIGH = 1000

# Display order + grouping. Each entry: (key, human label).
COMPLEXITY_METRICS = [
    ("ccn_gt_12", "functions CCN > 12"),
    ("ccn_gt_20", "functions CCN > 20"),
    ("file_nloc_gt_509", "files NLOC > 509"),
    ("file_nloc_gt_1000", "files NLOC > 1000"),
]
READABILITY_METRICS = [
    ("file_loc_ge_1000", "files >= 1000 LOC"),
    ("token_heavy_files", "files > ~5000 tokens"),
    ("deep_nesting_ge_5", "functions nesting depth >= 5"),
    ("swallowed_errors", "swallowed errors"),
    ("init_bloat", "bloated __init__.py"),
    ("interface_first_violators", "impl-before-interface files"),
    ("missing_seam_tests", "modules without a test"),
    ("cross_feature_imports", "lateral cross-feature imports"),
    ("cross_layer_special_cases", "cross-feature literal dispatch"),
    ("duplicate_literal_clusters", "duplicated long literals"),
    ("depth_below_threshold", "shallow features (impl/iface ratio)"),
    ("shallow_modules", "shallow features (impl/symbol)"),
    ("parallel_impl_pairs", "parallel-impl module pairs"),
    ("typing_any_total", "typing.Any uses"),
    ("typing_cast_total", "typing.cast uses"),
    ("type_ignore_total", "# type: ignore comments"),
    ("generic_filenames", "generic filenames"),
    ("generic_feature_dirs", "generic feature dirs"),
    ("internal_test_imports", "tests importing private paths"),
    ("adapter_violations", "adapter-seam violations"),
]


def _measure_complexity(src_root: Path) -> dict[str, int]:
    ccn_gt_12 = ccn_gt_20 = nloc_gt_509 = nloc_gt_1000 = 0
    for fileinfo in lizard.analyze([str(src_root)]):
        if any(part in SKIP_DIRS for part in Path(fileinfo.filename).parts):
            continue
        if fileinfo.nloc > NLOC_LOW:
            nloc_gt_509 += 1
        if fileinfo.nloc > NLOC_HIGH:
            nloc_gt_1000 += 1
        for fn in fileinfo.function_list:
            if fn.cyclomatic_complexity > CCN_LOW:
                ccn_gt_12 += 1
            if fn.cyclomatic_complexity > CCN_HIGH:
                ccn_gt_20 += 1
    return {
        "ccn_gt_12": ccn_gt_12,
        "ccn_gt_20": ccn_gt_20,
        "file_nloc_gt_509": nloc_gt_509,
        "file_nloc_gt_1000": nloc_gt_1000,
    }


def _measure_readability(src_root: Path) -> tuple[dict[str, int], int]:
    files, summary = scan(src_root)
    counts = {
        "file_loc_ge_1000": sum(1 for f in files if f.loc >= LOC_HIGH),
        "init_bloat": sum(
            1
            for f in files
            if f.is_init
            and (f.loc > INIT_LOC_THRESHOLD or f.init_reexports > INIT_REEXPORT_THRESHOLD)
        ),
        "swallowed_errors": sum(len(f.swallowed_error_lines) for f in files),
        "deep_nesting_ge_5": sum(1 for f in files if f.max_nesting_depth >= NESTING_THRESHOLD),
        "cross_feature_imports": sum(len(f.cross_feature_imports) for f in files),
        "typing_any_total": sum(f.any_count for f in files),
        "typing_cast_total": sum(f.cast_count for f in files),
        "type_ignore_total": sum(f.type_ignore_count for f in files),
        "generic_filenames": sum(1 for f in files if f.generic_name),
        "generic_feature_dirs": len(summary["generic_feature_dirs"]),
        "shallow_modules": len(summary["shallow_modules"]),
        "depth_below_threshold": len(summary["depth_below_threshold"]),
        "duplicate_literal_clusters": len(summary["duplicate_literal_clusters"]),
        "cross_layer_special_cases": len(summary["cross_layer_special_cases"]),
        "missing_seam_tests": len(summary["missing_seam_tests"]),
        "interface_first_violators": len(summary["interface_first_violators"]),
        "internal_test_imports": summary["internal_test_imports_total"],
        "token_heavy_files": len(summary["token_heavy_files"]),
        "parallel_impl_pairs": len(summary["parallel_impl_pairs"]),
        "adapter_violations": len(summary["adapter_violations"]),
    }
    return counts, summary["file_count"]


def measure() -> tuple[dict[str, int], int]:
    counts, file_count = _measure_readability(SRC_ROOT)
    counts.update(_measure_complexity(SRC_ROOT))
    return counts, file_count


def _load_targets() -> dict[str, int]:
    if not TARGETS_PATH.exists():
        return {}
    return json.loads(TARGETS_PATH.read_text())


def _write_targets(targets: dict[str, int]) -> None:
    TARGETS_PATH.write_text(json.dumps(targets, indent=2, sort_keys=True) + "\n")


def _status(cur: int, tgt: int | None) -> tuple[str, bool]:
    """Return (text status, is_over). is_over drives the over-target count."""
    if tgt is None:
        return "—", False
    if cur > tgt:
        return f"over (+{cur - tgt})", True
    if cur < tgt:
        return f"under (-{tgt - cur})", False
    return "ok", False


def _rows(current: dict, targets: dict) -> list[tuple[str, str, int, int | None, str, bool]]:
    """One combined row per metric: (group, label, current, target, status, is_over)."""
    spec = [("complexity/size", k, lbl) for k, lbl in COMPLEXITY_METRICS]
    spec += [("ai-friendliness", k, lbl) for k, lbl in READABILITY_METRICS]
    rows = []
    for group, key, label in spec:
        cur = current.get(key, 0)
        tgt = targets.get(key)
        status, is_over = _status(cur, tgt)
        rows.append((group, label, cur, tgt, status, is_over))
    return rows


def render_text(current: dict, targets: dict, file_count: int) -> tuple[str, int]:
    rows = _rows(current, targets)
    over = sum(1 for r in rows if r[5])
    lines = [
        f"tensilelite quality report — Tensile/ ({file_count} source files)",
        "",
        f"  {'group':<16} {'metric':<36} {'current':>8} {'target':>8}  status",
    ]
    for group, label, cur, tgt, status, _ in rows:
        tgt_str = "—" if tgt is None else str(tgt)
        lines.append(f"  {group:<16} {label:<36} {cur:>8} {tgt_str:>8}  {status}")
    return "\n".join(lines) + "\n", over


def render_markdown(current: dict, targets: dict, file_count: int, enforce: bool) -> tuple[str, int]:
    rows = _rows(current, targets)
    over = sum(1 for r in rows if r[5])
    total = len(rows)
    lines = [
        "## tensilelite quality + AI-friendliness report",
        "",
        f"`Tensile/` — {file_count} source files. "
        + ("**Enforcing** (fails on over-target)." if enforce else "Report-only (not gating)."),
        "",
        "| group | metric | current | target | status |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for group, label, cur, tgt, status, is_over in rows:
        icon = "⚠️ " if is_over else ("✅ " if tgt is not None and cur < tgt else "")
        tgt_str = "—" if tgt is None else str(tgt)
        lines.append(f"| {group} | {label} | {cur} | {tgt_str} | {icon}{status} |")
    lines.append("")
    if not targets:
        lines.append("_No targets set yet — run `check_quality.py --update`._")
    elif over:
        lines.append(f"**{over}/{total} metrics over target.**")
    else:
        lines.append(f"**All {total} metrics at or under target.**")
    return "\n".join(lines) + "\n", over


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update",
        action="store_true",
        help="Set targets to the current measurement (the 'hold the line' baseline).",
    )
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="Exit 1 if any metric exceeds its target (default is report-only, exit 0).",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Emit a GitHub-flavored markdown table (for CI job summaries).",
    )
    args = parser.parse_args(argv)

    current, file_count = measure()

    if args.update:
        _write_targets(current)
        print(f"targets updated: {TARGETS_PATH.name} ({len(current)} metrics, {file_count} files)")
        return 0

    targets = _load_targets()

    if args.markdown:
        md, over = render_markdown(current, targets, file_count, args.enforce)
        print(md, end="")
        return 1 if (args.enforce and over) else 0

    text, over = render_text(current, targets, file_count)
    print(text, end="")

    total = len(COMPLEXITY_METRICS) + len(READABILITY_METRICS)
    if not targets:
        print("\nNo targets set yet — run `--update` to record the current values as targets.")
    elif over:
        mode = "ENFORCING" if args.enforce else "report-only"
        print(f"\n{over}/{total} metrics over target  ({mode})")
    else:
        print(f"\nAll {total} metrics at or under target")

    if args.enforce and over:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
