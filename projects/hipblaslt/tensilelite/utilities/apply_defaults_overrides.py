# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Convert TensileLite Logic YAML files to use a defaults+overrides pattern.

Modes:
  apply     – replace element [5] (Solutions list) with {SolutionDefaults, Solutions}
  stats     – dry-run showing per-file and aggregate savings
  revert    – expand defaults+overrides back to flat solution dicts
  fullstats – compute flat YAML / d+o YAML / d+o JSON sizes for every file
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import multiprocessing
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import yaml


SOLUTIONS_INDEX = 5
MAJORITY_THRESHOLD = 0.5

LOGIC_BASE = (
    "projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full"
)

CATEGORIES = ("StreamK", "Equality", "GridBased", "Experimental", "FreeSize")


# ---------------------------------------------------------------------------
# Hashable wrapper for counting heterogeneous YAML values
# ---------------------------------------------------------------------------

def _make_hashable(val: Any) -> Any:
    """Return a hashable proxy for *val* so it can be used as a dict key."""
    if isinstance(val, dict):
        return tuple(sorted((_make_hashable(k), _make_hashable(v)) for k, v in val.items()))
    if isinstance(val, list):
        return tuple(_make_hashable(v) for v in val)
    return val


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _is_converted(element5: Any) -> bool:
    """Return True if element [5] is already in defaults+overrides form."""
    return isinstance(element5, dict) and "SolutionDefaults" in element5


def compute_defaults(solutions: list[dict]) -> dict:
    """Return a defaults dict: keys whose most-common value covers >50% of solutions."""
    if not solutions:
        return {}

    all_keys: set[str] = set()
    for s in solutions:
        all_keys.update(s.keys())

    defaults: dict[str, Any] = {}
    n = len(solutions)

    for key in sorted(all_keys):
        # Only keys present in EVERY solution are eligible to be defaulted.
        # Otherwise expansion (defaults + overrides) would resurrect the key on
        # the solutions that legitimately omitted it, breaking the round trip.
        if any(key not in s for s in solutions):
            continue

        counter: Counter = Counter()
        for s in solutions:
            if key in s:
                counter[_make_hashable(s[key])] += 1

        if not counter:
            continue

        most_common_hashable, count = counter.most_common(1)[0]
        if count / n > MAJORITY_THRESHOLD:
            for s in solutions:
                if key in s and _make_hashable(s[key]) == most_common_hashable:
                    defaults[key] = s[key]
                    break

    return defaults


def compute_overrides(solution: dict, defaults: dict) -> dict:
    """Return only the keys in *solution* that differ from *defaults*."""
    overrides: dict[str, Any] = {}
    for key, val in solution.items():
        if key not in defaults or _make_hashable(val) != _make_hashable(defaults[key]):
            overrides[key] = val
    return overrides


def expand_solution(overrides: dict, defaults: dict) -> dict:
    """Merge defaults with overrides to reconstruct a full solution dict."""
    full = copy.deepcopy(defaults)
    full.update(overrides)
    return full


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> list:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def dump_yaml(data: list, path: str) -> None:
    # Write to a temp file in the same directory then atomically replace, so a
    # crash/full-disk mid-write cannot leave the original file truncated.
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        yaml.dump(data, f, default_flow_style=None, Dumper=yaml.SafeDumper)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Apply / Revert / Stats
# ---------------------------------------------------------------------------

def apply_file(path: str, dry_run: bool = False) -> dict:
    """Convert a single file. Returns stats dict."""
    try:
        original_size = os.path.getsize(path)
        data = load_yaml(path)
    except Exception as e:
        return {"path": path, "skipped": True, "reason": f"load error: {e}"}

    if not isinstance(data, list) or len(data) < SOLUTIONS_INDEX + 1:
        return {"path": path, "skipped": True, "reason": "too few elements or not a list"}

    try:
        element5 = data[SOLUTIONS_INDEX]
    except (KeyError, IndexError, TypeError):
        return {"path": path, "skipped": True, "reason": "cannot access element[5]"}

    if _is_converted(element5):
        return {"path": path, "skipped": True, "reason": "already converted"}

    if not isinstance(element5, list):
        return {"path": path, "skipped": True, "reason": "element[5] is not a list"}

    solutions = element5
    num_solutions = len(solutions)
    if num_solutions == 0:
        return {"path": path, "skipped": True, "reason": "no solutions"}

    defaults = compute_defaults(solutions)
    override_list = [compute_overrides(s, defaults) for s in solutions]

    all_keys = set()
    for s in solutions:
        all_keys.update(s.keys())
    total_kv = sum(len(s) for s in solutions)
    override_kv = sum(len(o) for o in override_list)
    default_kv = len(defaults)

    new_element5 = {"SolutionDefaults": defaults, "Solutions": override_list}

    data[SOLUTIONS_INDEX] = new_element5

    if dry_run:
        serialized = yaml.dump(data, default_flow_style=None, Dumper=yaml.SafeDumper)
        new_size = len(serialized.encode("utf-8"))
    else:
        dump_yaml(data, path)
        new_size = os.path.getsize(path)

    return {
        "path": path,
        "skipped": False,
        "original_size": original_size,
        "new_size": new_size,
        "num_solutions": num_solutions,
        "total_keys": len(all_keys),
        "default_keys": default_kv,
        "total_kv_pairs": total_kv,
        "override_kv_pairs": override_kv,
    }


def revert_file(path: str) -> dict:
    """Revert a single file from defaults+overrides back to flat solutions."""
    try:
        original_size = os.path.getsize(path)
        data = load_yaml(path)
    except Exception as e:
        return {"path": path, "skipped": True, "reason": f"load error: {e}"}

    if not isinstance(data, list) or len(data) < SOLUTIONS_INDEX + 1:
        return {"path": path, "skipped": True, "reason": "too few elements or not a list"}

    try:
        element5 = data[SOLUTIONS_INDEX]
    except (KeyError, IndexError, TypeError):
        return {"path": path, "skipped": True, "reason": "cannot access element[5]"}

    if not _is_converted(element5):
        return {"path": path, "skipped": True, "reason": "not in converted format"}

    defaults = element5["SolutionDefaults"]
    overrides_list = element5["Solutions"]

    full_solutions = [expand_solution(o, defaults) for o in overrides_list]
    data[SOLUTIONS_INDEX] = full_solutions

    dump_yaml(data, path)
    new_size = os.path.getsize(path)

    return {
        "path": path,
        "skipped": False,
        "original_size": original_size,
        "new_size": new_size,
        "num_solutions": len(full_solutions),
    }


def stats_file(path: str) -> dict:
    """Compute stats for a single file (dry-run, no modification)."""
    return apply_file(path, dry_run=True)


def fullstats_file(path: str) -> dict:
    """Compute flat-YAML, d+o-YAML, and d+o-JSON sizes for a single file."""
    try:
        data = load_yaml(path)
    except Exception as e:
        return {"path": path, "skipped": True, "reason": f"load error: {e}"}

    if not isinstance(data, list) or len(data) < SOLUTIONS_INDEX + 1:
        return {"path": path, "skipped": True, "reason": "too few elements or not a list"}

    element5 = data[SOLUTIONS_INDEX]

    if _is_converted(element5):
        defaults = element5["SolutionDefaults"]
        flat_solutions = [expand_solution(o, defaults) for o in element5["Solutions"]]
    elif isinstance(element5, list):
        flat_solutions = element5
    else:
        return {"path": path, "skipped": True, "reason": "element[5] is not a list or dict"}

    if len(flat_solutions) == 0:
        return {"path": path, "skipped": True, "reason": "no solutions"}

    num_solutions = len(flat_solutions)

    data_flat = list(data)
    data_flat[SOLUTIONS_INDEX] = flat_solutions
    flat_yaml_size = len(yaml.dump(data_flat, default_flow_style=None, Dumper=yaml.SafeDumper).encode("utf-8"))

    new_defaults = compute_defaults(flat_solutions)
    override_list = [compute_overrides(s, new_defaults) for s in flat_solutions]
    data_do = list(data)
    data_do[SOLUTIONS_INDEX] = {"SolutionDefaults": new_defaults, "Solutions": override_list}

    do_yaml_size = len(yaml.dump(data_do, default_flow_style=None, Dumper=yaml.SafeDumper).encode("utf-8"))

    do_json_size = len(json.dumps(data_do, separators=(",", ":")).encode("utf-8"))

    return {
        "path": path,
        "skipped": False,
        "category": classify_category(path),
        "num_solutions": num_solutions,
        "flat_yaml_size": flat_yaml_size,
        "do_yaml_size": do_yaml_size,
        "do_json_size": do_json_size,
    }


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_paths(path_or_glob: str, category: str | None = None) -> list[str]:
    """Resolve a path/glob to a list of YAML file paths."""
    paths: list[str] = []

    if "*" in path_or_glob or "?" in path_or_glob:
        paths = sorted(glob.glob(path_or_glob, recursive=True))
    elif os.path.isdir(path_or_glob):
        for root, _dirs, files in os.walk(path_or_glob):
            for f in sorted(files):
                if f.endswith(".yaml") or f.endswith(".yml"):
                    paths.append(os.path.join(root, f))
    elif os.path.isfile(path_or_glob):
        paths = [path_or_glob]
    else:
        print(f"Error: '{path_or_glob}' is not a valid file, directory, or glob pattern.",
              file=sys.stderr)
        sys.exit(1)

    if category and category != "all":
        filtered = []
        for p in paths:
            parts = Path(p).parts
            if category in parts:
                filtered.append(p)
        paths = filtered

    return paths


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_size(nbytes: int) -> str:
    if nbytes < 1024:
        return f"{nbytes} B"
    if nbytes < 1024 * 1024:
        return f"{nbytes / 1024:.1f} KB"
    return f"{nbytes / (1024 * 1024):.2f} MB"


def _pct(old: int, new: int) -> str:
    if old == 0:
        return "N/A"
    reduction = (1 - new / old) * 100
    return f"{reduction:.1f}%"


def classify_category(path: str) -> str:
    parts = Path(path).parts
    for cat in CATEGORIES:
        if cat in parts:
            return cat
    return "Other"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_apply(args: argparse.Namespace) -> None:
    paths = resolve_paths(args.path, getattr(args, "category", None))
    if not paths:
        print("No files found.")
        return

    action = "Would convert" if args.dry_run else "Converting"
    print(f"{action} {len(paths)} file(s)...\n")

    total_orig = 0
    total_new = 0
    converted = 0

    for p in paths:
        result = apply_file(p, dry_run=args.dry_run)
        fname = os.path.basename(p)

        if result.get("skipped"):
            print(f"  SKIP  {fname}: {result['reason']}")
            continue

        converted += 1
        orig = result["original_size"]
        new = result["new_size"]
        total_orig += orig
        total_new += new

        print(
            f"  {'WOULD' if args.dry_run else 'OK'}    {fname}: "
            f"{_fmt_size(orig)} -> {_fmt_size(new)} "
            f"({_pct(orig, new)} reduction, "
            f"{result['num_solutions']} solutions, "
            f"{result['default_keys']}/{result['total_keys']} keys defaulted)"
        )

    if converted:
        print(f"\nTotal: {_fmt_size(total_orig)} -> {_fmt_size(total_new)} "
              f"({_pct(total_orig, total_new)} reduction across {converted} file(s))")
    else:
        print("\nNo files were converted.")


def cmd_stats(args: argparse.Namespace) -> None:
    paths = resolve_paths(args.path, getattr(args, "category", None))
    if not paths:
        print("No files found.")
        return

    workers = getattr(args, "jobs", None) or min(multiprocessing.cpu_count(), len(paths))
    print(f"Analyzing {len(paths)} file(s) with {workers} workers...\n")
    print(f"{'File':<70} {'Orig':>10} {'New':>10} {'Savings':>8} "
          f"{'#Sol':>5} {'Def/Tot':>8} {'KV Reduc':>9}")
    print("-" * 125)

    total_orig = 0
    total_new = 0
    total_solutions = 0
    analyzed = 0

    with multiprocessing.Pool(workers) as pool:
        for result in pool.imap_unordered(stats_file, paths):
            fname = os.path.basename(result["path"])

            if result.get("skipped"):
                print(f"  SKIP  {fname}: {result['reason']}")
                continue

            analyzed += 1
            orig = result["original_size"]
            new = result["new_size"]
            total_orig += orig
            total_new += new
            total_solutions += result["num_solutions"]

            kv_reduc = _pct(result["total_kv_pairs"], result["override_kv_pairs"])

            print(
                f"  {fname:<68} {_fmt_size(orig):>10} {_fmt_size(new):>10} "
                f"{_pct(orig, new):>8} {result['num_solutions']:>5} "
                f"{result['default_keys']:>3}/{result['total_keys']:<4} "
                f"{kv_reduc:>9}"
            )

    if analyzed:
        print("-" * 125)
        print(
            f"  {'TOTAL':<68} {_fmt_size(total_orig):>10} {_fmt_size(total_new):>10} "
            f"{_pct(total_orig, total_new):>8} {total_solutions:>5}"
        )
        print(f"\n  Analyzed {analyzed} file(s), {total_solutions} total solutions.")
        print(f"  Aggregate: {_fmt_size(total_orig)} -> {_fmt_size(total_new)} "
              f"({_pct(total_orig, total_new)} file-size reduction)")
    else:
        print("\nNo files were analyzed.")


def cmd_revert(args: argparse.Namespace) -> None:
    paths = resolve_paths(args.path, getattr(args, "category", None))
    if not paths:
        print("No files found.")
        return

    print(f"Reverting {len(paths)} file(s)...\n")

    total_orig = 0
    total_new = 0
    reverted = 0

    for p in paths:
        result = revert_file(p)
        fname = os.path.basename(p)

        if result.get("skipped"):
            print(f"  SKIP  {fname}: {result['reason']}")
            continue

        reverted += 1
        orig = result["original_size"]
        new = result["new_size"]
        total_orig += orig
        total_new += new

        print(
            f"  OK    {fname}: "
            f"{_fmt_size(orig)} -> {_fmt_size(new)} "
            f"({result['num_solutions']} solutions restored)"
        )

    if reverted:
        print(f"\nReverted {reverted} file(s). "
              f"Total: {_fmt_size(total_orig)} -> {_fmt_size(total_new)}")
    else:
        print("\nNo files were reverted.")


def cmd_fullstats(args: argparse.Namespace) -> None:
    paths = resolve_paths(args.path)
    if not paths:
        print("No files found.")
        return

    workers = args.jobs or min(multiprocessing.cpu_count(), len(paths))
    print(f"Analyzing {len(paths)} file(s) with {workers} workers...", file=sys.stderr)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(fullstats_file, paths))

    by_cat: dict[str, list[dict]] = {}
    skipped = 0
    for r in results:
        if r.get("skipped"):
            skipped += 1
            continue
        by_cat.setdefault(r["category"], []).append(r)

    if skipped:
        print(f"  ({skipped} file(s) skipped)", file=sys.stderr)

    rows = []
    for cat in sorted(by_cat.keys()):
        items = by_cat[cat]
        row = {
            "category": cat,
            "files": len(items),
            "solutions": sum(r["num_solutions"] for r in items),
            "flat_yaml_mb": sum(r["flat_yaml_size"] for r in items) / (1024 * 1024),
            "do_yaml_mb": sum(r["do_yaml_size"] for r in items) / (1024 * 1024),
            "do_json_mb": sum(r["do_json_size"] for r in items) / (1024 * 1024),
        }
        flat = row["flat_yaml_mb"]
        row["do_pct"] = (1 - row["do_yaml_mb"] / flat) * 100 if flat else 0
        row["json_on_do_pct"] = (1 - row["do_json_mb"] / row["do_yaml_mb"]) * 100 if row["do_yaml_mb"] else 0
        row["cumul_pct"] = (1 - row["do_json_mb"] / flat) * 100 if flat else 0
        rows.append(row)

    total = {
        "category": "TOTAL",
        "files": sum(r["files"] for r in rows),
        "solutions": sum(r["solutions"] for r in rows),
        "flat_yaml_mb": sum(r["flat_yaml_mb"] for r in rows),
        "do_yaml_mb": sum(r["do_yaml_mb"] for r in rows),
        "do_json_mb": sum(r["do_json_mb"] for r in rows),
    }
    flat = total["flat_yaml_mb"]
    total["do_pct"] = (1 - total["do_yaml_mb"] / flat) * 100 if flat else 0
    total["json_on_do_pct"] = (1 - total["do_json_mb"] / total["do_yaml_mb"]) * 100 if total["do_yaml_mb"] else 0
    total["cumul_pct"] = (1 - total["do_json_mb"] / flat) * 100 if flat else 0

    if args.json_output:
        print(json.dumps({"categories": rows, "total": total}, indent=2))
        return

    hdr = f"{'Category':<15} {'Files':>6} {'#Sol':>7}  {'Orig(MB)':>10} {'D+O(MB)':>10} {'D+O %':>7}  {'JSON(MB)':>10} {'J/DO %':>7} {'Cumul %':>8}"
    print(hdr)
    print("-" * len(hdr))
    for row in rows:
        print(
            f"  {row['category']:<13} {row['files']:>6} {row['solutions']:>7}  "
            f"{row['flat_yaml_mb']:>10.1f} {row['do_yaml_mb']:>10.1f} {row['do_pct']:>6.1f}%  "
            f"{row['do_json_mb']:>10.1f} {row['json_on_do_pct']:>6.1f}% {row['cumul_pct']:>7.1f}%"
        )
    print("-" * len(hdr))
    print(
        f"  {total['category']:<13} {total['files']:>6} {total['solutions']:>7}  "
        f"{total['flat_yaml_mb']:>10.1f} {total['do_yaml_mb']:>10.1f} {total['do_pct']:>6.1f}%  "
        f"{total['do_json_mb']:>10.1f} {total['json_on_do_pct']:>6.1f}% {total['cumul_pct']:>7.1f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert TensileLite Logic YAML to defaults+overrides format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # apply
    p_apply = subparsers.add_parser("apply", help="Convert files to defaults+overrides format")
    p_apply.add_argument("path", help="File, directory, or glob pattern")
    p_apply.add_argument("--dry-run", action="store_true",
                         help="Show what would happen without modifying files")
    p_apply.add_argument("--category", choices=["StreamK", "Equality", "GridBased", "all"],
                         default=None, help="Filter by category subdirectory")

    # stats
    p_stats = subparsers.add_parser("stats", help="Show savings statistics (dry-run)")
    p_stats.add_argument("path", help="File, directory, or glob pattern")
    p_stats.add_argument("--category", choices=["StreamK", "Equality", "GridBased", "all"],
                         default=None, help="Filter by category subdirectory")
    p_stats.add_argument("-j", "--jobs", type=int, default=None,
                         help="Number of parallel workers (default: CPU count)")

    # revert
    p_revert = subparsers.add_parser("revert", help="Revert to flat solution format")
    p_revert.add_argument("path", help="File, directory, or glob pattern")
    p_revert.add_argument("--category", choices=["StreamK", "Equality", "GridBased", "all"],
                          default=None, help="Filter by category subdirectory")

    # fullstats
    p_fullstats = subparsers.add_parser("fullstats",
        help="Compute flat/d+o/JSON sizes for all files (read-only)")
    p_fullstats.add_argument("path", help="File, directory, or glob pattern")
    p_fullstats.add_argument("-j", "--jobs", type=int, default=None,
                             help="Number of parallel workers (default: CPU count)")
    p_fullstats.add_argument("--json", dest="json_output", action="store_true",
                             help="Emit machine-readable JSON output")

    args = parser.parse_args()

    if args.command == "apply":
        cmd_apply(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "revert":
        cmd_revert(args)
    elif args.command == "fullstats":
        cmd_fullstats(args)


if __name__ == "__main__":
    main()
