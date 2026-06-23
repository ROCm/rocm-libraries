# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Convert TensileLite Logic YAML files to JSON format.

Works on top of the defaults+overrides pattern — files already converted
to {SolutionDefaults, Solutions} are serialized faithfully into JSON.

Modes:
  convert – write a .json file for each .yaml input
  stats   – dry-run showing per-file and aggregate size comparisons
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import yaml


LOGIC_BASE = (
    "projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full"
)

CATEGORIES = ("StreamK", "Equality", "GridBased", "Experimental", "FreeSize")


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> list:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def write_json(data, path: str, pretty: bool = False) -> None:
    with open(path, "w") as f:
        if pretty:
            json.dump(data, f, indent=2)
        else:
            json.dump(data, f, indent=None, separators=(",", ":"))


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
        paths = [p for p in paths if category in Path(p).parts]

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


# ---------------------------------------------------------------------------
# Convert / Stats
# ---------------------------------------------------------------------------

def convert_file(path: str, delete_original: bool = False, pretty: bool = False,
                 dry_run: bool = False) -> dict:
    """Convert a single YAML file to JSON. Returns stats dict."""
    yaml_size = os.path.getsize(path)
    data = load_yaml(path)

    json_path = os.path.splitext(path)[0] + ".json"

    if dry_run:
        json_bytes = json.dumps(data, indent=2 if pretty else None,
                                separators=None if pretty else (",", ":"))
        json_size = len(json_bytes.encode("utf-8"))
    else:
        write_json(data, json_path, pretty=pretty)
        json_size = os.path.getsize(json_path)
        if delete_original:
            os.remove(path)

    return {
        "path": path,
        "json_path": json_path,
        "yaml_size": yaml_size,
        "json_size": json_size,
    }


def stats_file(path: str, pretty: bool = False) -> dict:
    """Compute stats for a single file (dry-run, no modification)."""
    return convert_file(path, dry_run=True, pretty=pretty)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_convert(args: argparse.Namespace) -> None:
    paths = resolve_paths(args.path, getattr(args, "category", None))
    if not paths:
        print("No YAML files found.")
        return

    mode = "compact" if not args.pretty else "pretty"
    print(f"Converting {len(paths)} file(s) to JSON ({mode})...\n")

    total_yaml = 0
    total_json = 0

    for p in paths:
        result = convert_file(p, delete_original=args.delete_original,
                              pretty=args.pretty)
        fname = os.path.basename(p)
        yaml_sz = result["yaml_size"]
        json_sz = result["json_size"]
        total_yaml += yaml_sz
        total_json += json_sz

        action = "DEL+JSON" if args.delete_original else "JSON"
        print(f"  {action}  {fname}: "
              f"{_fmt_size(yaml_sz)} -> {_fmt_size(json_sz)} ({_pct(yaml_sz, json_sz)} reduction)")

    print(f"\nTotal: {_fmt_size(total_yaml)} (YAML) -> {_fmt_size(total_json)} (JSON) "
          f"({_pct(total_yaml, total_json)} reduction across {len(paths)} file(s))")


def cmd_stats(args: argparse.Namespace) -> None:
    paths = resolve_paths(args.path, getattr(args, "category", None))
    if not paths:
        print("No YAML files found.")
        return

    mode = "compact" if not args.pretty else "pretty"
    print(f"Analyzing {len(paths)} file(s) — JSON {mode} mode...\n")
    print(f"{'File':<70} {'YAML':>10} {'JSON':>10} {'Savings':>8}")
    print("-" * 102)

    total_yaml = 0
    total_json = 0

    for p in paths:
        result = stats_file(p, pretty=args.pretty)
        fname = os.path.basename(p)
        yaml_sz = result["yaml_size"]
        json_sz = result["json_size"]
        total_yaml += yaml_sz
        total_json += json_sz

        print(f"  {fname:<68} {_fmt_size(yaml_sz):>10} {_fmt_size(json_sz):>10} "
              f"{_pct(yaml_sz, json_sz):>8}")

    print("-" * 102)
    print(f"  {'TOTAL':<68} {_fmt_size(total_yaml):>10} {_fmt_size(total_json):>10} "
          f"{_pct(total_yaml, total_json):>8}")
    print(f"\n  {len(paths)} file(s) analyzed.")
    print(f"  Aggregate: {_fmt_size(total_yaml)} (YAML) -> {_fmt_size(total_json)} (JSON) "
          f"({_pct(total_yaml, total_json)} reduction)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert TensileLite Logic YAML files to JSON format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_convert = subparsers.add_parser("convert", help="Convert YAML files to JSON")
    p_convert.add_argument("path", help="File, directory, or glob pattern")
    p_convert.add_argument("--category",
                           choices=["StreamK", "Equality", "GridBased", "all"],
                           default=None, help="Filter by category subdirectory")
    p_convert.add_argument("--delete-original", action="store_true",
                           help="Delete original .yaml file after successful conversion")
    p_convert.add_argument("--pretty", action="store_true",
                           help="Use indent=2 for human-readable JSON (larger files)")

    p_stats = subparsers.add_parser("stats", help="Show size comparison statistics (dry-run)")
    p_stats.add_argument("path", help="File, directory, or glob pattern")
    p_stats.add_argument("--category",
                         choices=["StreamK", "Equality", "GridBased", "all"],
                         default=None, help="Filter by category subdirectory")
    p_stats.add_argument("--pretty", action="store_true",
                         help="Compute stats for pretty-printed JSON (indent=2)")

    args = parser.parse_args()

    if args.command == "convert":
        cmd_convert(args)
    elif args.command == "stats":
        cmd_stats(args)


if __name__ == "__main__":
    main()
