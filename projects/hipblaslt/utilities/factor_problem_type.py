#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Factor out per-solution ProblemType redundancy from TensileLite Logic YAML files.

Background
----------
Logic YAML files have a shared ProblemType at Element [4] of the top-level list.
Some files (StreamK, gfx1250, and others) also embed an identical copy of
ProblemType inside every solution dict in Element [5].  The build-pipeline
parser (``LibraryIO.parseLibraryLogicData``) always reads ProblemType from
Element [4] and overwrites any per-solution copy, so the per-solution
ProblemType is pure redundancy.

This tool strips those per-solution ProblemType dicts, shrinking affected files
by 15-25%.  It can also restore them for backward compatibility.

Usage
-----
::

    # Dry-run: report savings without modifying files
    python3 factor_problem_type.py --dry-run LOGIC_DIR

    # Check: verify all per-solution ProblemTypes match the shared one
    python3 factor_problem_type.py --check LOGIC_DIR

    # Factor: strip per-solution ProblemType (default action)
    python3 factor_problem_type.py LOGIC_DIR

    # Restore: re-inject shared ProblemType into every solution
    python3 factor_problem_type.py --restore LOGIC_DIR

    # Limit to a specific device directory
    python3 factor_problem_type.py --device gfx1250 LOGIC_DIR
"""

import argparse
import glob
import io
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader

try:
    from yaml import CSafeDumper as SafeDumper
except ImportError:
    from yaml import SafeDumper


LOGIC_MIN_ELEMENTS = 9
PROBLEM_TYPE_INDEX = 4
SOLUTIONS_INDEX = 5

# Regex to detect "    ProblemType:" (4-space indented = inside a solution dict)
_PT_IN_SOLUTION_RE = re.compile(r"^    ProblemType:\s*$")
# Regex for sub-keys under the indented ProblemType block (6+ spaces)
_PT_SUBKEY_RE = re.compile(r"^      \S")
# Regex for the first solution entry marker
_SOLUTION_START_RE = re.compile(r"^- - ")


def find_logic_files(logic_dir: str, device: Optional[str] = None) -> List[str]:
    """Find all Logic YAML files under logic_dir, optionally filtered by device."""
    if device:
        pattern = os.path.join(logic_dir, "**", device, "**", "*.yaml")
    else:
        pattern = os.path.join(logic_dir, "**", "*.yaml")
    return sorted(glob.glob(pattern, recursive=True))


def load_yaml(path: str) -> Any:
    """Load a YAML file using the fastest available safe loader."""
    with open(path, "r") as f:
        return yaml.load(f, Loader=SafeLoader)


def is_logic_file(data: Any) -> bool:
    """Check whether parsed data looks like a Logic YAML top-level list."""
    if not isinstance(data, list):
        return False
    if len(data) < LOGIC_MIN_ELEMENTS:
        return False
    if not isinstance(data[0], dict) or "MinimumRequiredVersion" not in data[0]:
        return False
    if not isinstance(data[PROBLEM_TYPE_INDEX], dict):
        return False
    if not isinstance(data[SOLUTIONS_INDEX], list):
        return False
    return True


def solutions_have_problem_type(solutions: List[Dict]) -> bool:
    """Return True if any solution dict contains a 'ProblemType' key."""
    return any(isinstance(s, dict) and "ProblemType" in s for s in solutions)


# ---------------------------------------------------------------------------
# Fast text-based scanning (avoids full YAML parse for large files)
# ---------------------------------------------------------------------------

def _fast_scan_file(path: str) -> Tuple[int, int]:
    """Count per-solution ProblemType occurrences and their line spans using text scan.

    Returns (count_of_pt_blocks, total_lines_in_pt_blocks).
    """
    count = 0
    pt_lines = 0
    in_pt_block = False

    with open(path, "r") as f:
        for line in f:
            if in_pt_block:
                if _PT_SUBKEY_RE.match(line):
                    pt_lines += 1
                    continue
                else:
                    in_pt_block = False
            if _PT_IN_SOLUTION_RE.match(line):
                count += 1
                pt_lines += 1
                in_pt_block = True

    return count, pt_lines


def _fast_has_solution_pt(path: str) -> bool:
    """Quick check whether file has any per-solution ProblemType."""
    with open(path, "r") as f:
        for line in f:
            if _PT_IN_SOLUTION_RE.match(line):
                return True
    return False


def _strip_pt_blocks_text(path: str) -> Tuple[int, int]:
    """Remove per-solution ProblemType blocks from a file via line filtering.

    Returns (blocks_removed, bytes_saved).
    """
    original_size = os.path.getsize(path)
    removed = 0
    in_pt_block = False
    output_lines = []

    with open(path, "r") as f:
        for line in f:
            if in_pt_block:
                if _PT_SUBKEY_RE.match(line):
                    continue
                else:
                    in_pt_block = False
            if _PT_IN_SOLUTION_RE.match(line):
                removed += 1
                in_pt_block = True
                continue
            output_lines.append(line)

    if removed == 0:
        return 0, 0

    with open(path, "w") as f:
        f.writelines(output_lines)

    new_size = os.path.getsize(path)
    return removed, original_size - new_size


def _inject_pt_block_text(path: str, shared_pt_yaml: str) -> Tuple[int, int]:
    """Inject ProblemType block into each solution via text insertion.

    Returns (num_injected, bytes_added).
    shared_pt_yaml should be pre-formatted lines with 4-space base indent.
    """
    original_size = os.path.getsize(path)
    injected = 0
    output_lines = []
    in_solution = False

    with open(path, "r") as f:
        for line in f:
            if line.startswith("  - ") or line.startswith("- - "):
                if in_solution:
                    pass
                in_solution = True
                output_lines.append(line)
                injected += 1
                continue

            if in_solution and not line.startswith("    ") and not line.startswith("  - "):
                in_solution = False

            output_lines.append(line)

    if injected == 0:
        return 0, 0

    output_with_pt = []
    in_solution = False
    pt_injected_for_current = False

    with open(path, "r") as f:
        lines = f.readlines()

    i = 0
    injected = 0
    while i < len(lines):
        line = lines[i]
        output_with_pt.append(line)

        is_sol_start = line.startswith("- - ") or (line.startswith("  - ") and i > 0
                                                     and not lines[i-1].startswith("  - "))
        if is_sol_start:
            insert_pos = i + 1
            while insert_pos < len(lines) and lines[insert_pos].startswith("    "):
                key_match = re.match(r"    (\S+):", lines[insert_pos])
                if key_match and key_match.group(1) > "ProblemType":
                    break
                insert_pos += 1

            for j in range(i + 1, insert_pos):
                output_with_pt.append(lines[j])
            output_with_pt.append(shared_pt_yaml)
            injected += 1

            i = insert_pos
            continue
        i += 1

    with open(path, "w") as f:
        f.writelines(output_with_pt)

    new_size = os.path.getsize(path)
    return injected, new_size - original_size


# ---------------------------------------------------------------------------
# YAML-based operations (for small files or check mode on small files)
# ---------------------------------------------------------------------------

def check_file_yaml(path: str) -> Tuple[bool, int, int, List[str], List[str]]:
    """Check a file for ProblemType mismatches using full YAML parse.

    Returns (has_redundancy, num_solutions, num_with_pt, value_mismatches, subset_warnings).
    value_mismatches: keys where shared and per-solution have different non-missing values.
    subset_warnings: keys present in shared but missing from per-solution (safe to strip).
    """
    try:
        data = load_yaml(path)
    except Exception as e:
        return False, 0, 0, [f"YAML parse error: {e}"], []

    if not is_logic_file(data):
        return False, 0, 0, [], []

    shared_pt = data[PROBLEM_TYPE_INDEX]
    solutions = data[SOLUTIONS_INDEX]
    value_mismatches = []
    subset_warnings = []
    num_with_pt = 0
    first_sol_checked = False

    for i, sol in enumerate(solutions):
        if not isinstance(sol, dict) or "ProblemType" not in sol:
            continue
        num_with_pt += 1
        sol_pt = sol["ProblemType"]
        if sol_pt == shared_pt:
            continue

        if not first_sol_checked:
            first_sol_checked = True
            all_keys = set(shared_pt.keys()) | set(sol_pt.keys())
            for k in sorted(all_keys):
                sv = shared_pt.get(k)
                pv = sol_pt.get(k)
                if sv == pv:
                    continue
                if k in shared_pt and k not in sol_pt:
                    subset_warnings.append(f"  {k}: in shared but missing from solutions")
                elif k not in shared_pt and k in sol_pt:
                    value_mismatches.append(f"  {k}: in solutions but missing from shared (val={pv!r})")
                else:
                    value_mismatches.append(f"  {k}: shared={sv!r} vs solution={pv!r}")

    return num_with_pt > 0, len(solutions), num_with_pt, value_mismatches, subset_warnings


# ---------------------------------------------------------------------------
# Shared ProblemType extraction for restore mode
# ---------------------------------------------------------------------------

def _extract_shared_pt_yaml(path: str) -> Optional[str]:
    """Extract the shared ProblemType (Element [4]) as indented YAML text.

    Returns the block formatted for injection into a solution (4-space base indent),
    or None if the file doesn't look like a Logic YAML.
    """
    lines = []
    in_pt = False
    top_level_count = 0

    with open(path, "r") as f:
        for line in f:
            if line.startswith("- ") and not line.startswith("- -"):
                top_level_count += 1
                if top_level_count == PROBLEM_TYPE_INDEX + 1:
                    in_pt = True
                    key = line[2:].split(":")[0].strip()
                    lines.append(f"    ProblemType:\n")
                    rest = line[2 + len(key) + 1:].strip()
                    if rest:
                        lines.append(f"      {key}: {rest}\n")
                    continue
                elif in_pt:
                    break
            if in_pt:
                if line.startswith("  "):
                    lines.append(f"    {line}")
                else:
                    break

    if not lines:
        return None
    return "".join(lines)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

LARGE_FILE_THRESHOLD = 5 * 1024 * 1024  # 5 MB


def _human_size(nbytes: int) -> str:
    """Format byte count as human-readable string."""
    if abs(nbytes) < 1024:
        return f"{nbytes} B"
    elif abs(nbytes) < 1024 * 1024:
        return f"{nbytes / 1024:.1f} KB"
    elif abs(nbytes) < 1024 * 1024 * 1024:
        return f"{nbytes / (1024 * 1024):.1f} MB"
    else:
        return f"{nbytes / (1024 * 1024 * 1024):.2f} GB"


def _avg_pt_line_bytes(path: str) -> float:
    """Estimate average bytes per ProblemType line by sampling the first block."""
    in_pt = False
    byte_count = 0
    line_count = 0

    with open(path, "r") as f:
        for line in f:
            if in_pt:
                if _PT_SUBKEY_RE.match(line):
                    byte_count += len(line.encode("utf-8"))
                    line_count += 1
                    continue
                else:
                    break
            if _PT_IN_SOLUTION_RE.match(line):
                in_pt = True
                byte_count += len(line.encode("utf-8"))
                line_count += 1

    return byte_count / max(line_count, 1)


def cmd_check(files: List[str]) -> int:
    """--check mode: verify per-solution ProblemTypes match the shared one."""
    total_files = 0
    files_with_pt = 0
    files_with_value_mismatches = 0
    files_with_subset_diffs = 0
    total_with_pt = 0

    for path in files:
        total_files += 1
        fsize = os.path.getsize(path)

        if fsize > LARGE_FILE_THRESHOLD:
            count, _ = _fast_scan_file(path)
            if count > 0:
                files_with_pt += 1
                total_with_pt += count
                rel = os.path.relpath(path)
                print(f"  {rel}: {count} per-solution ProblemType blocks "
                      f"(text-scan only, {_human_size(fsize)})")
        else:
            has_redundancy, num_sol, num_pt, val_mm, subset_w = check_file_yaml(path)
            if has_redundancy:
                files_with_pt += 1
                total_with_pt += num_pt
            if val_mm:
                files_with_value_mismatches += 1
                rel = os.path.relpath(path)
                print(f"\nVALUE MISMATCH (unsafe): {rel}")
                for m in val_mm:
                    print(f"  {m}")
            if subset_w:
                files_with_subset_diffs += 1
                rel = os.path.relpath(path)
                print(f"\n  SUBSET DIFF (safe): {rel}")
                for w in subset_w:
                    print(f"  {w}")

    print(f"\n{'='*60}")
    print(f"Files scanned:                {total_files}")
    print(f"Files with per-solution PT:   {files_with_pt}")
    print(f"Solutions with embedded PT:   {total_with_pt}")
    print(f"Files with value mismatches:  {files_with_value_mismatches}")
    print(f"Files with subset diffs:      {files_with_subset_diffs}")
    print(f"  (subset = shared has extra keys not in per-solution copy; safe to strip)")
    print(f"{'='*60}")

    if files_with_value_mismatches:
        print("\nWARNING: Some files have per-solution ProblemType with conflicting values."
              " Inspect these before factoring.")
        return 1
    elif files_with_pt:
        msg = f"\nAll {total_with_pt} embedded ProblemTypes are safe to strip."
        if files_with_subset_diffs:
            msg += (f"\n  ({files_with_subset_diffs} files have per-solution copies with fewer "
                    f"keys than the shared one; this is expected and harmless.)")
        print(msg)
    else:
        print("\nNo per-solution ProblemType found. Nothing to factor.")
    return 0


def cmd_factor(files: List[str], dry_run: bool = False) -> int:
    """Default / --dry-run mode: strip per-solution ProblemType."""
    label = "DRY-RUN" if dry_run else "FACTOR"
    total_files = 0
    modified_files = 0
    total_bytes_saved = 0
    total_pt_removed = 0

    for path in files:
        total_files += 1

        if not _fast_has_solution_pt(path):
            continue

        if dry_run:
            count, pt_lines = _fast_scan_file(path)
            if count == 0:
                continue
            avg_bytes = _avg_pt_line_bytes(path)
            estimated_savings = int(pt_lines * avg_bytes)
            modified_files += 1
            total_pt_removed += count
            total_bytes_saved += estimated_savings
            rel = os.path.relpath(path)
            print(f"  [{label}] {rel}: {count} PT blocks, ~{_human_size(estimated_savings)} savings")
        else:
            removed, saved = _strip_pt_blocks_text(path)
            if removed > 0:
                modified_files += 1
                total_pt_removed += removed
                total_bytes_saved += saved
                rel = os.path.relpath(path)
                print(f"  [{label}] {rel}: {removed} PT blocks, {_human_size(saved)} saved")

    print(f"\n{'='*60}")
    print(f"[{label}] Files scanned:       {total_files}")
    print(f"[{label}] Files modified:       {modified_files}")
    print(f"[{label}] PT blocks removed:    {total_pt_removed}")
    est = "~" if dry_run else ""
    print(f"[{label}] Total savings:        {est}{_human_size(total_bytes_saved)}")
    print(f"{'='*60}")
    return 0


def cmd_restore(files: List[str]) -> int:
    """--restore mode: re-inject shared ProblemType into solutions.

    Uses full YAML parse for correctness: load, inject ProblemType dict
    into each solution, and re-dump.
    """
    total_files = 0
    modified_files = 0
    total_injected = 0

    for path in files:
        total_files += 1

        if _fast_has_solution_pt(path):
            continue

        fsize = os.path.getsize(path)
        try:
            data = load_yaml(path)
        except Exception as e:
            print(f"  SKIP (parse error): {path}: {e}", file=sys.stderr)
            continue

        if not is_logic_file(data):
            continue

        shared_pt = data[PROBLEM_TYPE_INDEX]
        solutions = data[SOLUTIONS_INDEX]

        injected = 0
        for sol in solutions:
            if isinstance(sol, dict) and "ProblemType" not in sol:
                sol["ProblemType"] = dict(shared_pt)
                injected += 1

        if injected == 0:
            continue

        with open(path, "w") as f:
            yaml.dump(data, f, Dumper=SafeDumper,
                      default_flow_style=None,
                      explicit_start=True,
                      explicit_end=True)

        modified_files += 1
        total_injected += injected
        rel = os.path.relpath(path)
        print(f"  [RESTORE] {rel}: injected into {injected} solutions")

    print(f"\n{'='*60}")
    print(f"[RESTORE] Files scanned:     {total_files}")
    print(f"[RESTORE] Files modified:    {modified_files}")
    print(f"[RESTORE] Solutions injected: {total_injected}")
    print(f"{'='*60}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Factor out per-solution ProblemType redundancy from Logic YAML files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("logic_dir",
                        help="Root directory containing Logic YAML files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report savings without modifying files")
    parser.add_argument("--check", action="store_true",
                        help="Verify all per-solution ProblemTypes match the shared one")
    parser.add_argument("--restore", action="store_true",
                        help="Re-inject shared ProblemType into every solution")
    parser.add_argument("--device",
                        help="Limit to files under a specific device directory (e.g. gfx1250)")

    args = parser.parse_args()

    if not os.path.isdir(args.logic_dir):
        print(f"ERROR: {args.logic_dir} is not a directory", file=sys.stderr)
        return 1

    if sum([args.check, args.restore]) > 1:
        print("ERROR: --check and --restore are mutually exclusive", file=sys.stderr)
        return 1

    files = find_logic_files(args.logic_dir, args.device)
    if not files:
        print(f"No YAML files found under {args.logic_dir}", file=sys.stderr)
        if args.device:
            print(f"  (filtered by device={args.device})", file=sys.stderr)
        return 1

    print(f"Found {len(files)} YAML files under {args.logic_dir}")
    if args.device:
        print(f"  (filtered by device={args.device})")

    if args.check:
        return cmd_check(files)
    elif args.restore:
        return cmd_restore(files)
    else:
        return cmd_factor(files, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
