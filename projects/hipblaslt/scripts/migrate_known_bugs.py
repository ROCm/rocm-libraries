#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Resolve and verify the TensileLogic known-bugs list against the library logic.
#
# The known-bugs YAML keys each documented --check-all skip on
# (path, solution_name), where solution_name is the solution's stable
# SolutionNameMin. This helper:
#   * migrates a legacy (path, solution_index) list to (path, solution_name) by
#     resolving each index against the on-disk logic files (one-time upgrade), and
#   * verifies that every (path, solution_name) still resolves to a live solution,
#     flagging entries whose kernel was removed or renamed (stale).
#
# It parses the logic YAMLs with PyYAML directly (no rocisa / Tensile import), so
# it runs without a build. Solutions live at index 5 of each logic file's
# top-level list, and each carries "SolutionIndex" and "SolutionNameMin".
#
# How to run (from the hipblaslt project root):
#   python scripts/migrate_known_bugs.py            # verify only, exit 1 if stale
#   python scripts/migrate_known_bugs.py --stdout    # print migrated YAML
#   python scripts/migrate_known_bugs.py --write      # rewrite known_bugs.yaml in place

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import yaml
except ImportError:
    print(
        "Error: this script requires PyYAML. Install with: pip install PyYAML",
        file=sys.stderr,
    )
    raise SystemExit(1)


HEADER = """\
# Documented exceptions for TensileLogic --check-all (MatrixInstruction / WorkGroup / XCC).
# Paths use forward slashes, relative to the library logic root (the LogicPath argument).
#
# Each skip is keyed on (path, solution_name). solution_name is the solution's
# SolutionNameMin: a canonical, content-derived name that is stable across library
# re-tuning/regeneration (unlike the positional SolutionIndex, which shifts). If the
# buggy kernel is genuinely fixed or removed, its name stops matching and the entry
# should be pruned; `run_tensile_logic_check.py` reports such stale entries.
#
# ROCM-7144 - gfx950 logic: legacy MI / validation drift; tracked for cleanup.
# You may use YAML comments (#) and/or the optional "ticket" field for Jira keys.
"""


def _find_hipblaslt_root() -> Path:
    root = Path(__file__).resolve().parent.parent
    if not (root / "tensilelite").is_dir() or not (root / "library").is_dir():
        raise SystemExit(
            "Error: Cannot find hipblaslt root (expected tensilelite/ and library/). "
            "Run from hipblaslt root or keep scripts/ in the project tree."
        )
    return root


def _load_solution_names(logic_file: Path) -> Tuple[Dict[int, str], Set[str]]:
    """Return (solution_index -> SolutionNameMin, {SolutionNameMin}) for a logic file.

    Solution index follows the same rule as Run.py: the solution's "SolutionIndex"
    field when present, else its position in the solutions list.
    """
    with open(logic_file, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list) or len(data) < 6:
        raise ValueError(f"Unexpected logic file layout (no solutions list): {logic_file}")
    solutions = data[5]
    if not isinstance(solutions, list):
        raise ValueError(f"Solutions (index 5) is not a list: {logic_file}")

    idx_to_name: Dict[int, str] = {}
    names: Set[str] = set()
    for pos, sol in enumerate(solutions):
        if not isinstance(sol, dict):
            continue
        name = sol.get("SolutionNameMin")
        if not name:
            continue
        idx = int(sol.get("SolutionIndex", pos))
        idx_to_name[idx] = name
        names.add(name)
    return idx_to_name, names


class Resolution:
    def __init__(self, path: str, solution_name: Optional[str], ticket: Optional[str], status: str, detail: str = ""):
        self.path = path
        self.solution_name = solution_name
        self.ticket = ticket
        self.status = status  # "resolved" | "migrated" | "stale"
        self.detail = detail


def _resolve_entries(kb_path: Path, logic_root: Path) -> List[Resolution]:
    with open(kb_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return []
    if not isinstance(raw, dict):
        raise SystemExit(f"Error: known-bugs file must be a mapping at top level: {kb_path}")
    skips = raw.get("skips") or []
    if not isinstance(skips, list):
        raise SystemExit(f"Error: known-bugs 'skips' must be a list: {kb_path}")

    cache: Dict[str, Tuple[Dict[int, str], Set[str]]] = {}
    results: List[Resolution] = []
    for i, entry in enumerate(skips):
        if not isinstance(entry, dict):
            raise SystemExit(f"Error: skips[{i}] must be a mapping: {kb_path}")
        path = entry.get("path")
        if not path or not isinstance(path, str):
            raise SystemExit(f"Error: skips[{i}] requires string 'path': {kb_path}")
        ticket = entry.get("ticket")
        sol_name = entry.get("solution_name")
        sol_index = entry.get("solution_index")

        logic_file = logic_root / path
        if not logic_file.is_file():
            results.append(Resolution(path, sol_name, ticket, "stale", "logic file not found"))
            continue

        if path not in cache:
            cache[path] = _load_solution_names(logic_file)
        idx_to_name, names = cache[path]

        if sol_name:
            if sol_name in names:
                results.append(Resolution(path, sol_name, ticket, "resolved"))
            else:
                results.append(Resolution(path, sol_name, ticket, "stale", "solution_name not found"))
        elif sol_index is not None:
            name = idx_to_name.get(int(sol_index))
            if name:
                results.append(Resolution(path, name, ticket, "migrated", f"index {sol_index}"))
            else:
                results.append(Resolution(path, None, ticket, "stale", f"solution_index {sol_index} not found"))
        else:
            raise SystemExit(
                f"Error: skips[{i}] requires 'solution_name' or 'solution_index': {kb_path}"
            )
    return results


def _render(results: List[Resolution]) -> str:
    lines = [HEADER, "version: 1", "skips:"]
    for r in results:
        if not r.solution_name:
            continue
        lines.append(f"  - path: {r.path}")
        lines.append(f"    solution_name: {r.solution_name}")
        if r.ticket:
            lines.append(f"    ticket: {r.ticket}")
    return "\n".join(lines) + "\n"


def main() -> None:
    root = _find_hipblaslt_root()
    default_kb = root / "tensilelite" / "Tensile" / "TensileLogic" / "known_bugs.yaml"
    default_logic = root / "library"

    parser = argparse.ArgumentParser(
        description="Resolve/verify the TensileLogic known-bugs list against the library logic."
    )
    parser.add_argument("--known-bugs", type=Path, default=default_kb, help=f"default: {default_kb}")
    parser.add_argument("--logic-path", type=Path, default=default_logic, help=f"default: {default_logic}")
    out = parser.add_mutually_exclusive_group()
    out.add_argument("--write", action="store_true", help="rewrite the known-bugs file in place (name-based)")
    out.add_argument("--stdout", action="store_true", help="print the migrated name-based YAML to stdout")
    args = parser.parse_args()

    if not args.known_bugs.is_file():
        raise SystemExit(f"Error: known-bugs file not found: {args.known_bugs}")
    if not args.logic_path.is_dir():
        raise SystemExit(f"Error: logic path not found (build the library first?): {args.logic_path}")

    results = _resolve_entries(args.known_bugs, args.logic_path)
    stale = [r for r in results if r.status == "stale"]

    for r in results:
        if r.status == "stale":
            print(f"STALE   {r.path} :: {r.solution_name or '?'} ({r.detail})")
        elif r.status == "migrated":
            print(f"MIGRATE {r.path} :: {r.detail} -> {r.solution_name}")
        else:
            print(f"OK      {r.path} :: {r.solution_name}")

    if args.stdout:
        sys.stdout.write("\n" + _render(results))

    if args.write:
        if stale:
            raise SystemExit(
                f"Error: {len(stale)} stale entr{'y' if len(stale) == 1 else 'ies'} cannot be "
                "resolved; fix or remove them before --write."
            )
        args.known_bugs.write_text(_render(results), encoding="utf-8")
        print(f"\nWrote {args.known_bugs}")

    if stale:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
