#!/usr/bin/env python3
"""Splice arch cases from one rocRAND config header (or dir) into another.

select_best_config.py fully regenerates a header, so it only contains the
arches present in the benchmark JSON you pass. This tool merges the
`case target_arch::<arch>: return <val>;` lines from a freshly generated
header into an existing header, preserving every other arch.

For each generator file, both switch blocks are handled independently:
  get_threads -> block_size
  get_blocks  -> grid_size
An arch present in --new overwrites the same arch in --base; new arches are
inserted just before the `default:` line.
"""

import argparse
import os
import re
import sys

CASE_RE = re.compile(r"^(\s*)case\s+target_arch::(\w+)\s*:\s*return\s+([^;]+);")
DEFAULT_RE = re.compile(r"^(\s*)default\s*:")


def _func_of(line: str, current: str | None) -> str | None:
    if "get_threads" in line:
        return "threads"
    if "get_blocks" in line:
        return "blocks"
    return current


def parse_cases(text: str) -> dict[str, dict[str, str]]:
    """Extract {'threads': {arch: val}, 'blocks': {arch: val}} from a header."""
    result: dict[str, dict[str, str]] = {"threads": {}, "blocks": {}}
    func: str | None = None
    for line in text.splitlines():
        func = _func_of(line, func)
        m = CASE_RE.match(line)
        if m and func:
            result[func][m.group(2)] = m.group(3).strip()
    return result


def splice(base_text: str, new_map: dict[str, dict[str, str]]) -> str:
    """Return base_text with arch cases from new_map merged in."""
    out: list[str] = []
    func: str | None = None
    handled: set[str] = set()
    for line in base_text.splitlines():
        prev_func = func
        func = _func_of(line, func)
        if func != prev_func:
            handled = set()

        m = CASE_RE.match(line)
        if m and func and m.group(2) in new_map[func]:
            arch = m.group(2)
            out.append(f"{m.group(1)}case target_arch::{arch}: return {new_map[func][arch]};")
            handled.add(arch)
            continue

        d = DEFAULT_RE.match(line)
        if d and func:
            indent = d.group(1)
            for arch, val in new_map[func].items():
                if arch not in handled:
                    out.append(f"{indent}case target_arch::{arch}: return {val};")
                    handled.add(arch)

        out.append(line)

    trailing = "\n" if base_text.endswith("\n") else ""
    return "\n".join(out) + trailing


def splice_file(base_path: str, new_path: str) -> tuple[str, bool]:
    with open(base_path) as f:
        base_text = f.read()
    with open(new_path) as f:
        new_text = f.read()
    new_map = parse_cases(new_text)
    if not new_map["threads"] and not new_map["blocks"]:
        return base_text, False
    merged = splice(base_text, new_map)
    return merged, merged != base_text


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--base", required=True, help="Existing header file or directory to merge INTO")
    cli.add_argument("--new", required=True, help="Freshly generated header file or directory to merge FROM")
    cli.add_argument("--out", help="Output file/dir (default: overwrite --base in place)")
    cli.add_argument("--dry-run", action="store_true", help="Print what would change, write nothing")
    args = cli.parse_args()

    base_is_dir = os.path.isdir(args.base)
    new_is_dir = os.path.isdir(args.new)
    if base_is_dir != new_is_dir:
        cli.error("--base and --new must both be files or both be directories")

    if base_is_dir:
        pairs = []
        for name in sorted(os.listdir(args.new)):
            if not name.endswith("_config.hpp"):
                continue
            base_file = os.path.join(args.base, name)
            if os.path.exists(base_file):
                pairs.append((name, base_file, os.path.join(args.new, name)))
            else:
                print(f"skip (no base): {name}")
        out_dir = args.out or args.base
    else:
        pairs = [(os.path.basename(args.base), args.base, args.new)]
        out_dir = None

    changed = 0
    for name, base_file, new_file in pairs:
        merged, did_change = splice_file(base_file, new_file)
        if not did_change:
            print(f"unchanged: {name}")
            continue
        changed += 1
        if base_is_dir:
            dest = os.path.join(out_dir, name)
        else:
            dest = args.out or args.base
        if args.dry_run:
            print(f"would update: {dest}")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
            with open(dest, "w") as f:
                f.write(merged)
            print(f"updated: {dest}")

    print(f"\n{changed} file(s) {'would be ' if args.dry_run else ''}changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
