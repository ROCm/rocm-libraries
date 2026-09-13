#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run clang-tidy over only the translation units affected by changed files.

Why this exists: ``-DENABLE_CLANG_TIDY=ON`` runs clang-tidy inline with every
compile, which roughly quadruples a clean Windows build and re-runs on every
recompile. clang-tidy does not need any of that -- it only needs a
``compile_commands.json`` describing how each translation unit is compiled. So
this script drives ``run-clang-tidy`` after the fact, against a build that was
compiled with clang-tidy disabled, and narrows the work to what actually
changed.

Two things make "what changed" non-obvious:

1. **Headers are not translation units.** They never appear in the compile
   database, so a changed header cannot be handed to clang-tidy directly.
   Diagnostics inside a header only surface when some ``.cpp`` that includes it
   is analysed (``HeaderFilterRegex`` in .clang-tidy controls which of those get
   reported). This script therefore walks the include graph *backwards* from
   each changed header to every translation unit that reaches it, transitively.

2. **Selecting files by path regex is lossy.** ``run-clang-tidy`` takes
   positional regexes matched against database paths, which over- and
   under-matches in equal measure. Instead this script writes a filtered
   ``compile_commands.json`` containing exactly the chosen entries and points
   ``run-clang-tidy`` at that, so selection is exact.

The include scan is static (it reads ``#include`` lines) rather than derived
from build dependency files, so it works against a tree that has only been
configured and never compiled.
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path


# Extensions that can appear as a translation unit in the compile database.
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx"}
# Extensions that can only be reached through an includer.
HEADER_SUFFIXES = {".h", ".hh", ".hpp", ".hxx", ".inl"}
SCANNED_SUFFIXES = SOURCE_SUFFIXES | HEADER_SUFFIXES

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.MULTILINE)

# Mirror of the WIN32 block in projects/hipdnn/cmake/ClangTidy.cmake. Both checks
# fire on Microsoft STL internals rather than on hipDNN code and are clean
# against libstdc++, so Linux keeps the full rule set. Keep the two in sync.
WINDOWS_DISABLED_CHECKS = (
    "-bugprone-exception-escape",
    "-performance-noexcept-move-constructor",
)

# Vendored dependencies are excluded from analysis, matching the -source-filter
# used by the in-tree tidy targets.
EXCLUDED_PATH_PARTS = ("_deps",)


def run_git(repo_root, *args):
    """Return stdout of a git command, or None when the command fails."""
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def changed_paths(repo_root, base, include_untracked):
    """Collect files changed versus `base`, plus working-tree modifications.

    Uses the merge base so that commits landing on the base branch after this
    branch was cut are not reported as local changes.
    """
    paths = set()

    merge_base = None
    if base:
        out = run_git(repo_root, "merge-base", base, "HEAD")
        if out is None:
            print(
                f"tidy-changed: warning: cannot resolve merge base with '{base}'; "
                "falling back to working-tree changes only",
                file=sys.stderr,
            )
        else:
            merge_base = out.strip()

    if merge_base:
        out = run_git(
            repo_root, "diff", "--name-only", "--diff-filter=ACMR", merge_base, "HEAD"
        )
        paths.update((out or "").split())

    # Staged and unstaged edits that are not committed yet.
    out = run_git(repo_root, "diff", "--name-only", "--diff-filter=ACMR", "HEAD")
    paths.update((out or "").split())

    if include_untracked:
        out = run_git(repo_root, "ls-files", "--others", "--exclude-standard")
        paths.update((out or "").split())

    return {
        (repo_root / p).resolve() for p in paths if Path(p).suffix in SCANNED_SUFFIXES
    }


def build_include_index(scan_roots):
    """Map each scanned file to the set of files it includes.

    Include directives are resolved by suffix match against known files:
    ``#include <hipdnn_data_sdk/utilities/LineStore.hpp>`` matches any indexed
    path ending in that sequence of components. This avoids replicating the
    compiler's search path while still distinguishing same-named headers in
    different directories.
    """
    files = []
    for root in scan_roots:
        for path in root.rglob("*"):
            if path.suffix in SCANNED_SUFFIXES and path.is_file():
                if any(part in EXCLUDED_PATH_PARTS for part in path.parts):
                    continue
                files.append(path.resolve())

    # Index by trailing path components so an include can be resolved by suffix.
    by_suffix = {}
    for path in files:
        parts = path.parts
        for depth in range(1, min(len(parts), 6) + 1):
            key = "/".join(parts[-depth:])
            by_suffix.setdefault(key, set()).add(path)

    includes = {}
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        targets = set()
        for raw in INCLUDE_RE.findall(text):
            key = raw.replace("\\", "/").strip()
            targets.update(by_suffix.get(key, ()))
        includes[path] = targets
    return includes


def includers_of(seeds, includes):
    """Return `seeds` plus every file that transitively includes any of them."""
    reverse = {}
    for includer, targets in includes.items():
        for target in targets:
            reverse.setdefault(target, set()).add(includer)

    seen = set(seeds)
    queue = deque(seeds)
    while queue:
        current = queue.popleft()
        for parent in reverse.get(current, ()):
            if parent not in seen:
                seen.add(parent)
                queue.append(parent)
    return seen


def load_compile_db(build_dir):
    db_path = build_dir / "compile_commands.json"
    if not db_path.is_file():
        sys.exit(
            f"tidy-changed: no compile_commands.json in {build_dir}.\n"
            "Configure the build first (clang-tidy needs the compile database, "
            "not the compiled objects)."
        )
    with db_path.open(encoding="utf-8") as handle:
        entries = json.load(handle)
    keep = []
    for entry in entries:
        path = Path(entry["file"])
        if path.suffix not in SOURCE_SUFFIXES:
            continue
        if any(part in EXCLUDED_PATH_PARTS for part in path.parts):
            continue
        keep.append(entry)
    return keep


def find_source_dir(by_file):
    """Locate the hipDNN project directory from compile database source paths.

    Walks up from a source file to the highest ancestor that both carries a
    ``.clang-tidy`` and is named ``hipdnn``. The name check matters because the
    project nests per-directory ``.clang-tidy`` overrides under test folders,
    and those are not the root configuration.
    """
    for source in by_file:
        for ancestor in source.parents:
            if ancestor.name == "hipdnn" and (ancestor / ".clang-tidy").is_file():
                return ancestor
    # Fall back to any ancestor carrying a config, for non-standard layouts.
    for source in by_file:
        for ancestor in source.parents:
            if (ancestor / ".clang-tidy").is_file():
                return ancestor
    return None


def read_cmake_cache(build_dir, key):
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        return None
    pattern = re.compile(rf"^{re.escape(key)}:[A-Z]+=(.*)$", re.MULTILINE)
    match = pattern.search(cache.read_text(encoding="utf-8", errors="replace"))
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def resolve_tool(explicit, build_dir, cache_key, names, sibling_of=None):
    """Resolve a tool from an explicit path, the CMake cache, a sibling dir, or PATH.

    The CMake cache is preferred over PATH because the configure step already
    version-checked the binary it recorded.
    """
    if explicit:
        return str(Path(explicit).resolve())
    cached = read_cmake_cache(build_dir, cache_key)
    if cached and Path(cached).exists():
        return cached
    if sibling_of:
        for name in names:
            candidate = Path(sibling_of).parent / name
            if candidate.exists():
                return str(candidate)
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Run clang-tidy over translation units affected by changed files."
    )
    parser.add_argument(
        "--build-dir",
        required=True,
        help="Build directory containing compile_commands.json",
    )
    parser.add_argument(
        "--base",
        default="origin/develop",
        help="Branch or ref to diff against (default: origin/develop)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Explicit files to check instead of deriving them from git",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check every translation unit in the compile database",
    )
    parser.add_argument(
        "--no-untracked",
        action="store_true",
        help="Ignore untracked files when deriving changed files",
    )
    parser.add_argument("--clang-tidy", help="Path to clang-tidy")
    parser.add_argument("--run-clang-tidy", help="Path to the run-clang-tidy script")
    parser.add_argument(
        "--source-dir",
        help="hipDNN project directory holding .clang-tidy "
        "(default: derived from the compile database)",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Parallel clang-tidy jobs (default: processor count)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the selected translation units without running clang-tidy",
    )
    args = parser.parse_args()

    build_dir = Path(args.build_dir).resolve()

    entries = load_compile_db(build_dir)
    by_file = {Path(e["file"]).resolve(): e for e in entries}

    # Derived from the compile database rather than from __file__: skills are
    # installed outside the repository, so the script's own location says
    # nothing about which checkout the build came from.
    if args.source_dir:
        hipdnn_dir = Path(args.source_dir).resolve()
    else:
        hipdnn_dir = find_source_dir(by_file)
        if hipdnn_dir is None:
            sys.exit(
                "tidy-changed: cannot locate the hipDNN source directory from the "
                "compile database; pass --source-dir"
            )
    config_file = hipdnn_dir / ".clang-tidy"
    if not config_file.is_file():
        sys.exit(f"tidy-changed: no .clang-tidy at {config_file}")

    repo_root_out = run_git(hipdnn_dir, "rev-parse", "--show-toplevel")
    if repo_root_out is None:
        sys.exit("tidy-changed: not inside a git repository")
    repo_root = Path(repo_root_out.strip()).resolve()

    if args.all:
        selected = list(by_file.values())
        print(f"tidy-changed: checking all {len(selected)} translation units")
    else:
        if args.files:
            seeds = {Path(f).resolve() for f in args.files}
        else:
            seeds = changed_paths(repo_root, args.base, not args.no_untracked)
        seeds = {p for p in seeds if p.is_file()}
        if not seeds:
            print("tidy-changed: no changed C/C++ files; nothing to check")
            return 0

        print(f"tidy-changed: {len(seeds)} changed file(s):")
        for path in sorted(seeds):
            print(f"  {path.relative_to(repo_root)}")

        # Only headers need the include-graph walk; sources are units themselves.
        headers = {p for p in seeds if p.suffix in HEADER_SUFFIXES}
        affected = set(seeds)
        if headers:
            includes = build_include_index([hipdnn_dir])
            affected |= includers_of(headers, includes)

        selected = [by_file[p] for p in sorted(affected) if p in by_file]
        if not selected:
            print(
                "tidy-changed: no translation unit in the compile database is "
                "affected by those files; nothing to check"
            )
            return 0
        print(
            f"tidy-changed: {len(selected)} affected translation unit(s) "
            f"out of {len(by_file)} in the compile database"
        )

    if args.dry_run:
        for entry in selected:
            print(f"  {Path(entry['file']).relative_to(repo_root)}")
        return 0

    clang_tidy = resolve_tool(
        args.clang_tidy, build_dir, "CLANG_TIDY_EXE", ("clang-tidy-20", "clang-tidy")
    )
    if not clang_tidy:
        sys.exit("tidy-changed: clang-tidy not found; pass --clang-tidy")
    runner = resolve_tool(
        args.run_clang_tidy,
        build_dir,
        "RUN_CLANG_TIDY_EXE",
        ("run-clang-tidy-20", "run-clang-tidy"),
        sibling_of=clang_tidy,
    )
    if not runner:
        sys.exit("tidy-changed: run-clang-tidy not found; pass --run-clang-tidy")

    # Exact selection beats a path regex: hand run-clang-tidy a database that
    # contains only the chosen entries.
    with tempfile.TemporaryDirectory(prefix="hipdnn-tidy-") as tmp:
        subset = Path(tmp) / "compile_commands.json"
        subset.write_text(json.dumps(selected, indent=1), encoding="utf-8")

        command = []
        if platform.system() == "Windows":
            # run-clang-tidy is an extensionless Python script; its shebang means
            # nothing to the Windows loader, so it needs an explicit interpreter.
            command.append(sys.executable)
        command += [
            runner,
            "-p",
            str(subset.parent),
            "-clang-tidy-binary",
            clang_tidy,
            f"-config-file={config_file}",
            "-quiet",
            "-j",
            str(args.jobs),
        ]
        if platform.system() == "Windows":
            # Single dash: clang-tidy accepts either spelling, but run-clang-tidy
            # parses its own -checks with argparse and rejects --checks.
            command.append("-checks=" + ",".join(WINDOWS_DISABLED_CHECKS))

        print(f"tidy-changed: {clang_tidy}")
        result = subprocess.run(command, cwd=hipdnn_dir, check=False)

    if result.returncode == 0:
        print("tidy-changed: clean")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
