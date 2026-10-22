#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# advisory_checks.py -- NON-BLOCKING static analysis for ALL rocke Python
# (library, platform, tools, cmake):
#   pyright  -- type checking
#   radon    -- software metrics (cyclomatic complexity, maintainability index,
#               Halstead, raw line counts)
#   vulture  -- dead code (unused functions/variables, unreachable branches)
#
# ALWAYS exits 0: findings are reported, never enforced -- even if a tool is
# missing or crashes. Wired as pre-push hooks (rocke-pyright / rocke-radon /
# rocke-vulture) in the repo-root .pre-commit-config.yaml, which installs each
# tool into its own env. pyright settings live in rocke/pyrightconfig.json; the
# radon and vulture settings are below.
#
# Paths outside rocke are ignored; no paths = all of rocke.
#
# What the reports mean and what to act on: rocke/docs/static-analysis.md.
#
# Usage (pre-commit manages the tool installs):
#   pre-commit install --hook-type pre-push          # one-time: run on `git push`
#   pre-commit run rocke-radon --hook-stage pre-push --files <paths>
#   pre-commit run rocke-vulture --hook-stage pre-push --all-files
# Direct (the tool must be importable by this python):
#   python tools/advisory_checks.py radon library/kernels/gfx942/attention_tiled_2d.py
#   python tools/advisory_checks.py vulture          # all of rocke

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROCKE = Path(__file__).resolve().parents[1]  # tools -> rocke
GUIDE = ROCKE / "docs" / "static-analysis.md"
# Python trees under rocke; anything else (docs, style, ...) holds no code.
TREES = ("library", "platform", "tools", "cmake")
# Untracked artifact dirs, mirroring rocke/.gitignore.
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache", "_deps", "kernel_cache"}

TOOLS = ("pyright", "radon", "vulture")

# radon: list functions/methods at or above this cyclomatic complexity
# (11 = rank C, "moderately complex"), worst first, at most CC_TOP_N of them.
CC_REPORT_MIN = 11
CC_TOP_N = 15

# vulture: 60 = also report unused functions/methods/classes (100 = only
# unreachable code and unused imports/arguments).
VULTURE_MIN_CONFIDENCE = 60


def _show(p: Path) -> str:
    """Path relative to the invoking cwd, so editors/terminals can open it."""
    return os.path.relpath(p, Path.cwd())


def _skipped(p: Path) -> bool:
    # Directory parts only: a FILE called venv_probe.py is code, not a virtualenv.
    return any(
        part in SKIP_DIRS
        or part.endswith(".egg-info")
        or part.startswith((".venv", "venv"))
        for part in p.relative_to(ROCKE).parts[:-1]
    )


def _all_files() -> list[Path]:
    # is_file() drops dangling symlinks such as Emacs's `.#<name>.py` lock files:
    # vulture exits on a path that does not exist, and radon cannot read one.
    return sorted(
        p
        for tree in TREES
        for p in (ROCKE / tree).rglob("*.py")
        if p.is_file() and not _skipped(p)
    )


def _rocke_files(paths: list[str]) -> list[Path]:
    if not paths:
        return _all_files()
    files = []
    for raw in paths:
        p = Path(raw).resolve()
        if (
            p.suffix == ".py"
            and p.is_file()
            and p.is_relative_to(ROCKE)
            and not _skipped(p)
        ):
            files.append(p)
    return files


def run_pyright(files: list[Path], whole: bool) -> None:
    cmd = [sys.executable, "-m", "pyright", "--project", str(ROCKE), "--outputjson"]
    if not whole:
        # rocke-relative: shorter than the repo-relative names pre-commit sized
        # this batch for. Absolute names overflow Windows' 32,767-character
        # command line on a full batch (e.g. --all-files).
        cmd += [str(f.relative_to(ROCKE)) for f in files]
    proc = subprocess.run(cmd, cwd=ROCKE, capture_output=True, text=True)
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError:  # pyright failed before analysing; show why
        print(proc.stdout + proc.stderr, end="")
        return
    # One line per diagnostic (pyright's text mode spends 3 on each), then a
    # per-rule tally so recurring patterns stand out from one-offs.
    by_rule: Counter[str] = Counter()
    for d in report["generalDiagnostics"]:
        rule = d.get("rule", d["severity"])
        by_rule[rule] += 1
        line = d["range"]["start"]["line"] + 1
        msg = d["message"].splitlines()[0]
        print(f"{_show(Path(d['file']))}:{line}: {d['severity']}: {msg} ({rule})")
    s = report["summary"]
    print(
        f"pyright: {s['errorCount']} error(s), {s['warningCount']} warning(s)"
        + "".join(f"\n  {n:5d}  {r}" for r, n in by_rule.most_common())
    )


def run_vulture(files: list[Path], whole: bool) -> None:
    from vulture import Vulture

    # Always scan ALL of rocke in one pass, so a platform method called only
    # from library code (or a helper used only by tests) is not reported as
    # dead; then report just the requested files. Tests count as uses but are
    # not reported on: pytest fixtures and test_* functions look unused by design.
    scanned = _all_files()
    # In-process, not `python -m vulture <every file>`: that command line is
    # ~62,000 characters, past Windows' 32,767 limit, so the scan could never start.
    vulture = Vulture()
    vulture.scavenge(scanned)
    reported = {
        f
        for f in (scanned if whole else files)
        if "tests" not in f.relative_to(ROCKE).parts
    }
    shown = 0
    for item in vulture.get_unused_code(min_confidence=VULTURE_MIN_CONFIDENCE):
        if item.filename in reported:
            print(
                f"{_show(item.filename)}:{item.first_lineno}: {item.message}"
                f" ({item.confidence}% confidence)"
            )
            shown += 1
    print(f"vulture: {shown} finding(s)")


def _with_closures(fn: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Yield (qualified name, block) for ``fn`` and its nested functions.

    radon scores a nested function SEPARATELY -- its branches do not count
    toward the enclosing function. Kernel builders do most of their work in
    nested helpers, so they are listed too, as ``outer.inner``.
    """
    name = f"{prefix}.{fn.name}" if prefix else fn.fullname
    yield name, fn
    for inner in fn.closures:
        yield from _with_closures(inner, name)


def _print_summary(
    f: Path, raw: Any, mi: float, hal: Any, funcs: list[tuple[str, Any]]
) -> None:
    from radon.complexity import cc_rank
    from radon.metrics import mi_rank

    avg_cc = sum(fn.complexity for _, fn in funcs) / len(funcs) if funcs else 0.0
    comments = raw.comments + raw.multi
    print(
        f"{_show(f)}\n"
        f"    SLOC {raw.sloc}  comment lines {comments}"
        f" ({100 * comments / max(raw.loc, 1):.0f}% of LOC)  functions {len(funcs)}\n"
        f"    MI {mi:.1f} ({mi_rank(mi)})  avg CC {avg_cc:.1f} ({cc_rank(avg_cc)})"
        f"  Halstead V {hal.volume:.0f}  D {hal.difficulty:.1f}"
        f"  bugs {hal.bugs:.1f}"
    )


def _print_hot(hot: list[tuple[str, Any, Path]]) -> None:
    """The CC_TOP_N most complex functions across every reported file."""
    from radon.complexity import cc_rank

    hot.sort(key=lambda t: t[1].complexity, reverse=True)
    print(f"\nMost complex functions (CC >= {CC_REPORT_MIN}, top {CC_TOP_N}):")
    for name, fn, f in hot[:CC_TOP_N]:
        # Span includes nested functions; the CC score does not.
        print(
            f"  {cc_rank(fn.complexity)} {fn.complexity:4d}  "
            f"{_show(f)}:{fn.lineno}  {name}"
            f"  ({fn.endline - fn.lineno + 1} lines)"
        )
    if len(hot) > CC_TOP_N:
        print(f"  ... and {len(hot) - CC_TOP_N} more")


def run_radon(files: list[Path], whole: bool) -> None:
    from radon.complexity import cc_visit
    from radon.metrics import h_visit, mi_visit
    from radon.raw import analyze
    from radon.visitors import Function

    print(
        "Ranks -- CC: A 1-5, B 6-10, C 11-20, D 21-30, E 31-40, F 41+ (per function)."
        "\n         MI: A 20-100, B 10-19, C 0-9 (per file; size-penalised, so very"
        "\n             large files score 0 regardless of quality)."
        "\n  Halstead: V = volume (information content), D = difficulty (error"
        "\n            proneness), bugs = V/3000 delivered-bug estimate.\n"
    )
    hot: list[tuple[str, Any, Path]] = []
    for f in files:
        code = f.read_text(encoding="utf-8")
        try:
            raw = analyze(code)
            mi = mi_visit(code, multi=True)
            hal = h_visit(code).total
            # cc_visit lists functions and methods flat, plus a summary block
            # per class; keep the functions, each with its nested functions.
            funcs = [
                named
                for blk in cc_visit(code)
                if isinstance(blk, Function)
                for named in _with_closures(blk)
            ]
        except SyntaxError as e:
            print(f"{_show(f)}: radon could not parse ({e})")
            continue
        _print_summary(f, raw, mi, hal, funcs)
        hot += [(name, fn, f) for name, fn in funcs if fn.complexity >= CC_REPORT_MIN]
    if hot:
        _print_hot(hot)


RUNNERS = {"pyright": run_pyright, "radon": run_radon, "vulture": run_vulture}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="NON-BLOCKING static analysis for all rocke Python (always exits 0)"
    )
    ap.add_argument("tool", choices=TOOLS)
    ap.add_argument(
        "paths", nargs="*", help="files to report on (default: all of rocke)"
    )
    args = ap.parse_args()

    whole = not args.paths
    files = _rocke_files(args.paths)
    if not files:
        return 0
    print(
        f"[advisory, non-blocking] {args.tool} on {len(files)} file(s)"
        f" -- how to read this report: {_show(GUIDE)}",
        flush=True,
    )
    try:
        RUNNERS[args.tool](files, whole)
    except ImportError as e:
        print(f"{args.tool}: not installed for {sys.executable} ({e}); skipped")
    except (Exception, SystemExit) as e:  # advisory: never block a push
        # SystemExit too: vulture's scavenge() calls sys.exit() for a vanished file.
        print(f"{args.tool}: crashed ({type(e).__name__}: {e}); skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
