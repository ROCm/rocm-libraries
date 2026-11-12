#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# static_analysis_checks.py -- NON-BLOCKING static analysis for ALL rocke Python:
#   pyright  -- type checking
#   radon    -- software metrics (cyclomatic complexity, maintainability index,
#               Halstead, raw line counts)
#   vulture  -- dead code (unused functions/variables, unreachable branches)
#
# ALWAYS exits 0: findings are reported, never enforced -- even if a tool is
# missing or crashes. Wired as pre-push hooks (rocke-pyright / rocke-radon /
# rocke-vulture) in the repo-root .pre-commit-config.yaml, using the active Python
# environment without installing tools. pyright settings live next to this file, in
# pyrightconfig.json; the radon and vulture settings are below. Pinned versions of
# the tools: requirements.txt, also next to this file.
#
# Paths outside rocke are ignored; no paths = every tracked rocke .py file, as
# listed by `git ls-files` (git, not this script, knows what is source and what
# is an artifact).
#
# What the reports mean and what to act on: rocke/docs/static-analysis.md.
#
# Usage (install optional tools separately; see the guide):
#   pre-commit install --hook-type pre-push --install-hooks   # one-time; runs on push
#   pre-commit run rocke-radon --hook-stage pre-push --files <paths>
# Direct (the tool must be importable by this python). Use this form, not
# `pre-commit run --all-files`, for all of rocke: pre-commit splits a long file
# list into several runs, each printing its own report.
#   python tools/static_analysis/static_analysis_checks.py radon library/kernels/gfx942/attention_tiled_2d.py
#   python tools/static_analysis/static_analysis_checks.py vulture   # all of rocke, one report

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

ROCKE = Path(__file__).resolve().parents[2]  # static_analysis -> tools -> rocke
GUIDE = ROCKE / "docs" / "static-analysis.md"
# Relative to ROCKE, so a test that repoints ROCKE at a scratch tree finds its own copy.
PYRIGHT_CONFIG = Path("tools") / "static_analysis" / "pyrightconfig.json"

# Variables that tell git which repository to use. Git exports GIT_DIR to a hook
# that runs from a linked worktree; with it set and no GIT_WORK_TREE, git takes
# the cwd as the top of the work tree, so `git ls-files` run from rocke/ lists the
# whole repository with repo-root-relative paths. Dropping them makes git find the
# repository from the cwd, as it does in any other shell.
_GIT_REPO_ENV = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")

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


def _all_files() -> list[Path]:
    """Every tracked rocke ``.py`` file that exists on disk.

    This is the one list of what counts as rocke source: git already knows which
    files are source and which are artifacts (venvs, build trees, caches are
    ignored or untracked), so neither this script nor pyrightconfig.json keeps a
    list of directories to cover or to skip.
    """
    env = {k: v for k, v in os.environ.items() if k not in _GIT_REPO_ENV}
    proc = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.py"], cwd=ROCKE, env=env, capture_output=True
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode(errors="replace").strip() or "no error output"
        raise RuntimeError(
            f"git ls-files failed (exit {proc.returncode}) in {ROCKE}: {detail}"
        )
    names = (os.fsdecode(name) for name in proc.stdout.split(b"\0") if name)
    # is_file() drops what git lists but the disk lacks (a dangling symlink, a file
    # deleted but not yet staged, a path outside a sparse checkout): vulture exits
    # on a path that does not exist, and radon cannot read one.
    # A set: `git ls-files` lists a path once per index stage, so a file with a
    # merge conflict comes back up to three times.
    return sorted({p for name in names if (p := ROCKE / name).is_file()})


def _rocke_files(paths: list[str]) -> list[Path]:
    if not paths:
        return _all_files()
    files = []
    for raw in paths:
        p = Path(raw).resolve()
        if p.suffix == ".py" and p.is_file() and p.is_relative_to(ROCKE):
            files.append(p)
    return files


def run_pyright(files: list[Path]) -> None:
    # The file names go on stdin, not the command line: all of rocke is ~1,900
    # names (~98,000 characters), past Windows' 32,767-character command-line
    # limit. They are rocke-relative (the cwd), one per line. A list on stdin
    # replaces the config's `include`, so pyrightconfig.json holds none; pyright
    # splits the list on any whitespace, so a file name with a space would break it,
    # and its built-in excludes (hidden directories, node_modules) still apply.
    names = "\n".join(str(f.relative_to(ROCKE)) for f in files)
    cmd = [
        sys.executable,
        "-m",
        "pyright",
        "--project",
        str(ROCKE / PYRIGHT_CONFIG),
        "--outputjson",
        "-",
    ]
    # rocke/ is an import root (`library.kernels...`). While the config sat in rocke/
    # pyright had it on its import path without being told; with the config in
    # tools/static_analysis/ it does not, and the same code is then analysed with
    # different module names (~150 of ~30,000 reports change today). PYTHONPATH
    # is how a Python tool is told, and it reaches the interpreter pyright asks for
    # its search paths.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROCKE), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
    )
    # UTF-8 both ways: Node reads the names and writes the JSON as UTF-8, while
    # text=True would use the locale's code page (cp1252 on Windows).
    proc = subprocess.run(
        cmd,
        cwd=ROCKE,
        env=env,
        input=names,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
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
        location = _show(Path(d["file"]))
        if d.get("range") is not None:
            location += f":{d['range']['start']['line'] + 1}"
        msg = d["message"].splitlines()[0]
        print(f"{location}: {d['severity']}: {msg} ({rule})")
    s = report["summary"]
    print(
        f"pyright: {s['errorCount']} error(s), {s['warningCount']} warning(s)"
        + "".join(f"\n  {n:5d}  {r}" for r, n in by_rule.most_common())
    )


def run_vulture(files: list[Path]) -> None:
    from vulture import Vulture

    # Always scan ALL of rocke in one pass, so a platform method called only
    # from library code (or a helper used only by tests) is not reported as
    # dead; then report just the requested files. Tests count as uses but are
    # not reported on: pytest fixtures and test_* functions look unused by design.
    # A requested file git does not track yet (a new file) is scanned too, or its
    # report would be empty without saying so.
    scanned = sorted({*_all_files(), *files})
    # In-process, not `python -m vulture <every file>`: that command line is
    # ~98,000 characters even with rocke-relative names (1,887 files), past
    # Windows' 32,767 limit, so the scan could never start.
    vulture = Vulture()
    vulture.scavenge(scanned)
    reported = {f for f in files if "tests" not in f.relative_to(ROCKE).parts}
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


def run_radon(files: list[Path]) -> None:
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


def _tolerate_unencodable_output() -> None:
    """Keep print() from raising on a character the output encoding lacks.

    pre-commit pipes this script's output, so on Windows it is encoded strictly
    with the ANSI code page. A message that names a path (git's "dubious
    ownership" refusal, an OSError) can hold a character outside it, and the
    UnicodeEncodeError would escape the handler in main() and fail the push.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:  # absent when the stream is not a TextIOWrapper
            reconfigure(errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="NON-BLOCKING static analysis for all rocke Python (always exits 0)"
    )
    ap.add_argument("tool", choices=TOOLS)
    ap.add_argument(
        "paths", nargs="*", help="files to report on (default: all of rocke)"
    )
    args = ap.parse_args()
    _tolerate_unencodable_output()

    try:
        # Inside the guard: listing all of rocke runs git, which can fail too.
        files = _rocke_files(args.paths)
        if not files:
            return 0
        print(
            f"[advisory, non-blocking] {args.tool} on {len(files)} file(s)"
            f" -- how to read this report: {_show(GUIDE)}",
            flush=True,
        )
        RUNNERS[args.tool](files)
    except ImportError as e:
        print(f"{args.tool}: not installed for {sys.executable} ({e}); skipped")
    except (Exception, SystemExit) as e:  # advisory: never block a push
        # SystemExit too: vulture's scavenge() calls sys.exit() for a vanished file.
        print(f"{args.tool}: crashed ({type(e).__name__}: {e}); skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
