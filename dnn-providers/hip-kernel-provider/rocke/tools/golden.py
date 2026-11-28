#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""List, re-record and check rocKE's goldens, from the golden index.

Run from the rocke directory::

    python tools/golden.py list [--json]
    python tools/golden.py rerecord kda_gfx942
    python tools/golden.py rerecord representative --only attention_dense,gemm
    python tools/golden.py rerecord --all          # every golden a CPU can write
    python tools/golden.py rerecord --conflicted   # after a merge or rebase
    python tools/golden.py check

``rerecord`` runs each golden's own record commands (``golden_index.py``) with
this interpreter. Review the change before re-recording: a re-record accepts any
output. ``--conflicted`` resolves every golden file git reports as conflicted:
two changes re-recorded the same family, so the file is re-recorded whole on the
merged tree (git's side of the file is taken only to clear the markers); a
golden only a GPU can re-record is left conflicted. Then ``git add`` the files.

``check`` fails on a golden file the index does not own (no test checks it), a
golden whose files or checking tests are all gone, and a one-family JSON golden
whose fingerprint is missing or stale.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

_HERE = Path(__file__).resolve().parent
ROCKE = _HERE.parent  # tools -> rocke (in a source tree)
# The index beside this file, and the rocke package for the golden-file rules.
for _module_root in (_HERE, ROCKE / "platform" / "python"):
    if _module_root.is_dir() and str(_module_root) not in sys.path:
        sys.path.insert(0, str(_module_root))

from golden_index import (
    DOCUMENT,
    GOLDEN_DIR_NAMES,
    GOLDENS,
    NOT_GOLDEN_SUFFIXES,
    SEARCH_ROOTS,
    Golden,
    golden_named,
)
from rocke.core.golden_file import rerecord_command, rerecord_hint, verify_fingerprint

__all__ = ["TreeLayout", "golden_problems", "main"]


@dataclass(frozen=True)
class TreeLayout:
    """Where each rocke tree that holds goldens is on disk.

    ``roots`` maps a rocke-relative tree (``platform/tests``) to its directory.
    A source checkout keeps the trees in place; the installed test payload
    stages them elsewhere (:meth:`installed`). A tree whose directory is absent
    (a platform-only copy has no ``library/``) is not checked.
    """

    roots: tuple[tuple[str, Path], ...]

    @classmethod
    def source(cls, rocke: Path) -> TreeLayout:
        """The layout of a source checkout rooted at ``rocke``."""
        return cls(tuple((tree, rocke / tree) for tree in SEARCH_ROOTS))

    @classmethod
    def installed(cls, test_root: Path, package: Path) -> TreeLayout:
        """The installed test payload: platform tests in ``<test_root>/tests``,
        library tests in ``<test_root>/tests/library/tests``, and the rocke
        package at ``package``."""
        return cls(
            (
                ("platform/tests", test_root / "tests"),
                ("platform/python/rocke", package),
                ("library/tests", test_root / "tests" / "library" / "tests"),
            )
        )

    def _tree_of(self, path: str) -> tuple[str, Path] | None:
        """The (tree, directory) holding the rocke-relative ``path``."""
        matches = [
            (tree, directory)
            for tree, directory in self.roots
            if path == tree or path.startswith(tree + "/")
        ]
        return max(matches, key=lambda match: len(match[0]), default=None)

    def present(self, path: str) -> bool:
        """Whether the tree holding ``path`` is in this layout."""
        owner = self._tree_of(path)
        return owner is not None and owner[1].is_dir()

    def locate(self, path: str) -> Path:
        """The directory entry for the rocke-relative ``path``."""
        owner = self._tree_of(path)
        if owner is None:
            raise ValueError(
                f"path={path!r} is in no tree of this layout, expected one under "
                f"{[tree for tree, _ in self.roots]}"
            )
        tree, directory = owner
        return directory / path[len(tree) :].lstrip("/")

    def glob(self, pattern: str) -> list[str]:
        """The rocke-relative files matching ``pattern`` that exist."""
        parent = PurePosixPath(pattern).parent.as_posix()
        if not self.present(pattern):
            return []
        found = self.locate(parent).glob(PurePosixPath(pattern).name)
        return sorted(f"{parent}/{entry.name}" for entry in found if entry.is_file())

    def golden_files(self) -> Iterator[str]:
        """Every rocke-relative file under a golden directory of every tree."""
        directories = {directory.resolve() for _, directory in self.roots}
        for tree, directory in self.roots:
            if not directory.is_dir():
                continue
            for current, subdirs, files in os.walk(directory):
                here = Path(current)
                # A nested tree (the installed library tests) is walked as its own.
                subdirs[:] = sorted(
                    name
                    for name in subdirs
                    if name != "__pycache__"
                    and (here / name).resolve() not in directories
                )
                if not set(here.relative_to(directory).parts) & set(GOLDEN_DIR_NAMES):
                    continue
                for name in sorted(files):
                    if name.endswith(NOT_GOLDEN_SUFFIXES):
                        continue
                    relative = (here / name).relative_to(directory).as_posix()
                    yield f"{tree}/{relative}"


def golden_problems(
    layout: TreeLayout, goldens: Iterable[Golden] = GOLDENS
) -> list[str]:
    """Everything wrong with the goldens on disk against the index. Empty == OK."""
    goldens = tuple(goldens)
    problems: list[str] = []
    for path in layout.golden_files():
        owners = [golden.name for golden in goldens if golden.owns(path)]
        if not owners:
            problems.append(
                f"{path}: no golden in tools/golden_index.py owns it, so no test "
                "checks it; add it to the index with its test, or delete it"
            )
        elif len(owners) > 1:
            problems.append(f"{path}: owned by several goldens {owners}")
    for golden in goldens:
        problems.extend(_golden_entry_problems(layout, golden))
    return problems


def _golden_entry_problems(layout: TreeLayout, golden: Golden) -> list[str]:
    """What is wrong with one index entry: its files, tests and fingerprints."""
    present = [pattern for pattern in golden.files if layout.present(pattern)]
    if not present:
        return []  # its tree is not in this layout
    problems = [
        f"golden {golden.name}: no file matches {pattern}"
        for pattern in present
        if not layout.glob(pattern)
    ]
    checkers = [test for test in golden.checked_by if layout.present(test)]
    if checkers and not any(layout.locate(test).is_file() for test in checkers):
        problems.append(
            f"golden {golden.name}: no test checks it - none of {checkers} exists"
        )
    if golden.fingerprint == DOCUMENT:
        unfit = [
            f"  {path}: {reason}"
            for pattern in present
            for path in layout.glob(pattern)
            for ok, reason in [_document_fingerprint(layout.locate(path))]
            if not ok
        ]
        if unfit:
            problems.append(
                f"golden {golden.name}: {len(unfit)} file(s) need a re-record:\n"
                + "\n".join(unfit)
                + "\n"
                + rerecord_hint(golden.name)
            )
    return problems


def _document_fingerprint(path: Path) -> tuple[bool, str]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return False, f"not a readable JSON golden ({error})"
    if not isinstance(document, dict):
        return False, "a golden document must be a JSON object"
    return verify_fingerprint(document)


# --- re-record --------------------------------------------------------------------


def _run_golden_record(golden: Golden, families: list[str] | None) -> int:
    """Run each record step of ``golden`` in the source tree; the first nonzero
    exit status, else 0."""
    for step in golden.record:
        cwd = ROCKE / step.cwd
        env = dict(os.environ)
        paths = [str((cwd / entry).resolve()) for entry in step.pythonpath]
        if env.get("PYTHONPATH"):
            paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(paths)
        arguments = step.arguments(families)
        print(f"[golden] {golden.name}: (in {step.cwd}) python {' '.join(arguments)}")
        status = subprocess.run(
            [sys.executable, *arguments], cwd=cwd, env=env, check=False
        ).returncode
        if status:
            print(f"[golden] {golden.name}: record step failed with status {status}")
            return status
    return 0


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROCKE), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _cpu_recordable(goldens: Iterable[Golden]) -> list[Golden]:
    """``goldens`` without those only a GPU can re-record, each named as skipped."""
    kept = []
    for golden in goldens:
        if golden.needs_gpu:
            print(f"[golden] {golden.name}: skipped, re-record it on a GPU")
        else:
            kept.append(golden)
    return kept


def _conflicted_goldens() -> list[Golden]:
    """The CPU-recordable goldens owning a file git reports as conflicted, each
    such file checked out from git's side "ours" so its record tool can read it.
    A GPU golden's files are left conflicted."""
    conflicted = _git("diff", "--name-only", "--diff-filter=U", "--relative")
    owned: dict[str, tuple[Golden, list[str]]] = {}
    for path in conflicted.split():
        golden = next((g for g in GOLDENS if g.owns(path)), None)
        if golden is None:
            print(f"[golden] {path}: not a golden; resolve it by hand")
            continue
        owned.setdefault(golden.name, (golden, []))[1].append(path)
    goldens = _cpu_recordable(owned[name][0] for name in sorted(owned))
    for golden in goldens:
        _git("checkout", "--ours", "--", *owned[golden.name][1])
    return goldens


def _rerecord(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    chosen = [bool(args.names), args.all, args.conflicted]
    if sum(chosen) != 1:
        parser.error("rerecord takes golden names, --all or --conflicted")
    families = [name for name in (args.only or "").split(",") if name.strip()]
    if args.only is not None and not (len(args.names) == 1 and families):
        parser.error("--only takes a family list and exactly one golden name")
    if args.conflicted:
        goldens = _conflicted_goldens()
        if not goldens:
            print("[golden] no conflicted golden file")
            return 0
    elif args.all:
        goldens = _cpu_recordable(GOLDENS)
    else:
        try:
            goldens = [golden_named(name) for name in args.names]
        except KeyError as error:
            parser.error(str(error.args[0]))
    if families and not goldens[0].takes_families:
        parser.error(f"golden {goldens[0].name} is re-recorded whole; drop --only")
    for golden in goldens:
        status = _run_golden_record(golden, families or None)
        if status:
            return status
    print("[golden] done; review `git diff`, then `git add` the re-recorded files")
    return 0


# --- list and check ---------------------------------------------------------------


def _describe(golden: Golden) -> dict[str, object]:
    return {
        "name": golden.name,
        "files": list(golden.files),
        "checked_by": list(golden.checked_by),
        "fingerprint": golden.fingerprint,
        "needs_gpu": golden.needs_gpu,
        "takes_families": golden.takes_families,
        "rerecord": rerecord_command(golden.name),
        "record": [
            {
                "cwd": step.cwd,
                "argv": list(step.argv),
                "pythonpath": list(step.pythonpath),
                "only": list(step.only),
            }
            for step in golden.record
        ],
    }


def _list(args: argparse.Namespace) -> int:
    if args.json:
        print(json.dumps([_describe(golden) for golden in GOLDENS], indent=2))
        return 0
    for golden in GOLDENS:
        families = " [--only <families>]" if golden.takes_families else ""
        gpu = "  (needs a GPU)" if golden.needs_gpu else ""
        print(f"{golden.name}{families}{gpu}")
        for pattern in golden.files:
            print(f"    {pattern}")
    return 0


def _check(args: argparse.Namespace) -> int:
    problems = golden_problems(TreeLayout.source(ROCKE))
    for problem in problems:
        print(f"GOLDEN INDEX: {problem}")
    if problems:
        return 1
    print(f"golden index OK: {len(GOLDENS)} goldens")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The command line (also how a test parses a printed re-record command)."""
    parser = argparse.ArgumentParser(
        prog="tools/golden.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="the goldens and their files")
    listing.add_argument("--json", action="store_true", help="as JSON")
    rerecord = commands.add_parser("rerecord", help="re-record goldens")
    rerecord.add_argument("names", nargs="*", help="golden names (see `list`)")
    rerecord.add_argument(
        "--only", help="comma-separated families of the one named golden"
    )
    rerecord.add_argument(
        "--all", action="store_true", help="every golden a CPU can re-record"
    )
    rerecord.add_argument(
        "--conflicted",
        action="store_true",
        help="every golden with a file git reports as conflicted",
    )
    commands.add_parser("check", help="check the goldens on disk against the index")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list":
        return _list(args)
    if args.command == "rerecord":
        return _rerecord(args, parser)
    return _check(args)


if __name__ == "__main__":
    raise SystemExit(main())
