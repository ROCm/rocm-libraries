#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""List and re-record rocKE's goldens, from the golden index.

Run from the rocke directory::

    python tools/golden.py list [--json]
    python tools/golden.py rerecord kda_gfx942
    python tools/golden.py rerecord representative --only attention_dense,gemm
    python tools/golden.py rerecord --all          # every golden a CPU can write
    python tools/golden.py rerecord --conflicted   # after a merge or rebase

``rerecord`` runs each golden's own record commands (``golden_index.py``) with
this interpreter. Review the change before re-recording: a re-record accepts any
output. ``--conflicted`` resolves every golden file git reports as conflicted:
two changes re-recorded the same family, so the file is re-recorded whole on the
merged tree (git's side of the file is taken only to clear the markers); a
golden only a GPU can re-record is left conflicted. Then ``git add`` the files.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ROCKE = _HERE.parent  # tools -> rocke (in a source tree)
# The index beside this file, and the rocke package for the golden-file rules.
for _module_root in (_HERE, ROCKE / "platform" / "python"):
    if _module_root.is_dir() and str(_module_root) not in sys.path:
        sys.path.insert(0, str(_module_root))

from golden_index import GOLDENS, Golden, golden_named  # noqa: E402

__all__ = ["build_parser", "main"]


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


# --- list ---------------------------------------------------------------


def _describe(golden: Golden) -> dict[str, object]:
    return {
        "name": golden.name,
        "files": list(golden.files),
        "checked_by": list(golden.checked_by),
        "fingerprint": golden.fingerprint,
        "needs_gpu": golden.needs_gpu,
        "takes_families": golden.takes_families,
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list":
        return _list(args)
    return _rerecord(args, parser)


if __name__ == "__main__":
    raise SystemExit(main())
