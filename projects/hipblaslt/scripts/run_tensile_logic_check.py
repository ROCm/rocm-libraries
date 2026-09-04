#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Run the installed TensileLite logic validator without checkout path injection."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def _hipblaslt_root() -> Path:
    root = Path(__file__).resolve().parent.parent
    if not (root / "library").is_dir():
        raise SystemExit(f"Cannot find the hipBLASLt library directory below {root}")
    return root


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    root = _hipblaslt_root()
    if args and not args[0].startswith("-"):
        logic_path = Path(args.pop(0))
    else:
        logic_path = root / "library"
    if not logic_path.exists():
        raise SystemExit(f"Library logic path not found: {logic_path}")
    has_known_bugs_option = any(
        arg in ("--known-bugs", "--use-bundled-known-bugs")
        or arg.startswith("--known-bugs=")
        for arg in args
    )
    if not has_known_bugs_option:
        args.insert(0, "--use-bundled-known-bugs")
    if "--check-all" not in args:
        args.append("--check-all")
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "tensilelite",
            "logic",
            str(logic_path.resolve()),
            *args,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
