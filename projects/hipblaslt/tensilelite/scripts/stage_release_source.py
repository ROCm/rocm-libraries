#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Copy a clean TensileLite source tree for an isolated wheel build."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


_IGNORED_NAMES = {
    ".agents",
    ".codex",
    ".coverage",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "_skbuild",
    "build",
    "build-adaptor",
    "build_tmp",
    "dist",
    "htmlcov",
    "mutants",
}


def _ignore(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name in _IGNORED_NAMES or name.endswith((".egg-info", ".pyc", ".pyo"))
    }


def stage_source(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, ignore=_ignore)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args(argv)
    stage_source(args.source, args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
