# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The name rkmon was run by, which every message, help text and next line uses.

- NAME: "rkmon".
- `name(argv0=None) -> str`: the name the command was invoked with, from sys.argv[0] when ARGV0 is None: the file's
  name when the shell found it on the PATH, else the path as it was typed (`./rkmon` from the package's docker/
  folder); NAME when Python runs a module, a test or `-c`; inside `run_by`'s block, the rkmon it names. A text
  calls it where the text is made.
- `run_by(rkmon)`: a context manager: inside its block, `name()` is RKMON, another package folder's own rkmon (its
  path, as a message shows it), so that a text made there names commands that run in that folder, as the next line
  of `rkmon package update --folder` does (rkmon_lib/package_update.py); None changes nothing.
- `named(text) -> str`: TEXT with that name in place of each `{program}`.

Standard library only, on the host's Python (deployment.HOST_PYTHON): rkmon runs it on the host.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys

NAME = "rkmon"
_RUN_BY: list = []  # the rkmon of each run_by block the code is in, the innermost last


def name(argv0: str | None = None) -> str:
    """The name the command was invoked with (from sys.argv[0] when argv0 is None; run_by's inside its block)."""
    if argv0 is None and _RUN_BY:
        return _RUN_BY[-1]
    argv0 = sys.argv[0] if argv0 is None else argv0
    base = os.path.basename(argv0 or "")
    # python -m unittest names itself "python3 -m unittest"
    if not base or base.endswith(".py") or base.startswith("-") or " -m " in argv0:
        return NAME
    return base if shutil.which(base) == argv0 else argv0


@contextlib.contextmanager
def run_by(rkmon: str | None):
    """Inside the block, name() is RKMON: see the module's docstring."""
    if rkmon is None:
        yield
        return
    _RUN_BY.append(rkmon)
    try:
        yield
    finally:
        _RUN_BY.pop()


def named(text: str) -> str:
    """TEXT with the program's name in place of {program}."""
    return str(text).replace("{program}", name())
