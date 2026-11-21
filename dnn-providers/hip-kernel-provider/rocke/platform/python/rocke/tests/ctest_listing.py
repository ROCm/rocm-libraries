# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The CTest entries registered in a directory, as ``ctest --show-only=json-v1`` lists them.

One home for reading a registration without running it: the runners and checks
that need entry names, commands, labels or properties ask CTest rather than
parsing ``CTestTestfile.cmake``. CTest omits an entry's ``command`` when it
cannot find the program, so list a built tree, with the test interpreter on
``PATH`` for script entries.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path

__all__ = ["list_ctest_entries"]


def list_ctest_entries(
    test_dir: Path,
    *,
    config: str | None = None,
    name_pattern: str | None = None,
    env: Mapping[str, str] | None = None,
) -> list[dict]:
    """The json-v1 ``tests`` of ``test_dir``, optionally for one build
    ``config`` and only names matching the ``name_pattern`` regex (``-R``).
    ``env`` replaces the environment CTest evaluates the registration in."""
    command = ["ctest", "--test-dir", str(test_dir)]
    if config is not None:
        command += ["-C", config]
    command.append("--show-only=json-v1")
    if name_pattern is not None:
        command += ["-R", name_pattern]
    listed = subprocess.run(
        command, check=True, capture_output=True, text=True, env=env
    )
    return json.loads(listed.stdout)["tests"]
