# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A rocKE pytest session starts without numpy.

Each pytest root's conftest loads ``rocke.tests.pytest_roots`` when its session
starts, so a module that helper imports and that needs numpy at import time stops
every session where numpy is not installed, the tests that need no numpy included.
numpy is blocked in a child interpreter (``sys.modules["numpy"] = None`` makes its
import raise), which then imports the helper as a conftest does.
"""

from __future__ import annotations

import subprocess
import sys

from rocke.tests.subprocess_launcher import rocke_subprocess_env

_CHILD_TIMEOUT_SECONDS = 60
_IMPORT_WITHOUT_NUMPY = "; ".join(
    ("import sys", "sys.modules['numpy'] = None", "import rocke.tests.pytest_roots")
)


def test_the_session_helpers_import_without_numpy():
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_WITHOUT_NUMPY],
        env=rocke_subprocess_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=_CHILD_TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, result.stderr[-3000:]
