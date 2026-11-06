# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Runs the UI's JavaScript tests (*.test.mjs: the legal range, the bundle text) under node, so
pytest covers them."""

import shutil
import subprocess
from pathlib import Path

import pytest

TESTS = Path(__file__).parent


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_ui_modules_js() -> None:
    result = subprocess.run(
        ["node", "--test", *map(str, sorted(TESTS.glob("*.test.mjs")))],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
