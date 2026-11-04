# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""HIPDNN_REQUIRE_UHD_GEN_TESTS turns a prerequisite skip into a failed run (conftest.py)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2]

SKIPPING = {
    # Skipped at collection, as the suite's module-level importorskip guards do.
    "test_needs_a_package.py": (
        "import pytest\n"
        "pytest.importorskip('a_package_that_is_not_installed')\n"
        "def test_uses_it():\n"
        "    pass\n"
    ),
    # Skipped at run time, as the `evaluator` fixture does, beside a test that passes.
    "test_needs_a_binary.py": (
        "import pytest\n"
        "def test_runs():\n"
        "    pass\n"
        "def test_uses_the_binary():\n"
        "    pytest.skip('hipdnn_uhd_features was not found')\n"
    ),
}


def _run(tmp_path, required, suite):
    """pytest over `suite` with uhd_gen's conftest loaded as a plugin."""
    for name, text in suite.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    environment = dict(os.environ)
    environment.pop("HIPDNN_REQUIRE_UHD_GEN_TESTS", None)
    if required is not None:
        environment["HIPDNN_REQUIRE_UHD_GEN_TESTS"] = required
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(TOOLS), environment.get("PYTHONPATH")])
    )
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    command += ["-p", "uhd_gen.conftest", str(tmp_path)]
    return subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("required", [None, "", "0"])
def test_a_skip_passes_unless_required(tmp_path, required):
    result = _run(tmp_path, required, SKIPPING)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 skipped" in result.stdout


def test_a_required_run_fails_on_each_kind_of_skip_and_names_it(tmp_path):
    result = _run(tmp_path, "1", SKIPPING)

    assert result.returncode == 1, result.stdout + result.stderr
    lines = result.stdout.splitlines()
    for node, reason in (
        ("test_needs_a_package.py", "a_package_that_is_not_installed"),
        ("test_uses_the_binary", "hipdnn_uhd_features was not found"),
    ):
        assert any(node in line and reason in line for line in lines), result.stdout


def test_a_required_run_with_nothing_skipped_passes(tmp_path):
    result = _run(tmp_path, "1", {"test_plain.py": "def test_runs():\n    pass\n"})

    assert result.returncode == 0, result.stdout + result.stderr
