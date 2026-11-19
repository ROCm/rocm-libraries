# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The strict-skips plugin is what keeps CI from passing by not running a test. Each case
runs a throwaway suite in a child pytest, once without the plugin and once with it."""

import os
import re
import subprocess
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[2] / "pytest_plugins"

SUITE = """
import pytest

def test_passes():
    pass

def test_skips_for_a_missing_prerequisite():
    pytest.skip("no compiler")

@pytest.mark.skipif(True, reason="no such platform")
def test_skipif_for_a_missing_prerequisite():
    pass

@pytest.fixture
def fixture_that_skips():
    pytest.skip("no binary")

def test_skip_raised_in_a_fixture(fixture_that_skips):
    pass

@pytest.mark.xfail(reason="known bug")
def test_xfail():
    assert False

@pytest.mark.allow_skip(reason="author supplies the input")
def test_allowed_skip():
    pytest.skip("no input")

@pytest.mark.allow_skip(reason="no such platform")
class TestAllowedClass:
    @pytest.mark.skipif(True, reason="no such platform")
    def test_member(self):
        pass

@pytest.mark.allow_skip
def test_allow_skip_without_a_reason():
    pytest.skip("no input")
"""


def _run(tmp_path, *extra):
    (tmp_path / "test_suite.py").write_text(SUITE)
    env = dict(os.environ, PYTHONPATH=str(PLUGIN_DIR))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider", *extra],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    return {
        match.group(1): match.group(2)
        for match in re.finditer(
            r"^test_suite\.py::(\S+) (PASSED|SKIPPED|FAILED|ERROR|XFAIL)",
            result.stdout,
            flags=re.MULTILINE,
        )
    }, result.stdout


def test_without_the_plugin_every_skip_stays_a_skip(tmp_path):
    outcomes, out = _run(tmp_path)
    assert "FAILED" not in outcomes.values() and "ERROR" not in outcomes.values(), out
    assert outcomes["test_skips_for_a_missing_prerequisite"] == "SKIPPED"


def test_with_the_plugin_a_skip_fails_unless_marked_allowed(tmp_path):
    outcomes, out = _run(tmp_path, "-p", "hipdnn_strict_skips")
    assert outcomes == {
        "test_passes": "PASSED",
        "test_skips_for_a_missing_prerequisite": "FAILED",
        "test_skipif_for_a_missing_prerequisite": "ERROR",
        "test_skip_raised_in_a_fixture": "ERROR",
        "test_xfail": "FAILED",
        "test_allowed_skip": "SKIPPED",
        "TestAllowedClass::test_member": "SKIPPED",
        "test_allow_skip_without_a_reason": "FAILED",
    }, out
    assert "skipped in a strict run: no compiler" in out


def test_a_module_level_skip_fails_the_collection_under_the_plugin(tmp_path):
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    pass\n")
    (tmp_path / "test_modskip.py").write_text(
        "import pytest\npytest.importorskip('no_such_module_xyz')\n"
    )
    env = dict(os.environ, PYTHONPATH=str(PLUGIN_DIR))

    def run(*extra):
        return subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *extra],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
        )

    assert run().returncode == 0
    strict = run("-p", "hipdnn_strict_skips")
    assert strict.returncode != 0, strict.stdout
    assert "skipped while collecting test_modskip.py in a strict run" in strict.stdout
