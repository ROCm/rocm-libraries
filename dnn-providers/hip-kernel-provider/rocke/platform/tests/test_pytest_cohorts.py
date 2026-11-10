# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Installed cohorts cover the same items without relying on source topology."""

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest
from rocke.tests.subprocess_launcher import rocke_python_command, rocke_subprocess_env

RUNNER = Path(__file__).resolve().with_name("run_pytest_cohort.py")
_SPEC = importlib.util.spec_from_file_location("pytest_cohort_runner", RUNNER)
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


@pytest.fixture
def payload(tmp_path):
    root = tmp_path / "tests"
    root.mkdir()
    shutil.copy2(RUNNER, root / RUNNER.name)
    (root / "pytest.ini").write_text(
        "[pytest]\naddopts = --import-mode=importlib\npython_files = test_*.py\n"
        "markers = integration: excluded native build integration\n"
    )
    test = (
        "import pytest\n"
        "@pytest.mark.parametrize('value', [1, 2])\n"
        "def test_case(value): assert value > 0\n"
        "@pytest.mark.integration\n"
        "def test_integration(): raise AssertionError('must not run')\n"
    )
    for layer in (*runner.LAYERS, "new_layer", "library"):
        (root / layer).mkdir()
        (root / layer / "test_same_name.py").write_text(test)
    (root / "test_staged_only.py").write_text(test)
    (root / "__pycache__").mkdir()
    (root / ".pytest_cache").mkdir()
    return root


def invoke(args, root):
    return subprocess.run(
        rocke_python_command(*args),
        cwd=root.parent,
        env=rocke_subprocess_env(),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def test_cohorts_collect_exactly_the_unpartitioned_items(payload):
    baseline = invoke(
        [
            "-m",
            "pytest",
            "-c",
            str(payload / "pytest.ini"),
            str(payload),
            "--ignore",
            str(payload / "library"),
            "-m",
            "not integration",
            "--collect-only",
            "-q",
        ],
        payload,
    )
    assert baseline.returncode == 0, baseline.stdout + baseline.stderr

    def nodeids(output):
        return [line for line in output.splitlines() if "::test_case[" in line]

    expected = nodeids(baseline.stdout)
    actual = []
    for cohort in runner.COHORTS:
        result = invoke(
            [str(payload / RUNNER.name), cohort, "--collect-only", "-q"], payload
        )
        assert result.returncode == 0, result.stdout + result.stderr
        actual.extend(nodeids(result.stdout))
    assert len(expected) == 10
    assert len(actual) == len(set(actual))
    assert set(actual) == set(expected)


def test_runner_preserves_pytest_failure_status(payload):
    (payload / "core/test_failure.py").write_text("def test_failure(): assert False\n")
    result = invoke([str(payload / RUNNER.name), "core", "-q"], payload)
    assert result.returncode == 1
    assert "test_failure" in result.stdout


@pytest.mark.parametrize("cohort", ["core", "helpers", "instances", "other"])
def test_missing_cohort_is_an_error(tmp_path, cohort):
    with pytest.raises(ValueError, match="missing|empty"):
        runner.cohort_paths(tmp_path, cohort)


def test_unknown_cohort_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="unknown"):
        runner.cohort_paths(tmp_path, "misspelled")
