# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The rocke.tests.require_gpu plugin (its module docstring says why every
installed GPU pytest entry loads it).

The session cases run pytest in a child process, with and without the plugin.
Those whose child's HIP sees no device (``HIP_VISIBLE_DEVICES=-1``,
``ROCR_VISIBLE_DEVICES=-1``) run wherever this file does; the one that needs a
GPU is marked ``gpu``. The child's test asks for the GPU through the HIP
runtime directly, as some GPU tests do, so it holds the plugin to every way a
test looks for the GPU, not only ``rocke.tests.gpu``. The probe cases run real
child processes that fail or hang.
"""

from __future__ import annotations

import subprocess

import pytest
from rocke.tests.gpu import current_device_arch
from rocke.tests.require_gpu import PLUGIN_NAME, gpu_absence
from rocke.tests.run_records import fixture_session_environment
from rocke.tests.subprocess_launcher import rocke_python_command, rocke_subprocess_env

_GPU_CASE = """
import pytest


def test_gpu_case():
    from rocke.runtime import hip_module

    try:
        arch = hip_module.get_device_arch(hip_module.current_device())
    except Exception as exc:
        pytest.skip(f"no GPU: {exc}")
    if not arch:
        pytest.skip("no GPU")
"""


def _run_gpu_case(
    tmp_path, *, plugin: bool, hide: bool, collect_only: bool = False
) -> subprocess.CompletedProcess[str]:
    case = tmp_path / "test_gpu_case.py"
    case.write_text(_GPU_CASE)
    plugin_arguments = ("-p", PLUGIN_NAME) if plugin else ()
    env = rocke_subprocess_env(fixture_session_environment())
    if hide:
        env |= {"HIP_VISIBLE_DEVICES": "-1", "ROCR_VISIBLE_DEVICES": "-1"}
    return subprocess.run(
        rocke_python_command(
            "-m",
            "pytest",
            "-q",
            "-rs",
            "-p",
            "no:cacheprovider",
            *plugin_arguments,
            *(("--collect-only",) if collect_only else ()),
            f"--basetemp={tmp_path / 'basetemp'}",
            str(case),
        ),
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def test_a_missing_gpu_fails_the_run_with_the_plugin(tmp_path):
    """With the plugin, a run that finds no GPU fails, saying why, before its
    test can skip."""
    run = _run_gpu_case(tmp_path, plugin=True, hide=True)
    assert run.returncode == pytest.ExitCode.TESTS_FAILED, run.stdout + run.stderr
    assert f"{PLUGIN_NAME}: this GPU test run found no GPU" in (run.stdout + run.stderr)
    assert "skipped" not in run.stdout, run.stdout


def test_a_missing_gpu_skips_without_the_plugin(tmp_path):
    """Without the plugin, the same test skips: the gap the plugin closes."""
    run = _run_gpu_case(tmp_path, plugin=False, hide=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "1 skipped" in run.stdout, run.stdout


@pytest.mark.gpu
def test_a_present_gpu_runs_the_test_with_the_plugin(tmp_path):
    """With the plugin and a GPU, the test runs."""
    current_device_arch()
    run = _run_gpu_case(tmp_path, plugin=True, hide=False)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "1 passed" in run.stdout, run.stdout


def test_a_collect_only_run_without_a_gpu_lists_its_tests_with_the_plugin(tmp_path):
    """A --collect-only run runs no test, so the plugin leaves it alone: an
    entry's tests can be listed on a host without a GPU."""
    run = _run_gpu_case(tmp_path, plugin=True, hide=True, collect_only=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "test_gpu_case" in run.stdout, run.stdout


def test_a_probe_that_fails_is_the_reason():
    """A probe child that exits non-zero is a reason, naming its exit code and
    what it printed on stderr."""
    probe = "import sys; sys.stderr.write('HIP refused'); sys.exit(3)"
    assert gpu_absence(probe) == "the GPU probe failed (exit 3): HIP refused"


def test_a_probe_that_does_not_answer_in_time_is_the_reason():
    """A probe child that hangs is a reason once its time is up, rather than an
    error that ends the session unnamed."""
    probe = "import time; time.sleep(60)"
    assert gpu_absence(probe, timeout_seconds=0.5) == (
        "the GPU probe did not answer within 0.5 s"
    )
