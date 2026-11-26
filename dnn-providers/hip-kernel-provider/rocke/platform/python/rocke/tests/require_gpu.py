# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""pytest plugin: fail the session when it finds no GPU.

Loaded as ``-p rocke.tests.require_gpu`` by every installed GPU CTest entry
(``_ROCKE_INSTALLED_PYTEST_GPU_TEST_POLICY`` in ``platform/CMakeLists.txt``).
Without it, a runner whose HIP fails to start would turn every GPU entry
green: each test that looks for a GPU and finds none skips, whichever way it
looks (``rocke.tests.gpu``, or a test's own query of the HIP runtime). So
before collection the plugin asks for HIP's current device, as
``rocke.tests.gpu.probe_device_arch`` does, and ends the session with a
failure naming why when there is none. Which GPU arch the entry finds is still
each test's business. A ``--collect-only`` session runs no test, so it passes
nothing as skipped and is not checked: listing an entry's tests (as
``tools/check_entry_times.py`` does) works on a host without a GPU.

The question runs in a child process: loading HIP maps its own COMGR, and a
process with a COMGR mapped loads no different one, which would change the
compiler the session's tests use and the one ``rocke.tests.quarantine``
identifies.
"""

from __future__ import annotations

import subprocess

import pytest

from .subprocess_launcher import rocke_python_command, rocke_subprocess_env

__all__ = ["GPU_PROBE", "PLUGIN_NAME", "gpu_absence", "pytest_sessionstart"]

PLUGIN_NAME = __name__

#: The child's code: prints the arch of HIP's current device, or the reason
#: there is none.
GPU_PROBE = """
from rocke.tests.gpu import probe_device_arch

arch, absence = probe_device_arch()
print(f"arch={arch}" if arch else f"absent={absence}")
"""
# Bounds a HIP runtime that hangs while it starts.
_PROBE_TIMEOUT_S = 60


def gpu_absence(
    probe: str = GPU_PROBE, timeout_seconds: float = _PROBE_TIMEOUT_S
) -> str | None:
    """Why HIP finds no current device, asked by running ``probe`` in a child
    process, or None when it finds one. A child that fails, answers something
    else or does not answer within ``timeout_seconds`` is a reason too."""
    try:
        run = subprocess.run(
            rocke_python_command("-c", probe),
            env=rocke_subprocess_env(),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return f"the GPU probe did not answer within {timeout_seconds:g} s"
    lines = run.stdout.strip().splitlines()
    answer = lines[-1] if lines else ""
    kind, _, detail = answer.partition("=")
    if run.returncode != 0 or kind not in ("arch", "absent"):
        return f"the GPU probe failed (exit {run.returncode}): {run.stderr.strip()}"
    return detail if kind == "absent" else None


def pytest_sessionstart(session: pytest.Session) -> None:
    """End the session as failed when it finds no GPU, unless it only
    collects."""
    if session.config.option.collectonly:
        return
    absence = gpu_absence()
    if absence is not None:
        pytest.exit(
            f"{PLUGIN_NAME}: this GPU test run found no GPU: {absence}",
            returncode=pytest.ExitCode.TESTS_FAILED,
        )
