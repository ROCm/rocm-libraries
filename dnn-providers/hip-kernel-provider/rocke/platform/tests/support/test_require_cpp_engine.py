# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The rocke.tests.require_cpp_engine plugin, which the installed pytest
entries load where the build installs the shared C++ engine: in a pytest run
that loads it, a C++ case whose engine is missing is red, and one whose engine
is present runs.

Each case runs a one-test pytest session in a child process, with and without
the plugin; the child has no shared engine (``ROCKE_ONLINE_LIB=none``) when the
case asks it to, so the missing direction runs wherever this file does.
"""

from __future__ import annotations

import subprocess

from rocke.core import native_engine
from rocke.core.backend import CPP_STRICT_ENV
from rocke.tests.engines import native_engine_or_skip
from rocke.tests.lanes import REQUIRE_CPP_ENGINE_ENV
from rocke.tests.run_records import fixture_session_environment
from rocke.tests.subprocess_launcher import rocke_python_command, rocke_subprocess_env

_CPP_CASE = """
from rocke.tests import engines


def test_cpp_case():
    engines.native_engine_or_skip()
"""


def _run_cpp_case(tmp_path, *, plugin: bool, hide: bool) -> subprocess.CompletedProcess:
    case = tmp_path / "test_cpp_case.py"
    case.write_text(_CPP_CASE)
    plugin_arguments = ("-p", "rocke.tests.require_cpp_engine") if plugin else ()
    env = {
        key: value
        for key, value in rocke_subprocess_env(fixture_session_environment()).items()
        if key not in (CPP_STRICT_ENV, REQUIRE_CPP_ENGINE_ENV)
    }
    if hide:
        env[native_engine.SHARED_ENGINE_ENV] = native_engine.NO_SHARED_ENGINE
    return subprocess.run(
        rocke_python_command(
            "-m",
            "pytest",
            "-q",
            "-rs",
            "-p",
            "no:cacheprovider",
            *plugin_arguments,
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


def test_a_missing_engine_fails_the_run_with_the_plugin(tmp_path):
    """With the plugin, a C++ case whose engine is missing fails, saying why."""
    run = _run_cpp_case(tmp_path, plugin=True, hide=True)
    assert run.returncode == 1, run.stdout + run.stderr
    assert "1 failed" in run.stdout, run.stdout
    assert (
        f"{REQUIRE_CPP_ENGINE_ENV}=1 requires the C++ engine: no shared C++ engine"
        in run.stdout
    ), run.stdout


def test_a_missing_engine_skips_without_the_plugin(tmp_path):
    """Without the plugin, the same case skips, as waiting for the lane."""
    run = _run_cpp_case(tmp_path, plugin=False, hide=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "1 skipped" in run.stdout, run.stdout
    assert "waits for build-variant" in run.stdout, run.stdout


def test_a_present_engine_runs_the_case_with_the_plugin(tmp_path):
    """With the plugin and the engine found, the case runs and passes."""
    native_engine_or_skip()
    run = _run_cpp_case(tmp_path, plugin=True, hide=False)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "1 passed" in run.stdout, run.stdout
