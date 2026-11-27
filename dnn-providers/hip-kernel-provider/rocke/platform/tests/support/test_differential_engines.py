# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The rocke.tests.differential_engines plugin, which the installed GPU pytest
entries load where the build installs the shared C++ engine: in a pytest run
that loads it, every lowering resolves to both engines, in the session's
process and in each process it starts, unless a test chooses its engine
itself.

Each case runs a one-test pytest session in a child process, with and without
the plugin.
"""

from __future__ import annotations

import subprocess

from rocke.core.backend import BACKEND_CPP, BACKEND_ENV
from rocke.tests.differential_engines import PLUGIN_NAME
from rocke.tests.run_records import fixture_session_environment
from rocke.tests.subprocess_launcher import rocke_python_command, rocke_subprocess_env

# A one-test file the child session runs; it names the backend variable that
# the plugin sets.
_BACKEND_CASE = f"""
import subprocess
import sys

from rocke.core.backend import resolve_backend


def test_backend(monkeypatch):
    child = subprocess.run(
        [sys.executable, "-c",
         "from rocke.core.backend import resolve_backend; print(resolve_backend())"],
        capture_output=True, text=True, check=True,
    )
    print(f"session={{resolve_backend()}} child={{child.stdout.strip()}}")
    monkeypatch.setenv("{BACKEND_ENV}", "python")
    print(f"chosen={{resolve_backend()}} argument={{resolve_backend('cpp')}}")
"""


def _run_backend_case(tmp_path, *, plugin: bool) -> str:
    case = tmp_path / "test_backend_case.py"
    case.write_text(_BACKEND_CASE)
    plugin_arguments = ("-p", PLUGIN_NAME) if plugin else ()
    env = {
        key: value
        for key, value in rocke_subprocess_env(fixture_session_environment()).items()
        if key != BACKEND_ENV
    }
    run = subprocess.run(
        rocke_python_command(
            "-m",
            "pytest",
            "-q",
            "-s",
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
    assert run.returncode == 0, run.stdout + run.stderr
    return run.stdout


def test_the_plugin_lowers_with_both_engines_in_the_session_and_its_children(
    tmp_path,
):
    """With the plugin, the session and a process it starts resolve to both
    engines; a test that sets its own backend, or passes one, keeps it."""
    stdout = _run_backend_case(tmp_path, plugin=True)
    assert "session=both child=both" in stdout, stdout
    assert "chosen=python argument=cpp" in stdout, stdout


def test_without_the_plugin_the_session_keeps_the_default_backend(tmp_path):
    """Without the plugin, the session resolves to the package default, the
    C++ engine."""
    stdout = _run_backend_case(tmp_path, plugin=False)
    assert f"session={BACKEND_CPP} child={BACKEND_CPP}" in stdout, stdout
