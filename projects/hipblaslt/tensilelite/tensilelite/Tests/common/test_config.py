################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

"""Combined build-then-run test phase for YAML kernel configs (default mode).

This module runs when neither ``--build-only`` nor ``--use-cache`` is passed.
Two mechanisms keep it from running in split-CI mode: the
``pytest_ignore_collect`` hook in ``conftest.py`` excludes it at collection
time, and the test function itself calls ``pytest.skip`` if either flag is
present (fallback for pytest versions or invocation styles where the hook is
not called).

This is the single-machine equivalent of the full split-CI workflow. It drives
the same ``_build`` and ``_run`` helpers that live in ``test_config_build.py``
and ``test_config_run.py``, verifying the full artifact round-trip in one
pytest session:

  1. ``_build``  — compile kernels, compress the output to a temporary artifact
  2. wipe        — delete the build output directory
  3. ``_run``    — extract the artifact, benchmark against the cached kernels
  4. cleanup     — delete the temporary artifact

Wiping the output between steps 1 and 3 confirms the artifact is genuinely
self-contained and not relying on leftover build state.

Each phase is launched in a subprocess so that Tensile's process-level global
state accumulated during the build phase cannot bleed into the run phase.
The helpers are imported by name in the child process, keeping the logic
defined in exactly one place (the build/run modules) rather than duplicated
here.
"""

import contextlib
import os
import shutil
import subprocess
import sys

import py
import pytest

from artifact_helpers import artifact_name_for_config

_COMMON_DIR = os.path.dirname(os.path.abspath(__file__))


def _call_helper_in_subprocess(
    module: str,
    func: str,
    config: str,
    output_dir: str,
    artifact_dir: str,
    tensile_args: list[str],
) -> None:
    """Call module.func(config, output_dir, artifact_dir, tensile_args) in a subprocess.

    Each phase runs in a clean interpreter so Tensile's global state from the
    build phase cannot bleed into the run phase (uninstalled checkout case).
    PYTHONPATH is forwarded from sys.path so the child can import Tensile.
    """
    script = (
        f"import sys; sys.path.insert(0, {repr(_COMMON_DIR)}); "
        f"from {module} import {func}; "
        f"{func}(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    subprocess.run(
        [sys.executable, "-c", script, config, output_dir, artifact_dir, *tensile_args],
        check=True,
        env=env,
    )


def test_config(tensile_args: list[str], config: str, tmpdir: py.path.local, pytestconfig: pytest.Config) -> None:
    """Pytest wrapper: run the full build→artifact→run round-trip on a single machine.

    Activated in the default mode (no ``--build-only`` / ``--use-cache`` flags).
    Requires a GPU. See the module docstring for a description of the four steps.
    """
    if pytestconfig.getoption("--build-only") or pytestconfig.getoption("--use-cache"):
        pytest.skip("split mode active — use test_config_build or test_config_run")
    artifact_name = artifact_name_for_config(config)
    output_dir = os.path.join(tmpdir.strpath, artifact_name)
    artifact_dir = tmpdir.strpath
    artifact_path = os.path.join(artifact_dir, artifact_name + ".tar.gz")

    _call_helper_in_subprocess("test_config_build", "_build", config, output_dir, artifact_dir, tensile_args)
    shutil.rmtree(output_dir)
    try:
        _call_helper_in_subprocess("test_config_run", "_run", config, output_dir, artifact_dir, tensile_args)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(artifact_path)
