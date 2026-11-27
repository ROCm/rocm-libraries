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

Each phase is launched in a subprocess so that TensileLite's process-level global
state accumulated during the build phase cannot bleed into the run phase.
The helpers are imported by name in the child process, keeping the logic
defined in exactly one place (the build/run modules) rather than duplicated
here.
"""

import atexit
import contextlib
import os
import shutil
import signal
import subprocess
import sys
import warnings

import py
import pytest

from artifact_helpers import artifact_name_for_config

_COMMON_DIR = os.path.dirname(os.path.abspath(__file__))

# Process groups of helpers still running, so a worker stopped by SIGTERM (a
# cancelled CI job, for example) or a normal interpreter exit kills their
# clients too. POSIX only: on Windows start_new_session is ignored and only the
# helper itself can be killed. Nothing runs on SIGKILL or os._exit (which
# pytest-timeout's thread method uses); there the job's own cleanup has to reap
# the clients, as the rocjitsu launcher does as a child subreaper.
_live_helper_groups: set[int] = set()


def _kill_live_helper_groups() -> None:
    for group in list(_live_helper_groups):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(group, signal.SIGKILL)
    _live_helper_groups.clear()


def _kill_helpers_then_terminate(signum, frame) -> None:
    _kill_live_helper_groups()
    signal.signal(signum, _previous_sigterm_handler)
    os.kill(os.getpid(), signum)


_previous_sigterm_handler = signal.SIG_DFL
if os.name == "posix":
    atexit.register(_kill_live_helper_groups)
    # Only the main thread may install signal handlers; xdist workers import
    # this module there.
    with contextlib.suppress(ValueError):
        _previous_sigterm_handler = signal.signal(signal.SIGTERM, _kill_helpers_then_terminate)


def _call_helper_in_subprocess(
    module: str,
    func: str,
    config: str,
    output_dir: str,
    artifact_dir: str,
    tensile_args: list[str],
) -> None:
    """Call module.func(config, output_dir, artifact_dir, tensile_args) in a subprocess.

    Each phase runs in a clean interpreter so TensileLite's global state from the
    build phase cannot bleed into the run phase (uninstalled checkout case).
    PYTHONPATH is forwarded from sys.path so the child can import tensilelite.
    """
    script = (
        f"import sys; sys.path.insert(0, {repr(_COMMON_DIR)}); "
        f"from {module} import {func}; "
        f"{func}(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    # Own process group (POSIX) so a timeout interrupt can reap the helper's
    # whole child tree, not orphan it.
    proc = subprocess.Popen(
        [sys.executable, "-c", script, config, output_dir, artifact_dir, *tensile_args],
        env=env,
        start_new_session=True,
    )
    if os.name == "posix":
        _live_helper_groups.add(proc.pid)
    returncode = None
    try:
        returncode = proc.wait()
    finally:
        # The helper may exit before its client. Own the whole group until
        # the phase finishes, including normal exit and interruption races.
        if os.name == "posix":
            if returncode == 0 and _group_has_members(proc.pid):
                # A client that outlives a successful helper is a helper bug;
                # say so before killing it rather than hiding the leak.
                warnings.warn(f"{config}: a client outlived its helper and was killed")
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            _live_helper_groups.discard(proc.pid)
        elif proc.poll() is None:
            proc.kill()
        proc.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, proc.args)


def _group_has_members(group: int) -> bool:
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    return True


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
