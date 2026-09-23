################################################################################
#
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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
import signal
import shutil
import subprocess
import sys
import time

import py
import pytest

from artifact_helpers import artifact_name_for_config
from config_helpers import materializeConfig

_COMMON_DIR = os.path.dirname(os.path.abspath(__file__))
_PROCESS_TERMINATION_GRACE_SECONDS = 5.0


def _terminate_direct_process(process: subprocess.Popen) -> None:
    """Terminate and reap one process when process groups are unavailable."""
    try:
        process.terminate()
    except OSError:
        pass

    try:
        process.wait(timeout=_PROCESS_TERMINATION_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    except OSError:
        return

    try:
        process.kill()
    except OSError:
        pass
    try:
        process.wait(timeout=_PROCESS_TERMINATION_GRACE_SECONDS)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _reap_direct_process(process: subprocess.Popen) -> None:
    """Reap the group leader without allowing cleanup to mask an exception."""
    if process.poll() is not None:
        return
    try:
        process.wait(timeout=_PROCESS_TERMINATION_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    except OSError:
        return

    try:
        process.kill()
    except OSError:
        pass
    try:
        process.wait(timeout=_PROCESS_TERMINATION_GRACE_SECONDS)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _signal_process_group(process: subprocess.Popen, sig: int) -> bool:
    """Signal a POSIX helper process group; return False if unavailable."""
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return True


def _process_tree_is_alive(process: subprocess.Popen) -> bool:
    """Whether the helper or another member of its process group is alive."""
    if os.name == "posix":
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    return process.poll() is None


def _wait_for_process_tree(
    process: subprocess.Popen, timeout: float
) -> bool:
    """Wait up to ``timeout`` seconds for the helper process group to exit."""
    deadline = time.monotonic() + timeout
    while _process_tree_is_alive(process):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        if process.poll() is None:
            try:
                process.wait(timeout=min(0.05, remaining))
            except subprocess.TimeoutExpired:
                pass
        else:
            # The group leader can exit before one of its descendants.
            time.sleep(min(0.05, remaining))
    return True


def _terminate_process_tree(process: subprocess.Popen) -> None:
    """Terminate a helper tree, escalate after a grace period, and reap it."""
    if os.name != "posix" or not _signal_process_group(process, signal.SIGTERM):
        _terminate_direct_process(process)
        return

    if not _wait_for_process_tree(process, _PROCESS_TERMINATION_GRACE_SECONDS):
        _signal_process_group(process, getattr(signal, "SIGKILL"))
        _wait_for_process_tree(process, _PROCESS_TERMINATION_GRACE_SECONDS)

    # Reap the direct child even when one of its descendants outlives the
    # bounded SIGKILL wait.
    _reap_direct_process(process)


def _run_in_process_group(command: list[str], env: dict[str, str]) -> None:
    """Run ``command`` and tear down all descendants if it does not succeed."""
    popenArgs = {"env": env}
    if os.name == "posix":
        popenArgs["start_new_session"] = True
    elif hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        popenArgs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    process = subprocess.Popen(command, **popenArgs)
    try:
        returnCode = process.wait()
        if returnCode:
            raise subprocess.CalledProcessError(returnCode, command)
    except BaseException:
        _terminate_process_tree(process)
        raise


def _call_helper_in_subprocess(
    module: str,
    func: str,
    config: str,
    output_dir: str,
    artifact_dir: str,
    tensile_args: list[str],
    artifact_name: str,
) -> None:
    """Call module.func(config, output_dir, artifact_dir, tensile_args) in a subprocess.

    Each phase runs in a clean interpreter so Tensile's global state from the
    build phase cannot bleed into the run phase (uninstalled checkout case).
    PYTHONPATH is forwarded from sys.path so the child can import Tensile.
    """
    script = (
        f"import sys; sys.path.insert(0, {repr(_COMMON_DIR)}); "
        f"from {module} import {func}; "
        f"{func}(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[5:], "
        f"artifact_name=sys.argv[4])"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    _run_in_process_group(
        [
            sys.executable,
            "-c",
            script,
            config,
            output_dir,
            artifact_dir,
            artifact_name,
            *tensile_args,
        ],
        env,
    )


def test_config(
    tensile_args: list[str],
    config,
    tmpdir: py.path.local,
    pytestconfig: pytest.Config,
) -> None:
    """Pytest wrapper: run the full build→artifact→run round-trip on a single machine.

    Activated in the default mode (no ``--build-only`` / ``--use-cache`` flags).
    Requires a GPU. See the module docstring for a description of the four steps.
    """
    if pytestconfig.getoption("--build-only") or pytestconfig.getoption("--use-cache"):
        pytest.skip("split mode active — use test_config_build or test_config_run")
    config_path = materializeConfig(config, tmpdir.strpath)
    artifact_name = artifact_name_for_config(
        config.source_path, config.shard_label
    )
    output_dir = os.path.join(tmpdir.strpath, artifact_name)
    artifact_dir = tmpdir.strpath
    artifact_path = os.path.join(artifact_dir, artifact_name + ".tar.gz")

    _call_helper_in_subprocess(
        "test_config_build",
        "_build",
        config_path,
        output_dir,
        artifact_dir,
        tensile_args,
        artifact_name,
    )
    shutil.rmtree(output_dir)
    try:
        _call_helper_in_subprocess(
            "test_config_run",
            "_run",
            config_path,
            output_dir,
            artifact_dir,
            tensile_args,
            artifact_name,
        )
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(artifact_path)
