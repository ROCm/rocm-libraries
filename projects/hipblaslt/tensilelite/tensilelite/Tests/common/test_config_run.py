################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

"""Run-only test phase for YAML kernel configs.

This module is activated by the ``--use-cache`` pytest flag. Two mechanisms
keep it from running in other modes: the ``pytest_ignore_collect`` hook in
``conftest.py`` excludes it at collection time, and the test function itself
calls ``pytest.skip`` if the flag is absent (fallback for pytest versions or
invocation styles where the hook is not called).

Role in the split-CI workflow::

    build (test_config_build.py)  -->  artifact (.tar.gz)  -->  run (this file)

This phase extracts a pre-built artifact from ``--artifact-dir`` and benchmarks
the cached kernels on the target GPU. It requires a GPU and an artifact
previously produced by ``test_config_build.py``. The artifact is extracted into
``tmpdir`` rather than back into ``--artifact-dir``, so the shared artifact
directory is never modified by the run phase.

For local single-machine testing (build + run in one pytest session) see
``test_config.py``, which calls ``_build`` and ``_run`` via isolated
subprocesses to exercise the same artifact round-trip.
"""

import os

import py
import pytest

from Tensile import Tensile

from artifact_helpers import artifact_name_for_config, extract_artifact


def _run(config: str, output_dir: str, artifact_dir: str, tensile_args: list[str]) -> None:
    """Extract a pre-built artifact and run benchmarks against it.

    Callable from both the pytest wrapper below and from test_config.py via
    subprocess (where it runs in a clean process to avoid global-state bleed).
    """
    artifact_name = artifact_name_for_config(config)
    tarball = os.path.join(artifact_dir, artifact_name + ".tar.gz")
    assert os.path.isfile(tarball), f"Artifact tarball not found: {tarball}"
    extract_artifact(tarball, output_dir)
    Tensile.Tensile([config, output_dir, "--use-cache", *tensile_args])


def test_config_run(tensile_args: list[str], config: str, tmpdir: py.path.local, pytestconfig: pytest.Config) -> None:
    """Pytest wrapper: extract a pre-built artifact and benchmark on the GPU.

    Activated only when ``--use-cache`` is passed; collection is skipped in all
    other modes by ``conftest.pytest_ignore_collect``. Requires a GPU.

    ``--artifact-dir`` must point to the directory where ``test_config_build``
    wrote the artifact. The artifact is extracted into ``tmpdir`` so the shared
    artifact directory is not modified by this phase.
    """
    if not pytestconfig.getoption("--use-cache"):
        pytest.skip("requires --use-cache")
    artifact_name = artifact_name_for_config(config)
    output_dir = os.path.join(tmpdir.strpath, artifact_name)
    _run(config, output_dir, pytestconfig.getoption("--artifact-dir"), tensile_args)
