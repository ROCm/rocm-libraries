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

"""Build-only test phase for YAML kernel configs.

This module is activated by the ``--build-only`` pytest flag. Two mechanisms
keep it from running in other modes: the ``pytest_ignore_collect`` hook in
``conftest.py`` excludes it at collection time, and the test function itself
calls ``pytest.skip`` if the flag is absent (fallback for pytest versions or
invocation styles where the hook is not called).

Role in the split-CI workflow::

    build (this file)  -->  artifact (.tar.gz)  -->  run (test_config_run.py)

This phase compiles kernels and packs the results into a per-config .tar.gz
artifact written to ``--artifact-dir``. It does not require a GPU — kernel
compilation is CPU-only — and is intended to run on a CPU-heavy CI node.
The artifact is uploaded by CI and consumed by the run phase on a GPU node.

For local single-machine testing (build + run in one pytest session) see
``test_config.py``, which calls ``_build`` and ``_run`` via isolated
subprocesses to exercise the same artifact round-trip.
"""

import os

import py
import pytest

from Tensile import Tensile

from artifact_helpers import artifact_name_for_config, compress_output


def _build(config: str, output_dir: str, artifact_dir: str, tensile_args: list[str]) -> None:
    """Build kernels and compress the result to artifact_dir.

    Callable from both the pytest wrapper below and from test_config.py via
    subprocess (where it runs in a clean process to avoid global-state bleed).
    """
    Tensile.Tensile([config, output_dir, "--build-only", *tensile_args])
    compress_output(output_dir, dest_dir=artifact_dir, name=artifact_name_for_config(config))


def test_config_build(tensile_args: list[str], config: str, tmpdir: py.path.local, pytestconfig: pytest.Config) -> None:
    """Pytest wrapper: build kernels for one YAML config and emit a .tar.gz artifact.

    Activated only when ``--build-only`` is passed; collection is skipped in all
    other modes by ``conftest.pytest_ignore_collect``. Does not require a GPU.

    The artifact is written to ``--artifact-dir`` (not ``tmpdir``) so it
    survives after the test and can be uploaded by CI for the run phase.
    """
    if not pytestconfig.getoption("--build-only"):
        pytest.skip("requires --build-only")
    output_dir = os.path.join(tmpdir.strpath, artifact_name_for_config(config))
    _build(config, output_dir, pytestconfig.getoption("--artifact-dir"), tensile_args)
