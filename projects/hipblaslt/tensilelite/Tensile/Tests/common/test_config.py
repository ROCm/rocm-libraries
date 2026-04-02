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

import contextlib
import os
import pytest
import shutil
import subprocess
import sys

from artifact_helpers import compress_output, extract_artifact


def _run_tensile_in_subprocess(args):
    """Run Tensile in a subprocess so each phase starts with clean global state.

    PYTHONPATH is set from sys.path so the child can import Tensile even
    when only reachable via Tests/conftest.py's runtime sys.path.append
    (uninstalled checkouts).
    """
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    subprocess.run(
        [sys.executable, "-c",
         "from Tensile.Tensile import Tensile; import sys; Tensile(sys.argv[1:])",
         *args],
        check=True,
        env=env,
    )


def test_config(tensile_args, config, tmpdir, pytestconfig):
    """Build -> compress -> wipe -> extract -> run, verifying the artifact round-trips."""
    if pytestconfig.getoption("--build-only") or pytestconfig.getoption("--use-cache"):
        pytest.skip("split mode active — use test_config_build or test_config_run")

    output_dir = os.path.join(tmpdir.strpath, "output")
    artifact_path = None

    build_args = [config, output_dir, "--build-only", *tensile_args]
    _run_tensile_in_subprocess(build_args)

    try:
        artifact_path = compress_output(output_dir)
        shutil.rmtree(output_dir)
        extract_artifact(artifact_path, output_dir)

        run_args = [config, output_dir, "--use-cache", *tensile_args]
        _run_tensile_in_subprocess(run_args)
    finally:
        if artifact_path:
            with contextlib.suppress(FileNotFoundError):
                os.remove(artifact_path)
