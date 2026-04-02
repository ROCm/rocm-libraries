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

import os
import pytest

from Tensile import Tensile

from artifact_helpers import artifact_name_for_config, extract_artifact


def test_config_run(tensile_args, config, tmpdir, pytestconfig):
    """Run benchmarks using a previously built cache.

    Only runs when --use-cache is passed. Requires the target GPU and
    --artifact-dir pointing to where build artifacts are stored.
    Extracts the artifact into tmpdir so --artifact-dir is not modified.
    """
    if not pytestconfig.getoption("--use-cache"):
        pytest.skip("requires --use-cache")

    artifact_dir = pytestconfig.getoption("--artifact-dir")
    artifact_name = artifact_name_for_config(config)
    tarball = os.path.join(artifact_dir, artifact_name + ".tar.gz")
    assert os.path.isfile(tarball), \
        f"Artifact tarball not found: {tarball}"

    output_dir = os.path.join(tmpdir.strpath, artifact_name)
    extract_artifact(tarball, output_dir)

    args = [config, output_dir, "--use-cache", *tensile_args]
    Tensile.Tensile(args)
