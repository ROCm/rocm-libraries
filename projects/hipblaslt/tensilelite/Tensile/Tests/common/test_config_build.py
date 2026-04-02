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

from artifact_helpers import artifact_name_for_config, compress_output


def test_config_build(tensile_args, config, tmpdir, pytestconfig):
    """Build kernels without benchmarking. Produces a .tar.gz artifact.

    Only runs when --build-only is passed. Does not require the target GPU.
    The compressed artifact is written to --artifact-dir so it persists
    after the test for CI to upload.
    """
    if not pytestconfig.getoption("--build-only"):
        pytest.skip("requires --build-only")

    artifact_dir = pytestconfig.getoption("--artifact-dir")
    output_dir = os.path.join(tmpdir.strpath, "output")
    args = [config, output_dir, "--build-only", *tensile_args]
    Tensile.Tensile(args)

    artifact_name = artifact_name_for_config(config)
    compress_output(output_dir, dest_dir=artifact_dir, name=artifact_name)
