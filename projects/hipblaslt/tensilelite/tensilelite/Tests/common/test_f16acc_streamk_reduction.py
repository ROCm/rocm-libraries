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

"""f16 accumulation: StreamK's parallel reduction and in-kernel fixup give the same bits.

Runs f16acc_streamk_reduction_gfx1250.yaml twice with TENSILE_PERSISTENT_REDUCTION_FORCE_MODE forcing each path
(the launch summary confirms which one ran) and compares the printed D tensors.
"""

import os
import re
import subprocess
import sys

import pytest

_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemm", "gfx12", "f16acc_streamk_reduction_gfx1250.yaml")
# 6 tiles of 6 iterations: 18 workgroups give each tile three pieces of two iterations in both paths.
_GRID = "18"
_LAUNCH_SUMMARY = "0x200000"
_VALUES = re.compile(r"^[\[\]() 0-9eE+\-.,naif]*$")


def _tensor_d(output):
    # "D: <descriptor>", then "Tensor(<sizes>, data_ptr: <address>)", then the values.
    lines = output.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("D: ")), None)
    assert start is not None, "no D tensor printed"
    values = []
    for line in lines[start + 2:]:
        if not _VALUES.match(line):
            break
        values.append(line)
    return values


def _run(mode, tensile_args, tmp_path):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path), "TENSILE_PERSISTENT_FIXED_GRID": _GRID,
           "TENSILE_DB": _LAUNCH_SUMMARY, "TENSILE_PERSISTENT_REDUCTION_FORCE_MODE": mode}
    script = "import sys; from Tensile import Tensile; Tensile.Tensile(sys.argv[1:])"
    result = subprocess.run([sys.executable, "-c", script, _CONFIG, str(tmp_path / ("mode" + mode)), *tensile_args],
                            env=env, capture_output=True, text=True)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output[-4000:]
    return output


@pytest.mark.gfx1250
def test_f16acc_streamk_parallel_reduction_and_in_kernel_fixup_are_bitwise_identical(pytestconfig, tensile_args,
                                                                                     tmp_path):
    if pytestconfig.getoption("--build-only") or pytestconfig.getoption("--use-cache"):
        pytest.skip("runs kernels; split mode active")
    tensors = {}
    for mode, reduction in (("1", "parallel(DP)"), ("0", "tree")):
        output = _run(mode, tensile_args, tmp_path)
        launches = re.findall(r"^\s+reduction\s+=\s+(\S+)", output, re.M)
        assert launches and set(launches) == {reduction}, (mode, launches)
        tensors[mode] = _tensor_d(output)
    assert tensors["1"] and tensors["1"] == tensors["0"]
