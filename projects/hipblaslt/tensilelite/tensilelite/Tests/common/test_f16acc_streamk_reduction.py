# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""f16 accumulation: StreamK's parallel reduction and in-kernel fixup give the same bits.

Runs f16acc_streamk_reduction_gfx1250.yaml twice with TENSILE_PERSISTENT_REDUCTION_FORCE_MODE forcing each path
(the launch summary confirms which one ran) and compares the printed D tensors.
"""

import math
import os
import re
import subprocess
import sys

import pytest

from tensilelite.Tests.gpu_detection import get_available_archs

_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemm", "gfx12", "f16acc_streamk_reduction_gfx1250.yaml")
# 6 tiles of 6 iterations: 18 workgroups give each tile three pieces of two iterations in both paths.
_GRID = "18"
_LAUNCH_SUMMARY = "0x200000"
_HAS_GFX1250 = "gfx1250" in get_available_archs()
_VALUES = re.compile(r"^[\[\]() 0-9eE+\-.,naif]*$")


def _tensor_d(output):
    # "D: <descriptor>", then "Tensor(<sizes>, data_ptr: <address>)", then per batch "(<coordinate>)", "[", one line of
    # space-separated values per row, "]".
    lines = output.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("D: ")), None)
    assert start is not None, "no D tensor printed"
    sizes = re.match(r"Tensor\(([\d, ]+), data_ptr:", lines[start + 1])
    assert sizes, lines[start + 1]
    values = []
    for line in lines[start + 2:]:
        if not _VALUES.match(line):
            break
        if not line.startswith("("):
            values += line.strip("[] ").split()
    count = math.prod(int(size) for size in sizes.group(1).split(","))
    assert len(values) == count, (len(values), sizes.group(1))
    return values


def _run(mode, tensile_args, tmp_path):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path), "TENSILE_PERSISTENT_FIXED_GRID": _GRID,
           "TENSILE_DB": _LAUNCH_SUMMARY, "TENSILE_PERSISTENT_REDUCTION_FORCE_MODE": mode}
    script = "import sys; from tensilelite.run import main; main(sys.argv[1:])"
    result = subprocess.run([sys.executable, "-c", script, _CONFIG, str(tmp_path / ("mode" + mode)), *tensile_args],
                            env=env, capture_output=True, text=True)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output[-4000:]
    return output


@pytest.mark.gfx1250
@pytest.mark.skipif(not _HAS_GFX1250, reason="gfx1250 GPU not available")
def test_f16acc_streamk_parallel_reduction_and_in_kernel_fixup_are_bitwise_identical(pytestconfig, tensile_args,
                                                                                     tmp_path):
    if pytestconfig.getoption("--build-only") or pytestconfig.getoption("--use-cache"):
        pytest.skip("runs kernels; split mode active")
    tensors = {}
    # The launch summary prints "tree" for every in-kernel fixup, linear or tree.
    for mode, reduction in (("1", "parallel(DP)"), ("0", "tree")):
        output = _run(mode, tensile_args, tmp_path)
        launches = re.findall(r"^\s+reduction\s+=\s+(\S+)", output, re.M)
        assert launches and set(launches) == {reduction}, (mode, launches)
        tensors[mode] = _tensor_d(output)
    assert tensors["1"] == tensors["0"]
