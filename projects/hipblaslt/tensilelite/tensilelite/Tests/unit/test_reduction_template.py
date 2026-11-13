# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import subprocess

import pytest

from tensilelite import resources

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("phase", ["host", "device"])
@pytest.mark.parametrize("extended_host_min_max", [False, True], ids=["default", "extended"])
def test_reduction_template_compiles(tmp_path, cxx_compiler, phase, extended_host_min_max):
    resources.copy_static_headers(tmp_path)
    source = tmp_path / "reduction.hip"
    # Width 1 requires a nonzero array size; width 4 exercises the vector path.
    source.write_text(
        """
#include <hip/hip_runtime.h>
#include "ReductionTemplate.h"

__global__ void reduction_vw1(float* in, float* out)
{
    reductionKernel_ijk<float, float, 16, 4, 1>(in, out, 16, 4, 16);
}

__global__ void reduction_vw4(float* in, float* out)
{
    reductionKernel_ijk<float, float, 16, 4, 4>(in, out, 64, 4, 64);
}
""",
        encoding="utf-8",
    )
    command = [
        cxx_compiler,
        "-x",
        "hip",
        "-std=c++17",
        f"--offload-{phase}-only",
        # Syntax compilation needs a target, but does not need GPU hardware.
        "--offload-arch=gfx942",
        "-nogpulib",
        "-fsyntax-only",
        "-I",
        str(tmp_path),
    ]
    if extended_host_min_max:
        command.append("-D__HIP_DEFINE_EXTENDED_HOST_MIN_MAX__=1")
    command.append(str(source))
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
