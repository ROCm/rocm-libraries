################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""enableLDSTr + HasWMMA_V3 fp4/fp6 LDS-transpose local-read characterization.

Target code (Tensile/Components/LocalRead.py, LocalReadMFMA enableLDSTr block):
  With enableLDSTr and asmCaps["HasWMMA_V3"], the bpeDS==0.5 (F4/fp4) arm emits
  the LDS-transpose local reads. This config drives, before the fail-fast guard,
  LocalRead.py lines:
    861,862,863,864,865,867,868,869,870,871,873,874,875,876,877,878,879,880,
    882,883,884,888,889,890
  covering LocalReadX/wtRegStride/paddedOffset computation, the LdsPad branch,
  cal_offset_srcAddr, the plain destVgpr arm and (via HalfPLR fork [0,3]) the
  getHalfPLRValuStr arm.

EMIT-EXC template:
  A gfx1250 MX-F4/F6 NN-layout config derives to a VALID solution with
  UnrollMajorLDS{A,B}==0 and MXBlock{A,B}>0. Emission reaches the enableLDSTr
  WMMA_V3 block (recording the lines above), then localReadMX, where the M-major
  MX-scale layout floors tilePerRead to 0. localReadMX guards this and raises a
  descriptive Exception naming the unsupported M-major MX-scale local read
  (LocalRead.py:628, commit 4ab8940). The raise is the intended guard; the
  pre-raise coverage of the enableLDSTr lines above is the point of this test.

pytestmark = pytest.mark.unit. CPU-only; no GPU.
"""

import os

import pytest

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1250",
    "s10_enableldstr_wmma_v3_fp4_fp6_lds.yaml",
)


def test_s10_enableldstr_wmma_v3_fp4_fp6_lds_guard_raises():
    """enableLDSTr WMMA_V3 fp4/fp6 local reads run, then hit the M-major guard.

    Emitting the gfx1250 fp4/fp6 NN config executes the enableLDSTr WMMA_V3
    LDS-transpose local-read block in LocalRead.py (lines 861-890, recording
    coverage) and then reaches localReadMX, where tilePerRead floors to 0.
    localReadMX raises a descriptive Exception naming the unsupported M-major
    MX-scale local read; the raise is the intended fail-fast guard.
    """
    from config_harness import emit_kernels_from_config

    with pytest.raises(Exception, match=r"unsupported M-major MX-scale local read"):
        emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
