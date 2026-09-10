################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""MX-scale M-major (UnrollMajorLDS==0) local-read guard characterization.

Target code:
  - KernelWriterAssembly.py MXBlockA/B VGPR macro cold branch
    (`if not kernel["UnrollMajorLDS*"]:`), lines ~1145 (A) / ~1185 (B).
  - Tensile/Components/LocalRead.py localReadMX (611-onward), specifically the
    tilePerRead==0 guard.

Background:
  A gfx1250 MX-F4 NN-layout config derives to a VALID solution with
  UnrollMajorLDS{A,B}==0 and MXBlock{A,B}>0. Unlike gfx950 (which rejects MX
  TLU=1 subtile geometry pre-emit), gfx1250 admits this solution and emission
  reaches the MX-scale VGPR-macro cold branch, then localReadMX.

  In that M-major layout a single 0.25-register local read spans fewer bytes
  than one MX scale unit (mxUnit = MatrixInstK // MXBlock = 128 // 32 = 4), so
  stridePerRead (=blockWidth*4 = 1.0) floors to tilePerRead == 0. The subsequent
  `vectorWidth // tilePerRead` previously raised an opaque
  ZeroDivisionError; localReadMX now guards it with a descriptive Exception.
  The M-major MX-scale local-read layout is unimplemented (MX scales are
  per-32-K-block, so K-major LDS is the only implemented layout).

  This is the only currently-known way to exercise these branches with a passing
  test: gfx950 rejects the config before emit, and gfx1250 cannot build the
  kernel. The test asserts the guarded Exception is raised, which proves the
  cold branch + localReadMX prologue were executed (coverage recorded) before
  the fail-fast.

pytestmark = pytest.mark.unit. CPU-only; no GPU.
"""

import os

import pytest

pytestmark = pytest.mark.unit

_ARCH_1250 = "gfx1250"

_MX_F4_NN_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1250",
    "mx_f4_nn_umlds0.yaml",
)


def test_mx_umlds0_localreadmx_guard_raises():
    """M-major MX-scale local read fails fast with a descriptive Exception.

    Deriving + emitting the gfx1250 MX-F4 NN config reaches the MX-scale
    VGPR-macro cold branch in KernelWriterAssembly.py and then localReadMX, where
    tilePerRead floors to 0. localReadMX guards this and raises a descriptive
    Exception naming the unsupported M-major MX-scale local read; without the
    guard the code raised a bare ZeroDivisionError.
    """
    from config_harness import emit_kernels_from_config

    with pytest.raises(Exception, match=r"unsupported M-major MX-scale local read"):
        emit_kernels_from_config(_MX_F4_NN_CONFIG, limit=8, arch=_ARCH_1250)
