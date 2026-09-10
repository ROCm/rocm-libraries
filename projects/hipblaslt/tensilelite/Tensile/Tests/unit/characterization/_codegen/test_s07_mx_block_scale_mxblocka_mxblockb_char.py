# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""S07 MX block-scale (MXBlockA & MXBlockB) classic-allocator characterization.

Target code:
  - Tensile/KernelWriter.py MX VGPR/SGPR allocation cluster in
    vgprAllocationImplClassic() and surrounding MX macro/alloc arms.
    Pre-raise coverage of KernelWriter.py lines 7291, 7297, 8465, 8466, 8472,
    8504, 8505, 8511, 9898, 9935.

Background:
  A gfx1250 MX-F4 NN-layout config (MXBlockA=MXBlockB=32) derives to a VALID,
  NON-subtile solution. Unlike gfx950 (which forces UseSubtileImpl for MX and
  therefore never reaches the classic MX allocator), gfx1250 routes MX through
  vgprAllocationImplClassic(), exercising the MX VGPR/SGPR ValuPack allocation
  arms listed above during _initKernel.

  In that M-major layout (UnrollMajorLDS{A,B}==0), localReadMX is unimplemented:
  MX scales are per-32-K-block, so K-major LDS is the only implemented layout.
  emission reaches the classic allocator cluster (coverage recorded) and then
  fails fast in LocalRead.py:628 with the intended guard (commit 4ab8940),
  raising a descriptive Exception naming the unsupported M-major MX-scale local
  read. Asserting that raise proves the allocation cluster ran before the
  fail-fast.

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
    "s07_mx_block_scale_mxblocka_mxblockb.yaml",
)


def test_s07_mx_block_scale_classic_alloc_then_guard_raises():
    """MX block-scale classic allocation runs, then M-major local read guard raises.

    Emitting the gfx1250 MX-F4 NN config routes MX through the classic
    (non-subtile) VGPR allocator, exercising the MX ValuPack allocation cluster
    in KernelWriter.py (lines 7291, 7297, 8465, 8466, 8472, 8504, 8505, 8511,
    9898, 9935). Emission then reaches localReadMX for the unimplemented M-major
    MX-scale layout and fails fast with the intended guard (commit 4ab8940),
    raising a descriptive Exception. The raise proves the allocation cluster
    executed before the fail-fast.
    """
    from config_harness import emit_kernels_from_config

    with pytest.raises(Exception, match=r"unsupported M-major MX-scale local read"):
        emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
