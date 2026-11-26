# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""HasVALUReadSGPRHazard matches LLVM's hasVALUReadSGPRHazard(): gfx12 except gfx1250."""

import os
import shutil

import pytest

import rocisa


@pytest.mark.parametrize("isa, wavefront, expected", [
    ((12, 0, 0), 32, 1),
    ((12, 0, 1), 32, 1),
    ((12, 5, 0), 32, 0),
    ((11, 5, 1), 32, 0),
    ((9, 4, 2), 64, 0),
])
def test_valu_read_sgpr_hazard_cap(isa, wavefront, expected):
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    search_path = os.pathsep.join([
        os.path.join(rocm_path, "bin"),
        os.path.join(rocm_path, "lib", "llvm", "bin"),
    ])
    assembler = shutil.which("amdclang++", path=search_path) or "amdclang++"
    ri = rocisa.rocIsa.getInstance()
    prev = ri.getKernel()
    ri.init(isa, assembler, False)
    ri.setKernel(isa, wavefront)
    try:
        assert ri.getArchCaps()["HasVALUReadSGPRHazard"] == expected
    finally:
        ri.setKernel(tuple(prev.isa), prev.wavefrontSize)
