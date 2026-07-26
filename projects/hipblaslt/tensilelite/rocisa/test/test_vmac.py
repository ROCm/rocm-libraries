# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import shutil

import rocisa
from rocisa.container import vgpr
from rocisa.instruction import VMacF32


def test_vmac_f32_uses_fma_fallback_when_fmac_is_unavailable():
    isa = (9, 0, 12)
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    search_path = os.pathsep.join(
        [
            os.path.join(rocm_path, "bin"),
            os.path.join(rocm_path, "lib", "llvm", "bin"),
        ]
    )
    assembler = shutil.which("amdclang++", path=search_path) or "amdclang++"

    target = rocisa.rocIsa.getInstance()
    target.init(isa, assembler, False)
    target.setKernel(isa, 64)

    asm_caps = target.getAsmCaps()
    assert not asm_caps["v_fmac_f32"]
    assert asm_caps["v_fma_f32"]

    instruction = VMacF32(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2))
    assert str(instruction).strip() == "v_fma_f32 v0, v1, v2, v0"
