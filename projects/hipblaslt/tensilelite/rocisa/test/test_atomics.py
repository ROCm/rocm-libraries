# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Assembly-text regressions for the scalar atomics.

Covers:
  * s_atomic_umax_x2 and s_atomic_cmpswap_x2, with SMEM offset and glc.
  * the s_atomic_inc / s_atomic_dec argument forms, which differ in whether
    they carry an soffset operand.
"""

import os
import shutil

import pytest
import rocisa
from rocisa.container import SMEMModifiers, sgpr
from rocisa.instruction import (
    SAtomicCmpswapX2,
    SAtomicDec,
    SAtomicInc,
    SAtomicUmaxX2,
)


@pytest.fixture(scope="module", autouse=True, params=[(9, 4, 2), (9, 5, 0)], ids=rocisa.isaToGfx)
def _isa_context(request):
    isa = request.param
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    search_path = os.pathsep.join(
        [
            os.path.join(rocm_path, "bin"),
            os.path.join(rocm_path, "lib", "llvm", "bin"),
        ]
    )
    assembler = shutil.which("amdclang++", path=search_path) or "amdclang++"
    rocisa.rocIsa.getInstance().init(isa, assembler, False)
    rocisa.rocIsa.getInstance().setKernel(isa, 64)


def test_satomic_umax_x2_emits_smem_offset():
    inst = SAtomicUmaxX2(
        dst=sgpr(4, 2),
        base=sgpr(8, 2),
        soffset=sgpr(12),
        smem=SMEMModifiers(offset=64),
    )

    assert str(inst).strip() == "s_atomic_umax_x2 s[4:5], s[8:9], s12 offset:64"


def test_satomic_cmpswap_x2_glc_emits():
    inst = SAtomicCmpswapX2(
        dst=sgpr(4, 4),
        base=sgpr(8, 2),
        soffset=sgpr(12),
        smem=SMEMModifiers(glc=True, offset=64),
    )

    assert str(inst).strip() == "s_atomic_cmpswap_x2 s[4:7], s[8:9], s12 offset:64 glc"


def test_scalar_atomic_soffset_forms():
    inc = SAtomicInc(dst=sgpr(4), base=sgpr(8, 2), soffset=sgpr(12))
    dec = SAtomicDec(dst=sgpr(4), base=sgpr(8, 2))

    assert str(inc).strip() == "s_atomic_inc s4, s[8:9], s12"
    assert str(dec).strip() == "s_atomic_dec s4, s[8:9]"
