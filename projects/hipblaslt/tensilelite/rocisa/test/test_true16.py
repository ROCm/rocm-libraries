# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for the true16 half-select support.

These exercise pure assembly-string generation plus the version-based NoSDWA
arch-cap lookup, so they require neither LLVM (no assembler is executed by
rocIsa.init) nor a GPU. They cover:

  * RegisterContainer half-select rendering (.l / .h),
  * the true16 16-bit conditional select VCndMaskB16,
  * the ECvt* helpers picking the true16 (NoSDWA) vs legacy (SDWA) encoding,
  * the NoSDWA-gated t16() helper.
"""

import os
import shutil

import pytest

import rocisa
from rocisa.container import vgpr
from rocisa.enum import HighBitSel
from rocisa.instruction import ECvtF16toF32, ECvtF32toF16, VCndMaskB16, t16

# NoSDWA (true16) archs are gfx11/gfx12; legacy SDWA archs are gfx9/gfx10.
TRUE16_ISA = (12, 0, 0)
LEGACY_ISA = (9, 4, 2)


def _assembler_path():
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    search_path = os.pathsep.join(
        [
            os.path.join(rocm_path, "bin"),
            os.path.join(rocm_path, "lib", "llvm", "bin"),
        ]
    )
    # rocIsa.init only records the path; it does not run the assembler, so a
    # missing amdclang++ is fine for these string-only tests.
    return shutil.which("amdclang++", path=search_path) or "amdclang++"


def _use_isa(isa):
    inst = rocisa.rocIsa.getInstance()
    inst.init(isa, _assembler_path(), False)  # idempotent per ISA
    inst.setKernel(isa, 64)
    return inst


@pytest.fixture(autouse=True)
def _restore_legacy_isa():
    # rocIsa is a process-global singleton; restore the gfx942 default after each
    # test so sibling tests that assume the default arch are unaffected.
    yield
    _use_isa(LEGACY_ISA)


def test_half_select_render():
    # gfx942 keeps HasVgprMSB off, so a single VGPR renders without an _hi pad.
    _use_isa(LEGACY_ISA)

    assert str(vgpr(0)) == "v0"
    assert str(vgpr(0).lo()) == "v0.l"
    assert str(vgpr(0).hi()) == "v0.h"

    reg = vgpr(0)
    reg.setHalfSelect(HighBitSel.HIGH)
    assert str(reg) == "v0.h"
    reg.setHalfSelect(HighBitSel.NONE)
    assert str(reg) == "v0"


def test_vcndmask_b16_renders_halves():
    _use_isa(TRUE16_ISA)

    dst, src0, src1 = vgpr(0), vgpr(1), vgpr(2)
    dst.setHalfSelect(HighBitSel.LOW)
    src0.setHalfSelect(HighBitSel.LOW)
    src1.setHalfSelect(HighBitSel.HIGH)

    text = str(VCndMaskB16(dst=dst, src0=src0, src1=src1))
    assert text.startswith("v_cndmask_b16 ")
    assert "v0.l" in text
    assert "v1.l" in text
    assert "v2.h" in text


def test_ecvt_f16_to_f32_true16_vs_legacy():
    # true16: half selected via the src operand's .h suffix, no SDWA modifier.
    _use_isa(TRUE16_ISA)
    text = str(ECvtF16toF32(dst=vgpr(0), src=vgpr(1), sel=HighBitSel.HIGH))
    assert "v_cvt_f32_f16" in text
    assert "v1.h" in text
    assert "src0_sel" not in text

    # legacy: half selected via SDWA src0_sel, no true16 suffix.
    _use_isa(LEGACY_ISA)
    text = str(ECvtF16toF32(dst=vgpr(0), src=vgpr(1), sel=HighBitSel.HIGH))
    assert "v_cvt_f32_f16" in text
    assert "src0_sel:WORD_1" in text
    assert ".h" not in text


def test_ecvt_f32_to_f16_true16_vs_legacy():
    _use_isa(TRUE16_ISA)
    # No sel on true16 defaults to the low half (a plain 16-bit cvt is illegal there).
    assert "v0.l" in str(ECvtF32toF16(dst=vgpr(0), src=vgpr(1)))
    assert "v0.h" in str(ECvtF32toF16(dst=vgpr(0), src=vgpr(1), sel=HighBitSel.HIGH))

    _use_isa(LEGACY_ISA)
    # No sel on legacy is a plain cvt: no half suffix, no SDWA modifier.
    text = str(ECvtF32toF16(dst=vgpr(0), src=vgpr(1)))
    assert "v_cvt_f16_f32 v0, v1" in text
    assert ".l" not in text and ".h" not in text
    # sel on legacy packs the result via SDWA dst_sel.
    assert "dst_sel:WORD_1" in str(
        ECvtF32toF16(dst=vgpr(0), src=vgpr(1), sel=HighBitSel.HIGH)
    )


def test_t16_gating():
    _use_isa(TRUE16_ISA)
    assert ".h" in str(t16(vgpr(1), HighBitSel.HIGH))
    assert ".l" in str(t16(vgpr(1), HighBitSel.LOW))

    # On legacy targets t16 is a no-op: the operand is returned untagged.
    _use_isa(LEGACY_ISA)
    assert ".h" not in str(t16(vgpr(1), HighBitSel.HIGH))
    assert ".l" not in str(t16(vgpr(1), HighBitSel.LOW))


if __name__ == "__main__":
    test_half_select_render()
    test_vcndmask_b16_renders_halves()
    test_ecvt_f16_to_f32_true16_vs_legacy()
    test_ecvt_f32_to_f16_true16_vs_legacy()
    test_t16_gating()
