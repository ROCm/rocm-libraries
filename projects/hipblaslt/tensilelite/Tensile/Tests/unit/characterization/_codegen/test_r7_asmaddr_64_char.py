################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R7 -- gfx942 AsmAddressCalculation 64-bit / Scale-vector / InitialStrides arms.

Three configs and one direct-driver test exercise the remaining uncovered branches
of Tensile/AsmAddressCalculation.py (miss=163, 60% coverage target):

Config A -- asmaddr_srvw_scale.yaml:
  fp8n TN + UseScaleAB="Vector" + UseScaleAlphaVec=1 + StoreRemapVectorWidth=4
  Target ranges (LdsOffsetBias != 0 guard inside Scale* arms):
    - 320     : optSingleColVgpr ScaleAlphaVec LdsOffsetBias arm
    - 327-329 : optSingleColVgpr ScaleAVec arm body
    - 333-334 : optSingleColVgpr ScaleAVec LdsOffsetBias arm
    - 340-343 : optSingleColVgpr ScaleBVec arm body
    - 347-348 : optSingleColVgpr ScaleBVec LdsOffsetBias arm
    - 367-372 : optSharedColVgpr ScaleAlphaVec arm body
    - 376-377 : optSharedColVgpr ScaleAlphaVec LdsOffsetBias arm
    - 382-387 : optSharedColVgpr ScaleAVec arm body
    - 391-392 : optSharedColVgpr ScaleAVec LdsOffsetBias arm
    - 411-415 : optSharedColVgpr ScaleBVec arm body
    - 419-420 : optSharedColVgpr ScaleBVec LdsOffsetBias arm
    - 462-471 : else ScaleAlphaVec arms + LdsOffsetBias sub-arm
    - 482-490 : else ScaleAVec arms + LdsOffsetBias sub-arm
    - 497-498 : else ScaleBVec LdsOffsetBias arm

Config B -- asmaddr_initstrides.yaml:
  fp32 TN + UseInitialStridesCD=True (non-unit stride0 for D)
  Target ranges:
    - 269-273 : emitScaleToBpe non-unit stride0 (VMulLOU32 branch)

Direct-driver test:
  Calls AddrCalculation.incrementSrdMultipleRows() directly with positive,
  negative, and numRows==1 to hit the static method body (lines 779-816).

All pure-assert on emitted instructions and/or derived module content.
pytestmark=pytest.mark.unit. CPU-only.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx942"

_DIR = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx942",
)

_CFG_SRVW_SCALE = os.path.join(_DIR, "asmaddr_srvw_scale.yaml")
_CFG_INITSTRIDES = os.path.join(_DIR, "asmaddr_initstrides.yaml")


# ---------------------------------------------------------------------------
# Config-emit tests
# ---------------------------------------------------------------------------


def test_r7_asmaddr_srvw_scale_emits_assembly():
    """SRVW+ScaleAB+ScaleAlphaVec fp8n config emits real gfx942 assembly, err==0.

    Drives LdsOffsetBias != 0 sub-branches inside ScaleAlpha/ScaleAVec/ScaleBVec
    arms of emitScaleToBpe (lines 320, 327-334, 340-348, 367-392, 411-431,
    462-498).
    """
    results = emit_kernels_from_config(_CFG_SRVW_SCALE, limit=4, arch=_ARCH)
    assert len(results) >= 1, f"Expected >=1 kernel, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), (
        "All kernels must emit with err==0; "
        + str([(b, e) for (b, _s, e) in results if e != 0])
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 100, f"Kernel {base!r} source too short"
        assert ".amdgcn_target" in src, f"Kernel {base!r} missing .amdgcn_target"
        assert "gfx942" in src, f"Kernel {base!r} missing gfx942 arch marker"
        assert base.startswith("Cijk_"), f"Unexpected basename: {base!r}"
        # SRVW kernels use buffer_store_dword (for the LDS-based remap path),
        # not the raw SRD path; confirm real assembly with address instructions.
        assert "s_mul_i32" in src or "v_add_co_u32" in src, (
            f"Kernel {base!r} missing expected addressing instructions"
        )


def test_r7_asmaddr_initstrides_emits_assembly():
    """UseInitialStridesCD=True config emits real gfx942 assembly, err==0.

    Drives stride-D[0] as a runtime SGPR, making isConstUnitStride() return
    False, which activates lines 269-273 (VMulLOU32 scale by non-unit stride0).
    """
    results = emit_kernels_from_config(_CFG_INITSTRIDES, limit=4, arch=_ARCH)
    assert len(results) >= 1, f"Expected >=1 kernel, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), (
        "All kernels must emit with err==0; "
        + str([(b, e) for (b, _s, e) in results if e != 0])
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 100, f"Kernel {base!r} source too short"
        assert ".amdgcn_target" in src, f"Kernel {base!r} missing .amdgcn_target"
        assert "gfx942" in src, f"Kernel {base!r} missing gfx942 arch marker"
        assert base.startswith("Cijk_"), f"Unexpected basename: {base!r}"
        # UseInitialStridesCD generates stride loads before address computation.
        # The non-unit stride0 path emits a v_mul_lo_u32 for the coord0 scale.
        assert "v_mul_lo_u32" in src, (
            f"Kernel {base!r} missing v_mul_lo_u32 (non-unit stride0 scale)"
        )


# ---------------------------------------------------------------------------
# Direct-driver test: incrementSrdMultipleRows static method (lines 779-816)
# ---------------------------------------------------------------------------


def test_r7_incrementSrdMultipleRows_positive():
    """incrementSrdMultipleRows with numRows>1 emits SMulI32 + SAddU32/SAddCU32."""
    from Tensile.AsmAddressCalculation import AddrCalculation

    mod = AddrCalculation.incrementSrdMultipleRows(
        srcDstBaseSgpr="SrdD",
        strideSgpr="StrideD1",
        tmpSgpr="tmp0",
        numRows=3,
        bpe=4,
    )
    asm = str(mod)
    # numRows > 1 → SMulI32 (line 781-784)
    assert "s_mul_i32" in asm, f"Expected s_mul_i32 for numRows=3, got:\n{asm}"
    # numRows >= 0 → SAddU32 + SAddCU32 (lines 799-806)
    assert "s_add_u32" in asm, f"Expected s_add_u32 for positive numRows, got:\n{asm}"
    assert "s_addc_u32" in asm, f"Expected s_addc_u32 for positive numRows, got:\n{asm}"


def test_r7_incrementSrdMultipleRows_one():
    """incrementSrdMultipleRows with numRows==1 emits SLShiftLeftB32 + SAdd*."""
    from Tensile.AsmAddressCalculation import AddrCalculation

    mod = AddrCalculation.incrementSrdMultipleRows(
        srcDstBaseSgpr="SrdD",
        strideSgpr="StrideD1",
        tmpSgpr="tmp0",
        numRows=1,
        bpe=4,
    )
    asm = str(mod)
    # numRows == 1 → SLShiftLeftB32 (line 791-794)
    assert "s_lshl_b32" in asm, f"Expected s_lshl_b32 for numRows=1, got:\n{asm}"
    # numRows >= 0 → SAdd* (lines 799-806)
    assert "s_add_u32" in asm, f"Expected s_add_u32 for numRows=1, got:\n{asm}"


def test_r7_incrementSrdMultipleRows_negative():
    """incrementSrdMultipleRows with numRows<0 emits SMulI32 + SSubU32/SSubBU32."""
    from Tensile.AsmAddressCalculation import AddrCalculation

    mod = AddrCalculation.incrementSrdMultipleRows(
        srcDstBaseSgpr="SrdD",
        strideSgpr="StrideD1",
        tmpSgpr="tmp0",
        numRows=-2,
        bpe=4,
    )
    asm = str(mod)
    # numRows < 0 → SMulI32 with (-numRows)*bpe (line 786-789)
    assert "s_mul_i32" in asm, f"Expected s_mul_i32 for numRows=-2, got:\n{asm}"
    # numRows < 0 → SSubU32 + SSubBU32 (lines 808-815)
    assert "s_sub_u32" in asm, f"Expected s_sub_u32 for negative numRows, got:\n{asm}"
    assert "s_subb_u32" in asm, f"Expected s_subb_u32 for negative numRows, got:\n{asm}"


# ---------------------------------------------------------------------------
# Regression test: ROCM-25793 -- 32-bit overflow in SRD base address increment
# ---------------------------------------------------------------------------


def test_rocm25793_incrementSrd_must_handle_64bit_stride_product():
    """ROCM-25793: incrementSrdMultipleRows must compute a full 64-bit
    stride*bpe product so the SRD high word gets the upper 32 bits,
    not just the carry from the low-word add.

    The bug: s_lshl_b32 (or s_mul_i32) computes stride*bpe in 32 bits,
    then s_addc_u32 adds only the carry flag (src1=0) to SRD+1. When
    stride*bpe >= 2^32 the shift/mul silently wraps, corrupting the SRD
    base address. The next buffer_store writes to a wrong address and
    the GPU faults:

        s_lshl_b32  s12, s28, 2          ; s12 = stride << 2 (TRUNCATED)
        s_add_u32   s24, s24, s12        ; SRD_lo += truncated product
        s_addc_u32  s25, s25, 0          ; SRD_hi += carry only (WRONG)

    The fix must emit s_mul_hi_u32 (or equivalent) so s_addc_u32 adds
    the real high-word product instead of literal 0. This test FAILS on
    the buggy codegen and PASSES after the fix.
    """
    import re
    from Tensile.AsmAddressCalculation import AddrCalculation

    test_cases = [
        (1, 4, "numRows=1, bpe=4 (fp32 store, shift path)"),
        (1, 2, "numRows=1, bpe=2 (fp16/bf16 store, shift path)"),
        (3, 4, "numRows=3, bpe=4 (multi-row, mul path)"),
        (1, 1, "numRows=1, bpe=1 (int8/fp8 store, shift path)"),
    ]

    bugs_found = []
    for numRows, bpe, label in test_cases:
        mod = AddrCalculation.incrementSrdMultipleRows(
            srcDstBaseSgpr="SrdD",
            strideSgpr="StrideD1",
            tmpSgpr="tmp0",
            numRows=numRows,
            bpe=bpe,
        )
        asm = str(mod)

        # Bug pattern: s_addc_u32 with literal 0 as src1 means the high
        # word of the 64-bit SRD only gets the carry bit, not the actual
        # upper 32 bits of stride*numRows*bpe.
        if re.search(r's_addc_u32\s+\S+,\s*\S+,\s*0\b', asm):
            bugs_found.append((label, asm))

    assert not bugs_found, (
        "ROCM-25793 BUG DETECTED: SRD high-word increment uses literal 0 "
        "instead of the upper 32 bits of stride*bpe. This causes GPU memory "
        "faults when stride*bpe >= 2^32.\n\n"
        + "\n".join(
            f"--- {label} ---\n{asm}" for label, asm in bugs_found
        )
        + "\n\nExpected: s_mul_hi_u32 (or equivalent) to compute the high "
        "word, then s_addc_u32 using that result instead of 0."
    )
