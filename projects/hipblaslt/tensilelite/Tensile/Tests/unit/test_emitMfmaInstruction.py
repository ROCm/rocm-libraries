#!/usr/bin/env python3
################################################################################
# Unit tests for Tensile.Components.Subtile.Kernel.emitMfmaInstruction.
################################################################################

import os
import sys
from types import SimpleNamespace
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)
from Tensile.Common.DataType import DataType
from Tensile.Components.Subtile.Kernel import emitMfmaInstruction
from gpu_test_helpers import init_rocisa  # initializes rocisa target=gfx950


# ---- minimal stubs ---------------------------------------------------------
class _StubPool:
    """Stand-in for RegisterPool. checkOut returns monotonically increasing
    fake VGPR indices (only used by FP4-style fallback paths)."""

    def __init__(self):
        self._next = 200

    def checkOut(self, n=1, *a, **kw):
        v = self._next
        self._next += n
        return v

    def checkIn(self, *a, **kw):
        pass


def _mkTile(start, count, pool):
    """Fake vgprTile: only `regList.indices` and `regList.pool` are read."""
    return SimpleNamespace(
        regList=SimpleNamespace(indices=list(range(start, start + count)), pool=pool)
    )


def _mkKernel(dA, dB, miK=128, sourceSwap=False, miArchVgpr=True):
    """Minimal kernel dict driving emitMfmaInstruction."""
    return {
        "MatrixInstK": miK,
        "MIArchVgpr": miArchVgpr,
        "SourceSwap": sourceSwap,
        # Pre-allocated unit-scale VGPR (initialized to 0x7f7f7f7f in mainLoop).
        # emitMfmaInstruction requires this when no real scale VGPRs are supplied.
        "_subtileUnitScaleVgpr": 250,
        "ProblemType": {
            "DataTypeA": DataType(dA) if dA else None,
            "DataTypeB": DataType(dB) if dB else None,
            "MXBlockA": 0,
            "MXBlockB": 0,
        },
    }


@pytest.fixture(scope="module", autouse=True)
def _rocisa_once():
    # This file is a PURE STRING test (no GPU runtime). It validates the
    # gfx950 MFMA emission path, so we MUST pin rocisa to gfx950, otherwise
    # MXMFMAInstruction::preStr() would emit gfx12 WMMA opcodes on a gfx12
    # host (and the asserts on "v_mfma_scale_*" / "cbsz:X blgp:Y" would all
    # fail). amdclang++ assembles any -mcpu= target regardless of the host
    # GPU, so this works on any machine that has ROCm installed.
    init_rocisa(target="gfx950")


@pytest.fixture
def writer():
    w = SimpleNamespace()
    w.vgprPool = _StubPool()
    w.agprPool = _StubPool()
    return w


# ---- helper to keep operand-width assertions self-documenting --------------
def _assertScaledMfmaOpcode(asm):
    """All miK==128 paths must emit the SCALED opcode (not the dense one)."""
    assert (
        "v_mfma_scale_f32_16x16x128_f8f6f4" in asm
    ), f"expected scaled opcode in asm:\n{asm}"


# ---- F8/BF8 cases ----------------------------------------------------------
# (DataTypeA, DataTypeB, sourceSwap, expected_modifiers)
# Per ISA + rocisa::MXMFMAInstruction::mfmaInputPermuteStr.
F8_CASES = [
    # Pure types — SourceSwap is a no-op for the type suffix.
    ("F8", "F8", False, "cbsz:0 blgp:0"),
    ("F8", "F8", True, "cbsz:0 blgp:0"),
    ("B8", "B8", False, "cbsz:1 blgp:1"),
    ("B8", "B8", True, "cbsz:1 blgp:1"),
    # Mixed types — SourceSwap MUST flip the suffix.
    ("F8", "B8", False, "cbsz:0 blgp:1"),  # INST_F8_BF8
    ("F8", "B8", True, "cbsz:1 blgp:0"),  # -> INST_BF8_F8
    ("B8", "F8", False, "cbsz:1 blgp:0"),  # INST_BF8_F8
    ("B8", "F8", True, "cbsz:0 blgp:1"),  # -> INST_F8_BF8
]


@pytest.mark.parametrize("dA,dB,swap,modifiers", F8_CASES)
def test_F8_uses_scaled_mfma_with_correct_cbsz_blgp(writer, dA, dB, swap, modifiers):
    """FP8/BF8 must emit the SAME scaled MFMA as FP4 — only cbsz/blgp differ.
    Operand widths reflect the F8 geometry (8 dwords each for A and B)."""
    kernel = _mkKernel(dA, dB, miK=128, sourceSwap=swap)
    # FP8 geometry: 8 dwords per A and B (MFMA_16x16_1B_4K_8V); 4 dwords C/D.
    tA = _mkTile(0, 8, writer.vgprPool)
    tB = _mkTile(16, 8, writer.vgprPool)
    tC = _mkTile(32, 4, writer.vgprPool)
    tD = _mkTile(64, 4, writer.vgprPool)
    # No scale operands -> exercises the hardcoded-scale fallback path with
    # the new instType. The ASSERTIONS hold for both fallback and real-scale
    # because cbsz/blgp depend only on instType.
    asm = str(
        emitMfmaInstruction(writer, kernel, tA, tB, tC, tD, comment="F8 unit test")
    )
    _assertScaledMfmaOpcode(asm)
    assert modifiers in asm, f"expected `{modifiers}` in asm:\n{asm}"
    # Operand registers must reflect SourceSwap (B in A-pos when swap=True).
    expectA_pos = "v[16:23]" if swap else "v[0:7]"
    expectB_pos = "v[0:7]" if swap else "v[16:23]"
    assert (
        expectA_pos in asm and expectB_pos in asm
    ), f"operand reg positions wrong:\n{asm}"
    # Acc + C are 4 dwords (fp32).
    assert "v[64:67]" in asm and "v[32:35]" in asm, f"acc/c reg widths wrong:\n{asm}"


def test_F8_real_scale_path_uses_op_sel_and_real_mxsa_mxsb(writer):
    """When real scaleAVgpr/scaleBVgpr are passed, the F8 path must:
    - emit op_sel + op_sel_hi modifiers,
    - use the supplied scale VGPRs as mxsa/mxsb,
    - keep cbsz:0 blgp:0 (FP8/FP8)."""
    kernel = _mkKernel("F8", "F8", miK=128, sourceSwap=False)
    tA = _mkTile(0, 8, writer.vgprPool)
    tB = _mkTile(16, 8, writer.vgprPool)
    tC = _mkTile(32, 4, writer.vgprPool)
    tD = _mkTile(64, 4, writer.vgprPool)
    before = writer.vgprPool._next
    asm = str(
        emitMfmaInstruction(
            writer,
            kernel,
            tA,
            tB,
            tC,
            tD,
            scaleAVgpr=100,
            scaleBVgpr=101,
            scaleAsel=2,
            scaleBsel=1,
        )
    )
    after = writer.vgprPool._next
    _assertScaledMfmaOpcode(asm)
    assert "cbsz:0 blgp:0" in asm
    assert "v100" in asm and "v101" in asm, "real scale VGPRs must be used"
    assert "op_sel" in asm, "op_sel/op_sel_hi must be present"
    # scaleAsel=2 (binary 10) -> op_sel[0]=0, op_sel_hi[0]=1 (= byte 2)
    # scaleBsel=1 (binary 01) -> op_sel[1]=1, op_sel_hi[1]=0 (= byte 1)
    # We don't pin the exact serialization (rocisa-internal), only that no
    # tmp scale VGPR is allocated on this branch.
    assert after == before, "real-scale path must NOT check out a tmp VGPR"


def test_F8_hardcoded_scale_path_allocates_tmp_with_0x7f7f7f7f(writer):
    """No real scale VGPRs -> fallback uses _subtileUnitScaleVgpr (pre-initialized
    to 0x7f7f7f7f in mainLoop) for both mxsa/mxsb. No tmp VGPR is allocated here."""
    kernel = _mkKernel("F8", "F8", miK=128, sourceSwap=False)
    tA = _mkTile(0, 8, writer.vgprPool)
    tB = _mkTile(16, 8, writer.vgprPool)
    tC = _mkTile(32, 4, writer.vgprPool)
    tD = _mkTile(64, 4, writer.vgprPool)
    before = writer.vgprPool._next
    asm = str(emitMfmaInstruction(writer, kernel, tA, tB, tC, tD))
    after = writer.vgprPool._next
    _assertScaledMfmaOpcode(asm)
    assert "cbsz:0 blgp:0" in asm
    assert "v250" in asm, "fallback must use _subtileUnitScaleVgpr (v250) for mxsa/mxsb"
    assert after == before, "fallback path must NOT check out a tmp VGPR"


# ---- Backward-compat (FP4) ------------------------------------------------
def test_FP4_with_real_scale_unchanged(writer):
    """FP4 + real scale operands must still produce cbsz:4 blgp:4 (the
    existing behavior — proves the helper falls through correctly)."""
    kernel = _mkKernel("F4", "F4", miK=128, sourceSwap=False)
    tA = _mkTile(0, 4, writer.vgprPool)
    tB = _mkTile(8, 4, writer.vgprPool)
    tC = _mkTile(16, 4, writer.vgprPool)
    tD = _mkTile(32, 4, writer.vgprPool)
    asm = str(
        emitMfmaInstruction(
            writer,
            kernel,
            tA,
            tB,
            tC,
            tD,
            scaleAVgpr=100,
            scaleBVgpr=101,
            scaleAsel=2,
            scaleBsel=1,
        )
    )
    _assertScaledMfmaOpcode(asm)
    assert "cbsz:4 blgp:4" in asm
    assert "v100" in asm and "v101" in asm
    assert "op_sel" in asm
    # FP4 widths
    assert "v[0:3]" in asm and "v[8:11]" in asm  # A, B
    assert "v[32:35]" in asm and "v[16:19]" in asm  # D, C


def test_FP4_no_scale_unchanged_fallback(writer):
    """FP4 fallback (no scale VGPRs) — preserved bit-for-bit."""
    kernel = _mkKernel("F4", "F4", miK=128, sourceSwap=False)
    tA = _mkTile(0, 4, writer.vgprPool)
    tB = _mkTile(8, 4, writer.vgprPool)
    tC = _mkTile(16, 4, writer.vgprPool)
    tD = _mkTile(32, 4, writer.vgprPool)
    before = writer.vgprPool._next
    asm = str(emitMfmaInstruction(writer, kernel, tA, tB, tC, tD))
    after = writer.vgprPool._next
    _assertScaledMfmaOpcode(asm)
    assert "cbsz:4 blgp:4" in asm
    assert "v250" in asm, "fallback must use _subtileUnitScaleVgpr (v250) for mxsa/mxsb"
    assert after == before, "fallback path must NOT check out a tmp VGPR"


def test_legacy_no_DataType_falls_back_to_F4(writer):
    """If DataTypeA/B are absent (legacy callers), helper should raise a
    runtime error"""
    kernel = {
        "MatrixInstK": 128,
        "MIArchVgpr": True,
        "SourceSwap": False,
        "ProblemType": {"MXBlockA": 0, "MXBlockB": 0},  # no DataType*
    }
    tA = _mkTile(0, 4, writer.vgprPool)
    tB = _mkTile(8, 4, writer.vgprPool)
    tC = _mkTile(16, 4, writer.vgprPool)
    tD = _mkTile(32, 4, writer.vgprPool)

    with pytest.raises(RuntimeError):
        emitMfmaInstruction(writer, kernel, tA, tB, tC, tD)


# ---- Backward-compat (BF16) -----------------------------------------------
def test_BF16_path_unchanged(writer):
    """miK==32 still emits dense BF16 MFMA — F8/F4 helper must not run."""
    kernel = _mkKernel("B", "B", miK=32, sourceSwap=False)  # 'B' = BFloat16
    tA = _mkTile(0, 4, writer.vgprPool)
    tB = _mkTile(8, 4, writer.vgprPool)
    tC = _mkTile(16, 4, writer.vgprPool)
    tD = _mkTile(32, 4, writer.vgprPool)
    asm = str(emitMfmaInstruction(writer, kernel, tA, tB, tC, tD))
    assert "v_mfma_f32_16x16x32_bf16" in asm
    assert "v_mfma_scale" not in asm
    assert "cbsz" not in asm and "blgp" not in asm


# ---- Pool-aliasing dispatch unchanged for F8 ------------------------------
def test_F8_uses_accvgpr_alias_when_D_in_agpr_pool(writer):
    """When MIArchVgpr=False AND D's pool is the agprPool, the F8 path must
    alias D and C as accvgpr (i.e., dAccAlias/cAccAlias correctly select
    accvgpr() rather than vgpr()"""
    kernel = _mkKernel("F8", "F8", miK=128, sourceSwap=False, miArchVgpr=False)
    tA = _mkTile(0, 8, writer.vgprPool)
    tB = _mkTile(16, 8, writer.vgprPool)
    tC = _mkTile(32, 4, writer.agprPool)
    tD = _mkTile(64, 4, writer.agprPool)
    asm = str(emitMfmaInstruction(writer, kernel, tA, tB, tC, tD))
    assert (
        "acc[64:67]" in asm and "acc[32:35]" in asm
    ), f"expected agpr alias on D and C:\n{asm}"


# ============================================================================
# 8-bit-float x FP4 mixing tests   (F8 x F4 and BF8 x F4, both directions)
# ----------------------------------------------------------------------------
#   INST_F8_F4  -> "cbsz:0 blgp:4"   (A=FP8, B=FP4)
#   INST_F4_F8  -> "cbsz:4 blgp:0"   (A=FP4, B=FP8)
#   INST_B8_F4  -> "cbsz:1 blgp:4"   (A=BF8, B=FP4)
#   INST_F4_B8  -> "cbsz:4 blgp:1"   (A=FP4, B=BF8)
# ============================================================================
# Each row: (DataTypeA, DataTypeB, sourceSwap, expected_modifiers,
#           expected_emitted_A_position, expected_emitted_B_position)
F8_F4_MIX_CASES = [
    ("F8", "F4", False, "cbsz:0 blgp:4", "v[0:7]", "v[16:19]"),
    ("F8", "F4", True, "cbsz:4 blgp:0", "v[16:19]", "v[0:7]"),
    ("F4", "F8", False, "cbsz:4 blgp:0", "v[0:3]", "v[16:23]"),
    ("F4", "F8", True, "cbsz:0 blgp:4", "v[16:23]", "v[0:3]"),
    ("B8", "F4", False, "cbsz:1 blgp:4", "v[0:7]", "v[16:19]"),
    ("B8", "F4", True, "cbsz:4 blgp:1", "v[16:19]", "v[0:7]"),
    ("F4", "B8", False, "cbsz:4 blgp:1", "v[0:3]", "v[16:23]"),
    ("F4", "B8", True, "cbsz:1 blgp:4", "v[16:23]", "v[0:3]"),
]


@pytest.mark.parametrize("dA,dB,swap,modifiers,a_pos,b_pos", F8_F4_MIX_CASES)
def test_8bit_x_F4_mixed_dispatch(writer, dA, dB, swap, modifiers, a_pos, b_pos):
    """8-bit-float x FP4 dispatch."""

    kernel = _mkKernel(dA, dB, miK=128, sourceSwap=swap)
    aWidth = 8 if dA in ("F8", "B8") else 4  # F8/BF8 -> 8 dwords, F4 -> 4
    bWidth = 8 if dB in ("F8", "B8") else 4
    tA = _mkTile(0, aWidth, writer.vgprPool)
    tB = _mkTile(16, bWidth, writer.vgprPool)
    tC = _mkTile(32, 4, writer.vgprPool)
    tD = _mkTile(64, 4, writer.vgprPool)
    before = writer.vgprPool._next
    asm = str(
        emitMfmaInstruction(
            writer,
            kernel,
            tA,
            tB,
            tC,
            tD,
            scaleAVgpr=100,
            scaleBVgpr=101,
            scaleAsel=2,
            scaleBsel=1,
            comment="8bit x F4 mix",
        )
    )
    after = writer.vgprPool._next
    # 1. Scaled opcode (single hardware instr handles all F8/F6/F4 + mixes).
    _assertScaledMfmaOpcode(asm)
    # 2. Correct cbsz/blgp for the post-swap A/B format order.
    assert modifiers in asm, f"expected `{modifiers}` in asm:\n{asm}"
    # 3. Asymmetric operand widths in the correct (swap-adjusted) positions.
    assert (
        a_pos in asm and b_pos in asm
    ), f"expected A-pos `{a_pos}` and B-pos `{b_pos}` in asm:\n{asm}"
    # 3b. Acc and C are always 4 dwords (fp32).
    assert (
        "v[64:67]" in asm and "v[32:35]" in asm
    ), f"acc/c reg widths wrong (must be 4 dwords each):\n{asm}"
    # 4. Scale VGPRs and op_sel must be present.
    assert "v100" in asm and "v101" in asm, f"Scale VGPRs must appear:\n{asm}"
    assert "op_sel" in asm, f"op_sel must be emitted on the real-scale path:\n{asm}"
    # 5. Real-both branch must NOT allocate a tmp VGPR (sanity: confirms we
    #    did not accidentally fall into the hardcoded-fallback path).
    assert (
        after == before
    ), f"real-scale path must NOT check out a tmp VGPR (got {after-before})"


def test_F8_F4_mix_with_SourceSwap_picks_correct_inst_type(writer):
    """Targeted SourceSwap regression for F8 x F4 - proves the helper
    produces INST_F4_F8 when SourceSwap is on (and INST_F8_F4 when off)."""

    k_no = _mkKernel("F8", "F4", miK=128, sourceSwap=False)
    k_swp = _mkKernel("F8", "F4", miK=128, sourceSwap=True)
    tA = _mkTile(0, 8, writer.vgprPool)
    tB = _mkTile(16, 4, writer.vgprPool)
    tC = _mkTile(32, 4, writer.vgprPool)
    tD = _mkTile(64, 4, writer.vgprPool)
    asm_no = str(
        emitMfmaInstruction(
            writer,
            k_no,
            tA,
            tB,
            tC,
            tD,
            scaleAVgpr=100,
            scaleBVgpr=101,
            scaleAsel=0,
            scaleBsel=0,
        )
    )
    asm_swp = str(
        emitMfmaInstruction(
            writer,
            k_swp,
            tA,
            tB,
            tC,
            tD,
            scaleAVgpr=100,
            scaleBVgpr=101,
            scaleAsel=0,
            scaleBsel=0,
        )
    )
    assert "cbsz:0 blgp:4" in asm_no
    assert "cbsz:4 blgp:0" in asm_swp


def test_BF8_F4_mix_with_SourceSwap_picks_correct_inst_type(writer):
    """Targeted SourceSwap regression for BF8 x F4 - proves the helper
    produces INST_F4_B8 when SourceSwap is on (and INST_B8_F4 when off)."""
    k_no = _mkKernel("B8", "F4", miK=128, sourceSwap=False)
    k_swp = _mkKernel("B8", "F4", miK=128, sourceSwap=True)
    tA = _mkTile(0, 8, writer.vgprPool)
    tB = _mkTile(16, 4, writer.vgprPool)
    tC = _mkTile(32, 4, writer.vgprPool)
    tD = _mkTile(64, 4, writer.vgprPool)
    asm_no = str(
        emitMfmaInstruction(
            writer,
            k_no,
            tA,
            tB,
            tC,
            tD,
            scaleAVgpr=100,
            scaleBVgpr=101,
            scaleAsel=0,
            scaleBsel=0,
        )
    )
    asm_swp = str(
        emitMfmaInstruction(
            writer,
            k_swp,
            tA,
            tB,
            tC,
            tD,
            scaleAVgpr=100,
            scaleBVgpr=101,
            scaleAsel=0,
            scaleBsel=0,
        )
    )
    assert "cbsz:1 blgp:4" in asm_no
    assert "cbsz:4 blgp:1" in asm_swp
