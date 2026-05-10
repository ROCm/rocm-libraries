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
        "ProblemType": {
            "DataTypeA": DataType(dA) if dA else None,
            "DataTypeB": DataType(dB) if dB else None,
            "MXBlockA": 0,
            "MXBlockB": 0,
        },
    }


@pytest.fixture(scope="module", autouse=True)
def _rocisa_once():
    init_rocisa()


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
    """No real scale VGPRs -> fallback writes 0x7f7f7f7f (E8M0 = 1.0 in every
    byte) into a tmp VGPR and uses it for both mxsa/mxsb."""
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
    assert "0x7f7f7f7f" in asm, "fallback must load 0x7f7f7f7f into tmp scale"
    assert after > before, "fallback path must check out one tmp VGPR"


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
    assert "0x7f7f7f7f" in asm
    assert after > before


def test_legacy_no_DataType_falls_back_to_F4(writer):
    """If DataTypeA/B are absent (legacy callers), helper returns None and
    caller defaults to INST_F4 -> cbsz:4 blgp:4. Bit-stable behavior."""
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
    asm = str(emitMfmaInstruction(writer, kernel, tA, tB, tC, tD))
    assert "cbsz:4 blgp:4" in asm  # FP4 default preserved


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


