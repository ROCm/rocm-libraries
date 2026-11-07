# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for generateSrcStrForMFMA register string generation.

Tests the critical bk remapping logic when UseDirect32XEmulation is enabled.
With emulation and bk%8 < 4, the function must remap bk to:
    adjustedBk = (bk // 8) * 4 + (bk % 4)

This remapping affects MFMA register operand strings. If broken, kernels
compile with silently wrong operands, causing numerical correctness bugs.
"""

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

pytestmark = pytest.mark.unit


def _make_states():
    """Create mock states with required tensor attributes."""
    state_a = SimpleNamespace(
        useDirect32XEmulationThis=False,
        useDirect32XEmulationNext=False,
        numVgprG2LAllocated=0,
    )
    state_b = SimpleNamespace(
        useDirect32XEmulationThis=False,
        useDirect32XEmulationNext=False,
        numVgprG2LAllocated=0,
    )
    return SimpleNamespace(
        a=state_a,
        b=state_b,
        mxsa=state_a,  # Reuse for simplicity
        mxsb=state_b,
        numIterPerCoalescedReadA=1,
        numReadsIterCoalescedA=1,
        numIterPerCoalescedReadB=1,
        numReadsIterCoalescedB=1,
        numIterPerCoalescedReadMXSA=1,
        numReadsIterCoalescedMXSA=1,
        numIterPerCoalescedReadMXSB=1,
        numReadsIterCoalescedMXSB=1,
        lrvwTileA=1,
        lrvwTileB=1,
        packDTVA=False,
        packDTVB=False,
        convDTVA=False,
        convDTVB=False,
        bpr=4,
    )


def _make_kernel(use_emulation=True, direct_to_vgpr_a=False, half_plr_a=False):
    """Create mock kernel dict."""
    return {
        "UseDirect32XEmulation": use_emulation,
        "DirectToVgprA": direct_to_vgpr_a,
        "DirectToVgprB": False,
        "HalfPLRA": half_plr_a,
        "HalfPLRB": False,
        "MIWaveTileA": 1,
        "MIWaveTileB": 1,
        "MIInputPerThreadA": 1,
        "MIInputPerThreadB": 1,
        "LoopIters": 1,
    }


def _make_tP(tensor_char="A"):
    """Create mock tensor parameters dict."""
    return {
        "tensorChar": tensor_char,
        "isA": tensor_char == "A",
        "isB": tensor_char == "B",
        "bpe": 4,
    }


def _make_writer():
    """Create mock KernelWriterAssembly with required methods."""
    writer = MagicMock()
    writer.states = _make_states()

    # Mock rapResidentBufferIdx to return input m unchanged
    writer.rapResidentBufferIdx = lambda kernel, tc, u, m: m

    # Mock getHalfPLRGroups and getHalfPLRValuStr (not used in our tests)
    writer.getHalfPLRGroups = lambda kernel, lc, u: []
    writer.getHalfPLRValuStr = lambda groups, idx, tc: f"HalfPLR_{tc}"

    return writer


# ---------------------------------------------------------------------------
# UseDirect32XEmulation bk remapping tests
# ---------------------------------------------------------------------------

def test_bk_remap_emulation_bk0():
    """With emulation enabled, bk=0 → adjustedBk = (0//8)*4 + (0%4) = 0."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer()
    kernel = _make_kernel(use_emulation=True)
    tP = _make_tP("A")

    result = KernelWriterAssembly.generateSrcStrForMFMA(
        writer, kernel, tP,
        innerUnroll=1, vregSetIdx=0, vgprPerInput=1,
        m=0, u=0, iui=0, idxAB=0, lc=0, bk=0, shiftK=False
    )

    # Expected: "ValuA_X0_I0+0+0+0+0" (from line 9540, 9558, 9572)
    # bk=0: (0 % 8) = 0 < 4 → adjustedBk = 0
    expected = "ValuA_X0_I0+0+0+0+0"
    assert result == expected, f"Expected complete operand '{expected}', got: {result}"


def test_bk_remap_emulation_bk3():
    """With emulation enabled, bk=3 → adjustedBk = (3//8)*4 + (3%4) = 3."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer()
    kernel = _make_kernel(use_emulation=True)
    tP = _make_tP("A")

    result = KernelWriterAssembly.generateSrcStrForMFMA(
        writer, kernel, tP,
        innerUnroll=1, vregSetIdx=0, vgprPerInput=1,
        m=0, u=0, iui=0, idxAB=0, lc=0, bk=3, shiftK=False
    )

    # bk=3: (3 % 8) = 3 < 4 → adjustedBk = 0 + 3 = 3
    expected = "ValuA_X0_I0+0+0+0+3"
    assert result == expected, f"Expected complete operand '{expected}', got: {result}"


def test_bk_remap_emulation_bk4_no_remap():
    """With emulation enabled, bk=4 → no remap (4%8 = 4, NOT < 4)."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer()
    kernel = _make_kernel(use_emulation=True)
    tP = _make_tP("A")

    result = KernelWriterAssembly.generateSrcStrForMFMA(
        writer, kernel, tP,
        innerUnroll=1, vregSetIdx=0, vgprPerInput=1,
        m=0, u=0, iui=0, idxAB=0, lc=0, bk=4, shiftK=False
    )

    # bk=4: (4 % 8) = 4 NOT < 4 → use bk unchanged
    expected = "ValuA_X0_I0+0+0+0+4"
    assert result == expected, f"Expected complete operand '{expected}', got: {result}"


def test_bk_remap_emulation_bk7_no_remap():
    """With emulation enabled, bk=7 → no remap (7%8 = 7, NOT < 4)."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer()
    kernel = _make_kernel(use_emulation=True)
    tP = _make_tP("A")

    result = KernelWriterAssembly.generateSrcStrForMFMA(
        writer, kernel, tP,
        innerUnroll=1, vregSetIdx=0, vgprPerInput=1,
        m=0, u=0, iui=0, idxAB=0, lc=0, bk=7, shiftK=False
    )

    # bk=7: (7 % 8) = 7 NOT < 4 → use bk unchanged
    expected = "ValuA_X0_I0+0+0+0+7"
    assert result == expected, f"Expected complete operand '{expected}', got: {result}"


def test_bk_remap_emulation_bk8():
    """With emulation enabled, bk=8 → adjustedBk = (8//8)*4 + (8%4) = 4."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer()
    kernel = _make_kernel(use_emulation=True)
    tP = _make_tP("A")

    result = KernelWriterAssembly.generateSrcStrForMFMA(
        writer, kernel, tP,
        innerUnroll=1, vregSetIdx=0, vgprPerInput=1,
        m=0, u=0, iui=0, idxAB=0, lc=0, bk=8, shiftK=False
    )

    # bk=8: (8 % 8) = 0 < 4 → adjustedBk = 1*4 + 0 = 4
    expected = "ValuA_X0_I0+0+0+0+4"
    assert result == expected, f"Expected complete operand '{expected}', got: {result}"


def test_bk_remap_emulation_bk11():
    """With emulation enabled, bk=11 → adjustedBk = (11//8)*4 + (11%4) = 7."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer()
    kernel = _make_kernel(use_emulation=True)
    tP = _make_tP("A")

    result = KernelWriterAssembly.generateSrcStrForMFMA(
        writer, kernel, tP,
        innerUnroll=1, vregSetIdx=0, vgprPerInput=1,
        m=0, u=0, iui=0, idxAB=0, lc=0, bk=11, shiftK=False
    )

    # bk=11: (11 % 8) = 3 < 4 → adjustedBk = 1*4 + 3 = 7
    expected = "ValuA_X0_I0+0+0+0+7"
    assert result == expected, f"Expected complete operand '{expected}', got: {result}"


def test_bk_remap_emulation_bk16():
    """With emulation enabled, bk=16 → adjustedBk = (16//8)*4 + (16%4) = 8."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer()
    kernel = _make_kernel(use_emulation=True)
    tP = _make_tP("A")

    result = KernelWriterAssembly.generateSrcStrForMFMA(
        writer, kernel, tP,
        innerUnroll=1, vregSetIdx=0, vgprPerInput=1,
        m=0, u=0, iui=0, idxAB=0, lc=0, bk=16, shiftK=False
    )

    # bk=16: (16 % 8) = 0 < 4 → adjustedBk = 2*4 + 0 = 8
    expected = "ValuA_X0_I0+0+0+0+8"
    assert result == expected, f"Expected complete operand '{expected}', got: {result}"


# ---------------------------------------------------------------------------
# No emulation baseline tests
# ---------------------------------------------------------------------------

def test_bk_no_emulation_bk0():
    """With emulation disabled, bk=0 appends unchanged."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer()
    kernel = _make_kernel(use_emulation=False)
    tP = _make_tP("A")

    result = KernelWriterAssembly.generateSrcStrForMFMA(
        writer, kernel, tP,
        innerUnroll=1, vregSetIdx=0, vgprPerInput=1,
        m=0, u=0, iui=0, idxAB=0, lc=0, bk=0, shiftK=False
    )

    expected = "ValuA_X0_I0+0+0+0+0"
    assert result == expected, f"Expected complete operand '{expected}', got: {result}"


def test_bk_no_emulation_bk8():
    """With emulation disabled, bk=8 appends unchanged (no remap to 4)."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer()
    kernel = _make_kernel(use_emulation=False)
    tP = _make_tP("A")

    result = KernelWriterAssembly.generateSrcStrForMFMA(
        writer, kernel, tP,
        innerUnroll=1, vregSetIdx=0, vgprPerInput=1,
        m=0, u=0, iui=0, idxAB=0, lc=0, bk=8, shiftK=False
    )

    # Without emulation, bk=8 stays +8 (not remapped to +4)
    expected = "ValuA_X0_I0+0+0+0+8"
    assert result == expected, f"Expected complete operand '{expected}', got: {result}"


def test_bk_no_emulation_bk11():
    """With emulation disabled, bk=11 appends unchanged (no remap to 7)."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer()
    kernel = _make_kernel(use_emulation=False)
    tP = _make_tP("A")

    result = KernelWriterAssembly.generateSrcStrForMFMA(
        writer, kernel, tP,
        innerUnroll=1, vregSetIdx=0, vgprPerInput=1,
        m=0, u=0, iui=0, idxAB=0, lc=0, bk=11, shiftK=False
    )

    # Without emulation, bk=11 stays +11 (not remapped to +7)
    expected = "ValuA_X0_I0+0+0+0+11"
    assert result == expected, f"Expected complete operand '{expected}', got: {result}"


# ---------------------------------------------------------------------------
# Tensor B tests (verify logic applies to both A and B)
# ---------------------------------------------------------------------------

def test_bk_remap_emulation_tensor_b_bk8():
    """Verify bk remapping works for tensor B, bk=8 → +4."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer()
    kernel = _make_kernel(use_emulation=True)
    tP = _make_tP("B")

    result = KernelWriterAssembly.generateSrcStrForMFMA(
        writer, kernel, tP,
        innerUnroll=1, vregSetIdx=0, vgprPerInput=1,
        m=0, u=0, iui=0, idxAB=0, lc=0, bk=8, shiftK=False
    )

    # Tensor B should also remap bk=8 → +4
    expected = "ValuB_X0_I0+0+0+0+4"
    assert result == expected, f"Expected complete operand '{expected}', got: {result}"


def test_bk_remap_emulation_tensor_b_bk11():
    """Verify bk remapping works for tensor B, bk=11 → +7."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer()
    kernel = _make_kernel(use_emulation=True)
    tP = _make_tP("B")

    result = KernelWriterAssembly.generateSrcStrForMFMA(
        writer, kernel, tP,
        innerUnroll=1, vregSetIdx=0, vgprPerInput=1,
        m=0, u=0, iui=0, idxAB=0, lc=0, bk=11, shiftK=False
    )

    # Tensor B should also remap bk=11 → +7
    expected = "ValuB_X0_I0+0+0+0+7"
    assert result == expected, f"Expected complete operand '{expected}', got: {result}"


# ---------------------------------------------------------------------------
# Edge case: bk=None (no bk parameter)
# ---------------------------------------------------------------------------

def test_bk_none_no_offset():
    """When bk=None, no offset is appended."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer()
    kernel = _make_kernel(use_emulation=True)
    tP = _make_tP("A")

    result = KernelWriterAssembly.generateSrcStrForMFMA(
        writer, kernel, tP,
        innerUnroll=1, vregSetIdx=0, vgprPerInput=1,
        m=0, u=0, iui=0, idxAB=0, lc=0, bk=None, shiftK=False
    )

    # With bk=None, the complete operand should be "ValuA_X0_I0+0+0+0" (no bk offset)
    # Verify the exact format to detect unwanted appended offsets like "+0+8" or "+0+4"
    expected = "ValuA_X0_I0+0+0+0"
    assert result == expected, (
        f"Expected complete operand '{expected}' when bk=None, got: {result}"
    )


# ---------------------------------------------------------------------------
# Formula validation tests
# ---------------------------------------------------------------------------

def test_bk_remap_formula_coverage():
    """Systematic test of remapping formula across multiple bk values."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer()
    kernel = _make_kernel(use_emulation=True)
    tP = _make_tP("A")

    # Test cases: (bk_input, expected_offset)
    test_cases = [
        (0, 0),    # (0//8)*4 + (0%4) = 0
        (1, 1),    # (1//8)*4 + (1%4) = 1
        (2, 2),    # (2//8)*4 + (2%4) = 2
        (3, 3),    # (3//8)*4 + (3%4) = 3
        (4, 4),    # 4%8 >= 4, no remap
        (5, 5),    # 5%8 >= 4, no remap
        (6, 6),    # 6%8 >= 4, no remap
        (7, 7),    # 7%8 >= 4, no remap
        (8, 4),    # (8//8)*4 + (8%4) = 4
        (9, 5),    # (9//8)*4 + (9%4) = 5
        (10, 6),   # (10//8)*4 + (10%4) = 6
        (11, 7),   # (11//8)*4 + (11%4) = 7
        (12, 12),  # 12%8 >= 4, no remap
        (16, 8),   # (16//8)*4 + (16%4) = 8
        (19, 11),  # (19//8)*4 + (19%4) = 8+3 = 11
    ]

    for bk_input, expected_offset in test_cases:
        result = KernelWriterAssembly.generateSrcStrForMFMA(
            writer, kernel, tP,
            innerUnroll=1, vregSetIdx=0, vgprPerInput=1,
            m=0, u=0, iui=0, idxAB=0, lc=0, bk=bk_input, shiftK=False
        )

        # Compare complete operand to avoid substring matching issues
        # (e.g., "+40" would match expected "+4", base "+0" could hide incorrect bk offset)
        expected = f"ValuA_X0_I0+0+0+0+{expected_offset}"
        assert result == expected, \
            f"bk={bk_input}: expected complete operand '{expected}', got: {result}"
