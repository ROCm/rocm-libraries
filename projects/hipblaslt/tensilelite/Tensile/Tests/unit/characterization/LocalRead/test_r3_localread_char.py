################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""P4 characterization: LocalReadMFMA helper methods + CheckValue1 path.

Directly exercises the pure-Python helper methods on LocalReadMFMA that live
on the missing-coverage ranges in Tensile/Components/LocalRead.py:

  117-151  LocalReadVALU CheckValue1 debug path (lines 117-118 reachable;
           119-151 are dead code due to RegisterContainer.split() bug)
  217-225  LocalReadMFMA.TXInterleaveLayoutIdx (full body)
  265-299  LocalReadMFMA.getTransposeXorTIndex: TF32EmuInterleaveTreg and
           doFullPackCodePrefetch branches (all arms)
  362-365  LocalReadMFMA.getVgprStrForEmuMfma ForceUnrollSubIter A/B branches

All tests are CPU-only. No GPU. No kernel emit overhead. The methods are
invoked through minimal SimpleNamespace mock objects (writer.states.a, etc.)
that carry exactly the fields LocalReadMFMA reads at the call sites.

Pattern: A (codegen-component method calls, pure Python).
"""

import pytest
from types import SimpleNamespace

import rocisa  # noqa: F401 — must import before Tensile.Component to avoid SIGABRT
import Tensile.Component  # noqa: F401 — ensures Component.py is loaded before LocalRead.py
from Tensile.Components.LocalRead import LocalReadMFMA

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _abinfo(
    TF32EmuInterleaveTreg=False,
    useDirect32XEmulationThis=False,
    useDirect32XEmulationNext=False,
    useTransposeCodeThis=False,
    useTransposeCodeNext=False,
    swapBlockSizeSub=16,
):
    """Minimal ABMatrixInfo-like namespace for LocalReadMFMA method calls."""
    return SimpleNamespace(
        TF32EmuInterleaveTreg=TF32EmuInterleaveTreg,
        useDirect32XEmulationThis=useDirect32XEmulationThis,
        useDirect32XEmulationNext=useDirect32XEmulationNext,
        useTransposeCodeThis=useTransposeCodeThis,
        useTransposeCodeNext=useTransposeCodeNext,
        swapBlockSizeSub=swapBlockSizeSub,
    )


def _writer(
    abinfo_a=None,
    abinfo_b=None,
    doFullPackCodePrefetch=False,
    numItersPLR=1,
    numIterPerCoalescedReadA=1,
    numIterPerCoalescedReadB=1,
    SubTileIdx=0,
):
    """Minimal writer mock with states namespace."""
    if abinfo_a is None:
        abinfo_a = _abinfo()
    if abinfo_b is None:
        abinfo_b = abinfo_a
    states = SimpleNamespace(
        a=abinfo_a,
        b=abinfo_b,
        doFullPackCodePrefetch=doFullPackCodePrefetch,
        numItersPLR=numItersPLR,
        numIterPerCoalescedReadA=numIterPerCoalescedReadA,
        numIterPerCoalescedReadB=numIterPerCoalescedReadB,
        SubTileIdx=SubTileIdx,
    )
    return SimpleNamespace(states=states)


_LRMFMA = LocalReadMFMA()

_KERNEL_BASE = {"MIInputPerThread": 8, "ForceUnrollSubIter": False}
_KERNEL_SUBITER = {"MIInputPerThread": 8, "ForceUnrollSubIter": True}

# ---------------------------------------------------------------------------
# Lines 217-225 — TXInterleaveLayoutIdx
# ---------------------------------------------------------------------------

class TestTXInterleaveLayoutIdx:
    """Covers lines 217-225 of LocalRead.py (TXInterleaveLayoutIdx method body).

    The method partitions the 0..miInputPerThread range into two halves.
    Lower half (idx%M < M/2) maps to 'T' registers; upper half stays 'X'.
    """

    def test_lower_half_returns_T(self):
        """idx in lower half (idx%8 < 4) -> XTchar='T' (lines 220-224)."""
        # idx=0: 0%8=0 < 4 -> T
        retIdx, xchar = _LRMFMA.TXInterleaveLayoutIdx(0, miInputPerThread=8)
        assert xchar == "T"
        assert retIdx == 0  # (0//8)*4 + 0%4 = 0

    def test_upper_half_returns_X(self):
        """idx in upper half (idx%8 >= 4) -> XTchar='X' (lines 218-219, falls through)."""
        # idx=4: 4%8=4 not < 4 -> X
        retIdx, xchar = _LRMFMA.TXInterleaveLayoutIdx(4, miInputPerThread=8)
        assert xchar == "X"
        assert retIdx == 4  # unchanged

    def test_retIdx_lower_half_arithmetic(self):
        """T-branch index arithmetic: retIdx = (idx//M)*halfM + idx%halfM."""
        # idx=2, M=8: 2%8=2 < 4 -> T; retIdx = (2//8)*4 + 2%4 = 0+2 = 2
        retIdx, xchar = _LRMFMA.TXInterleaveLayoutIdx(2, miInputPerThread=8)
        assert xchar == "T"
        assert retIdx == 2

    def test_second_group_lower_half(self):
        """idx=9 with M=8: 9%8=1 < 4 -> T; retIdx=(9//8)*4 + 1%4 = 4+1 = 5."""
        retIdx, xchar = _LRMFMA.TXInterleaveLayoutIdx(9, miInputPerThread=8)
        assert xchar == "T"
        assert retIdx == 5

    def test_second_group_upper_half(self):
        """idx=12 with M=8: 12%8=4 not < 4 -> X, retIdx=12."""
        retIdx, xchar = _LRMFMA.TXInterleaveLayoutIdx(12, miInputPerThread=8)
        assert xchar == "X"
        assert retIdx == 12

    def test_boundary_exactly_at_halfM(self):
        """idx=3, M=8: 3%8=3 < 4 -> T (boundary below halfM)."""
        retIdx, xchar = _LRMFMA.TXInterleaveLayoutIdx(3, miInputPerThread=8)
        assert xchar == "T"

    def test_boundary_at_halfM_upper(self):
        """idx=4, M=8: 4%8=4 == halfM=4, NOT < halfM -> X."""
        _, xchar = _LRMFMA.TXInterleaveLayoutIdx(4, miInputPerThread=8)
        assert xchar == "X"


# ---------------------------------------------------------------------------
# Lines 265-299 — getTransposeXorTIndex branches
# ---------------------------------------------------------------------------

class TestGetTransposeXorTIndexLocalRead:
    """Covers localRead=True branches (lines 256-277).

    Three sub-cases:
    A. TF32EmuInterleaveTreg=True, doFullPackCodePrefetch=True -> TXInterleaveLayoutIdx (line 265)
    B. TF32EmuInterleaveTreg=True, doFullPackCodePrefetch=False -> withinGroup branch (267-270)
    C. TF32EmuInterleaveTreg=False, doFullPackCodePrefetch=True -> swapBlockSizeSub (271-276)
    """

    def test_localread_TF32_DFPCP_calls_TXInterleave(self):
        """localRead+TF32+DFPCP -> delegates to TXInterleaveLayoutIdx (line 265)."""
        ab = _abinfo(TF32EmuInterleaveTreg=True)
        w = _writer(abinfo_a=ab, doFullPackCodePrefetch=True)
        # idx=0, M=8: TXInterleave(0,8) = (0, 'T')
        idx, xchar = _LRMFMA.getTransposeXorTIndex(w, _KERNEL_BASE, 0, "A", 1, False, False, localRead=True)
        assert xchar == "T"
        assert idx == 0

    def test_localread_TF32_no_DFPCP_within_lower_half(self):
        """localRead+TF32+no DFPCP, idx%8<4 -> XTchar='T' (lines 267-270)."""
        ab = _abinfo(TF32EmuInterleaveTreg=True)
        w = _writer(abinfo_a=ab, doFullPackCodePrefetch=False)
        # idx=2: withinGroup=2 < 4 -> T, idx = (2//8)*4+2 = 2
        idx, xchar = _LRMFMA.getTransposeXorTIndex(w, _KERNEL_BASE, 2, "A", 1, False, False, localRead=True)
        assert xchar == "T"

    def test_localread_TF32_no_DFPCP_upper_half(self):
        """localRead+TF32+no DFPCP, idx%8>=4 -> XTchar='X' (falls through 267-270)."""
        ab = _abinfo(TF32EmuInterleaveTreg=True)
        w = _writer(abinfo_a=ab, doFullPackCodePrefetch=False)
        # idx=5: withinGroup=5 not < 4 -> stays X
        idx, xchar = _LRMFMA.getTransposeXorTIndex(w, _KERNEL_BASE, 5, "A", 1, False, False, localRead=True)
        assert xchar == "X"

    def test_localread_no_TF32_with_DFPCP(self):
        """localRead+no TF32+DFPCP -> T + swapBlockSizeSub mod (lines 274-276)."""
        ab = _abinfo(TF32EmuInterleaveTreg=False, swapBlockSizeSub=8)
        w = _writer(abinfo_a=ab, doFullPackCodePrefetch=True)
        # idx=5, swapBlockSizeSub=8: idx = 5%8 = 5; xchar='T'
        idx, xchar = _LRMFMA.getTransposeXorTIndex(w, _KERNEL_BASE, 5, "A", 1, False, False, localRead=True)
        assert xchar == "T"
        assert idx == 5 % 8

    def test_localread_no_TF32_with_DFPCP_zero_mod(self):
        """localRead+no TF32+DFPCP: idx=8, swapBlockSizeSub=8 -> idx=0."""
        ab = _abinfo(TF32EmuInterleaveTreg=False, swapBlockSizeSub=8)
        w = _writer(abinfo_a=ab, doFullPackCodePrefetch=True)
        idx, xchar = _LRMFMA.getTransposeXorTIndex(w, _KERNEL_BASE, 8, "A", 1, False, False, localRead=True)
        assert xchar == "T"
        assert idx == 0


class TestGetTransposeXorTIndexSrcCase:
    """Covers src path (not dst, not localRead) branches (lines 278-303).

    Sub-cases (not localRead, dst=False):
    D. useDirect32XEmulation=False -> pass (lines 280-283)
    E. useDirect32XEmulation=True, TF32EmuInterleaveTreg=True, DFPCP=True -> TXInterleave (line 286)
    F. useDirect32XEmulation=True, TF32EmuInterleaveTreg=True, DFPCP=False -> withinGroup (287-291)
    G. useDirect32XEmulation=True, TF32EmuInterleaveTreg=False, DFPCP=True -> swapMod+getTranspose (292-299)
    """

    def test_src_no_direct32x_returns_unchanged(self):
        """src, useDirect32X=False -> pass (lines 280-283), idx/xchar unchanged."""
        ab = _abinfo(TF32EmuInterleaveTreg=False, useDirect32XEmulationThis=False)
        w = _writer(abinfo_a=ab, doFullPackCodePrefetch=False)
        idx, xchar = _LRMFMA.getTransposeXorTIndex(w, _KERNEL_BASE, 7, "A", 1, False, False, localRead=False)
        assert xchar == "X"
        assert idx == 7

    def test_src_TF32_DFPCP_calls_TXInterleave(self):
        """src+TF32+useDirect32X+DFPCP -> TXInterleaveLayoutIdx (line 286)."""
        ab = _abinfo(TF32EmuInterleaveTreg=True, useDirect32XEmulationThis=True)
        w = _writer(abinfo_a=ab, doFullPackCodePrefetch=True)
        # idx=1, M=8: TXInterleave(1,8) = (1%8=1<4 -> T; retIdx=0*4+1=1)
        idx, xchar = _LRMFMA.getTransposeXorTIndex(w, _KERNEL_BASE, 1, "A", 1, False, False, localRead=False)
        assert xchar == "T"
        assert idx == 1

    def test_src_TF32_no_DFPCP_within_lower(self):
        """src+TF32+useDirect32X+no DFPCP, idx%8<4 -> T (lines 288-291)."""
        ab = _abinfo(TF32EmuInterleaveTreg=True, useDirect32XEmulationThis=True)
        w = _writer(abinfo_a=ab, doFullPackCodePrefetch=False)
        idx, xchar = _LRMFMA.getTransposeXorTIndex(w, _KERNEL_BASE, 2, "A", 1, False, False, localRead=False)
        assert xchar == "T"

    def test_src_TF32_no_DFPCP_upper_half(self):
        """src+TF32+useDirect32X+no DFPCP, idx%8>=4 -> stays X."""
        ab = _abinfo(TF32EmuInterleaveTreg=True, useDirect32XEmulationThis=True)
        w = _writer(abinfo_a=ab, doFullPackCodePrefetch=False)
        idx, xchar = _LRMFMA.getTransposeXorTIndex(w, _KERNEL_BASE, 6, "A", 1, False, False, localRead=False)
        assert xchar == "X"

    def test_src_no_TF32_with_DFPCP_lrvwTile1(self):
        """src+no TF32+useDirect32X+DFPCP, lrvwTile=1 -> T+mod, no getTranspose (lines 292-297)."""
        ab = _abinfo(TF32EmuInterleaveTreg=False, useDirect32XEmulationThis=True, swapBlockSizeSub=8)
        w = _writer(abinfo_a=ab, doFullPackCodePrefetch=True)
        idx, xchar = _LRMFMA.getTransposeXorTIndex(w, _KERNEL_BASE, 5, "A", 1, False, False, localRead=False)
        assert xchar == "T"
        assert idx == 5 % 8

    def test_src_no_TF32_with_DFPCP_lrvwTile2_with_transpose(self):
        """src+no TF32+useDirect32X+DFPCP, lrvwTile=2+useTransposeCode=False -> getTranspose (lines 297-299)."""
        ab = _abinfo(
            TF32EmuInterleaveTreg=False,
            useDirect32XEmulationThis=True,
            useDirect32XEmulationNext=True,
            useTransposeCodeThis=False,
            swapBlockSizeSub=16,
        )
        w = _writer(abinfo_a=ab, doFullPackCodePrefetch=True)
        # lrvwTile=2, no transposeCode -> getTransposeIndex called
        # idx=3, mod 16 = 3; then getTransposeIndex(kernel, 3, 2)
        kernel = {"MIInputPerThread": 8, "ForceUnrollSubIter": False}
        idx, xchar = _LRMFMA.getTransposeXorTIndex(w, kernel, 3, "A", 2, False, False, localRead=False)
        assert xchar == "T"
        # The exact idx depends on getTransposeIndex(kernel, 3%16=3, lrvwTile=2)
        # Just assert it ran without error
        assert isinstance(idx, int)


# ---------------------------------------------------------------------------
# Lines 362-365 — getVgprStrForEmuMfma ForceUnrollSubIter branches
# ---------------------------------------------------------------------------

class TestGetVgprStrForEmuMfmaForceUnrollSubIter:
    """Covers lines 362-365 of LocalRead.py (ForceUnrollSubIter A/B branches).

    When ForceUnrollSubIter=True:
      - tc=='A': isNext = (u & 1) == 0  (line 362-363)
      - tc!='A': isNext = u < 2         (line 364-365)
    """

    def test_forcesubiter_A_even_u_isNext(self):
        """tc='A', u=0 (even) -> isNext=True (u & 1 == 0) (lines 362-363)."""
        w = _writer()
        result = _LRMFMA.getVgprStrForEmuMfma(w, _KERNEL_SUBITER, "A", 0, 0, 0, 1, u=0)
        # isNext=True -> bufferIdx used as-is; just check it doesn't crash and returns a string
        assert isinstance(result, str)
        assert "ValuA" in result

    def test_forcesubiter_A_odd_u_not_isNext(self):
        """tc='A', u=1 (odd) -> isNext=False (u & 1 == 1, not 0) (lines 362-363)."""
        w = _writer()
        result = _LRMFMA.getVgprStrForEmuMfma(w, _KERNEL_SUBITER, "A", 0, 0, 0, 1, u=1)
        assert isinstance(result, str)
        assert "ValuA" in result

    def test_forcesubiter_B_u_lt2_isNext(self):
        """tc='B', u=1 (< 2) -> isNext=True (u < 2) (lines 364-365)."""
        w = _writer()
        result = _LRMFMA.getVgprStrForEmuMfma(w, _KERNEL_SUBITER, "B", 0, 0, 0, 1, u=1)
        assert isinstance(result, str)
        assert "ValuB" in result

    def test_forcesubiter_B_u_ge2_not_isNext(self):
        """tc='B', u=2 (>= 2) -> isNext=False (u < 2 is False) (lines 364-365)."""
        w = _writer()
        result = _LRMFMA.getVgprStrForEmuMfma(w, _KERNEL_SUBITER, "B", 0, 0, 0, 1, u=2)
        assert isinstance(result, str)
        assert "ValuB" in result

    def test_forcesubiter_A_u4_even(self):
        """tc='A', u=4 (even) -> isNext=True."""
        w = _writer()
        result = _LRMFMA.getVgprStrForEmuMfma(w, _KERNEL_SUBITER, "A", 0, 0, 0, 1, u=4)
        assert isinstance(result, str)
        assert "ValuA" in result

    def test_forcesubiter_B_u3_not_isNext(self):
        """tc='B', u=3 (>= 2) -> isNext=False."""
        w = _writer()
        result = _LRMFMA.getVgprStrForEmuMfma(w, _KERNEL_SUBITER, "B", 0, 0, 0, 1, u=3)
        assert isinstance(result, str)
        assert "ValuB" in result

    def test_no_forcesubiter_uses_getIsNext(self):
        """ForceUnrollSubIter=False -> uses getIsNext path (line 367), not the ForceUnroll branches."""
        w = _writer(numItersPLR=1, numIterPerCoalescedReadA=1, SubTileIdx=0)
        result = _LRMFMA.getVgprStrForEmuMfma(w, _KERNEL_BASE, "A", 0, 0, 0, 1, u=0)
        assert isinstance(result, str)
        assert "ValuA" in result


# ---------------------------------------------------------------------------
# Lines 117-118 — LocalReadVALU CheckValue1 debug path
# ---------------------------------------------------------------------------

class TestLocalReadVALUCheckValue1:
    """Covers lines 117-118 of LocalRead.py (CheckValue1 debug gate).

    Lines 119-151 are dead code: line 118 calls destVgpr.split() but destVgpr
    is a rocisa.RegisterContainer which lacks .split(), so an AttributeError
    is always raised before reaching lines 119+.

    The test verifies:
    - Lines 116-118 are reached when CheckValue1A=True (checked via error type).
    - Lines 116-118 are NOT reached when CheckValue1A=False (no error).
    """

    def _make_minimal_valu_emit(self, enableDebugA=False):
        """Emit one VALU (EnableMatrixInstruction=False) kernel.

        Uses the lra.yaml designed VALU arm (fp16 HPA, WorkGroup [8,4,2]).
        Returns (basename, err_or_exception).
        """
        import os
        import sys
        sys.path.insert(0, os.path.join(
            os.path.dirname(__file__), "..", "_codegen"
        ))
        from config_harness import _toolchain_for, _isolated_globals_with_isa, _solutions_from_config_unguarded
        from codegen_harness import _init_rocisa_for, _prepare_kernel
        from Tensile.TensileCreateLibrary.Run import generateKernelObjectsFromSolutions, processKernelSource
        from Tensile.KernelWriterAssembly import KernelWriterAssembly
        from Tensile.Common.Types import DebugConfig

        arch = "gfx942"
        assembler, iim = _toolchain_for(arch)
        # lra.yaml has VALU (EnableMatrixInstruction=False) arm with fp16 HPA
        lra_cfg = os.path.join(
            os.path.dirname(__file__), "..", "_codegen", "data", "test_data",
            "_designed", "gfx942", "lra.yaml"
        )

        with _isolated_globals_with_isa(iim):
            sols = _solutions_from_config_unguarded(lra_cfg, assembler, iim, limit_solutions=1)
            if not sols:
                return None, "no_solutions"
            kernels = list(generateKernelObjectsFromSolutions(sols))[:1]
            if not kernels:
                return None, "no_kernels"
            dbg = DebugConfig(enableDebugA=enableDebugA)
            kwa = KernelWriterAssembly(assembler, dbg)
            k = kernels[0]
            ri = _init_rocisa_for(k)
            data = ri.getData()
            outOptions = ri.getOutputOptions()
            base = _prepare_kernel(k)
            try:
                res = processKernelSource(kwa, data, outOptions, False, k)
                return base, res.err
            except AttributeError as exc:
                # AttributeError: 'RegisterContainer' has no attribute 'split'
                # This is the real behavior at line 118 when CheckValue1A=True
                return base, exc

    def test_checkvalue1_debug_false_emits_ok(self):
        """enableDebugA=False: CheckValue1A=False, lines 117-118 NOT reached, kernel emits (err=0)."""
        base, result = self._make_minimal_valu_emit(enableDebugA=False)
        # Without debug, no exception; kernel should emit successfully
        assert base is not None
        assert result == 0, f"Expected err=0 without debug, got {result!r}"

    def test_checkvalue1_debug_true_reaches_line118(self):
        """enableDebugA=True: CheckValue1A=True, lines 117-118 ARE reached (AttributeError on line 118).

        This pins the actual behavior: line 118 calls destVgpr.split('v[') but
        destVgpr is a RegisterContainer (no .split()), so AttributeError is raised.
        The test confirms line 118 runs and raises that exact error.
        """
        base, result = self._make_minimal_valu_emit(enableDebugA=True)
        # With debug enabled, line 116 check is True, line 117 runs,
        # line 118 raises AttributeError (destVgpr is RegisterContainer)
        assert isinstance(result, AttributeError), (
            f"Expected AttributeError at line 118 (debug mode), got {result!r}"
        )
        assert "split" in str(result) or "RegisterContainer" in str(result), (
            f"AttributeError content unexpected: {result}"
        )
