################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R4 — multi-index summation (tensor contraction) graIncrements characterization.

Exercises the "other summation" arm of
``Tensile/KernelWriterAssembly.py:graIncrements`` (lines 5022-5092) that runs
when ``loopIdx != self.states.unrollIdx`` — i.e. when a kernel has more than
one summation index (NumIndicesSummation > 1).

Background
----------
Standard GEMM always produces exactly one summation index, so the "other
summation" arm is unreachable through the normal config/logic YAML pipeline
(``ProblemType.__init__`` hard-codes ``OperationType == 'GEMM'`` and
``initGEMM`` derives a single summation index regardless of what the YAML
says).  The arm IS reachable by constructing a synthetic kernel dict with
``NumIndicesSummation=2`` and calling ``graIncrements`` directly on a
minimally-initialised ``KernelWriterAssembly``, which is the approach taken
here.  See ``KernelWriterAssembly.py`` line 4993:
  assert(self.states.unrollIdx == kernel["ProblemType"]["NumIndicesSummation"]-1)
With NumIndicesSummation=2 and unrollIdx=1, calling graIncrements(kernel, 0, tP)
triggers the "other summation" branch (loopIdx=0 != unrollIdx=1).

Two sub-paths tested
--------------------
1. **SGPR path (BufferLoad=True, globalReadIncsUseVgpr=False):**
   Lines 5026-5092.  Uses tP_B (``isA=False``) to avoid the ``None``-rReg bug
   in the ``isA=True`` branch's ``scalarStaticDivideAndRemainder`` call at
   line 5047.  The SGPR path emits SMulI32 + SSubI32 + SLShiftLeftB32.

2. **printExit path (BufferLoad=False, globalReadIncsUseVgpr=True):**
   Lines 5023-5025.  With non-buffer-load the code calls ``printExit``
   (``sys.exit(-1)`` → ``SystemExit``).  The test asserts the expected
   SystemExit(-1) is raised, pinning the actual validity-reject behaviour.

CPU-only.  No GPU, no compile.  Uses the rocisa singleton for rocIsa init +
a real ``RegisterPool`` for SGPR allocation (required by ``allocTmpSgpr``).
"""

import shutil

import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_rocisa_gfx942():
    """Initialise the rocisa singleton for gfx942 / wave64.  Idempotent."""
    import rocisa

    ri = rocisa.rocIsa.getInstance()
    asmpath = shutil.which("amdclang++") or "/usr/bin/amdclang++"
    ri.init((9, 4, 2), asmpath)
    ri.setKernel((9, 4, 2), 64)
    return ri


def _build_minimal_kwa(kernel, *, globalReadIncsUseVgpr: bool):
    """Construct a *minimally initialised* KernelWriterAssembly.

    Only the state required by ``graIncrements`` is wired in.  Everything
    else is left at defaults or empty so the test stays focused.
    """
    from unittest.mock import MagicMock

    import rocisa as _rocisa_mod
    from rocisa.enum import RegisterType
    from rocisa.register import RegisterPool

    from Tensile.Common.Types import DebugConfig
    from Tensile.KernelWriter import KernelWriter, StateValues
    from Tensile.KernelWriterAssembly import KernelWriterAssembly, GlobalReadGprRecord

    ri = _rocisa_mod.rocIsa.getInstance()

    # Mock assembler satisfies KernelWriter.__init__ without a real toolchain.
    mock_asm = MagicMock()
    mock_asm.rocm_version = MagicMock()
    mock_asm.rocm_version.major = 6

    kwa = object.__new__(KernelWriterAssembly)
    KernelWriter.__init__(kwa, mock_asm, DebugConfig())
    kwa.globalread_gpr_record = GlobalReadGprRecord()

    # SGPR pool: 80 registers, all available.
    pool = RegisterPool(
        0, RegisterType.Sgpr, defaultPreventOverflow=False, printRP=False
    )
    pool.add(0, 80, "multisum-test")
    kwa.sgprPool = pool
    kwa.sgprs = {}
    kwa.dontAppendCode = False

    # Kernel-level states required by graIncrements.
    version = (9, 4, 2)
    kwa.states = StateValues(version=version, kernel=kernel, kernelName="test_multisum")
    kwa.states.unrollIdx = 1  # NumIndicesSummation=2 => unrollIdx=1
    kwa.states.globalReadIncsUseVgpr = globalReadIncsUseVgpr
    # indexChars: simple I,J,K,L,... mapping (indices 0-3 -> I,J,K,L)
    kwa.states.indexChars = list("IJKLMNOPQRSTUVWXYZ")
    kwa.states.archCaps = ri.getArchCaps()
    kwa.states.asmCaps = ri.getAsmCaps()
    kwa.states.regCaps = ri.getRegCaps()
    kwa.states.overflowedResources = 0

    return kwa


# ---------------------------------------------------------------------------
# Synthetic kernel dict used by both tests
# ---------------------------------------------------------------------------

# Two summation indices: [2, 3]
#   index 2  = "other" summation (the outer loop, loopIdx=0)
#   index 3  = unroll summation (the inner K loop, loopIdx=1 == unrollIdx)
# IndexAssignmentsA = [0, 2, 3]  (free-M=0, other-sum=2, unroll-K=3)
# IndexAssignmentsB = [3, 1, 2]  (unroll-K=3, free-N=1, other-sum=2)
_MULTISUM_KERNEL = {
    "ProblemType": {
        "NumIndicesSummation": 2,
        "IndicesSummation": [2, 3],
        "MirrorDimsA": [],
        "MirrorDimsB": [],
        "UseInitialStridesAB": False,
        "IndexAssignmentsA": [0, 2, 3],
        "IndexAssignmentsB": [3, 1, 2],
    },
    "DepthU": 16,
    "GlobalSplitU": 1,
    "_GlobalAccumulation": "MultipleBuffer",
    "GlobalSplitUAlgorithm": "MultipleBuffer",
}

# tP for the B tensor (isA=False skips the broken None-rReg arm at line 5047).
_TP_B = {
    "tensorChar": "B",
    "isA": False,
    "isB": True,
    "idx": 1,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMultiSumGraIncrementsSGPR:
    """SGPR path: BufferLoad=True → globalReadIncsUseVgpr=False.

    Exercises KernelWriterAssembly.py lines 5026-5092 (the else branch of the
    globalReadIncsUseVgpr check).  With no mirror dims, loopIdxPrev==unrollIdx,
    and isA=False the emitted Module contains:
      1. a comment block (addComment1)
      2. SMulI32   — stride * loopCounter product
      3. SSubI32   — stride subtract (the no-mirror-on-either-side else branch)
      4. SLShiftLeftB32 — scale by BpeGRLog2
    """

    def setup_method(self):
        _init_rocisa_gfx942()

    def test_emits_nonempty_module(self):
        """SGPR other-summation arm returns a non-empty Module."""
        kwa = _build_minimal_kwa(_MULTISUM_KERNEL, globalReadIncsUseVgpr=False)
        result = kwa.graIncrements(_MULTISUM_KERNEL, 0, _TP_B)
        items = list(result.items())
        assert len(items) >= 1, (
            f"Expected non-empty Module from graIncrements SGPR path, got {len(items)}"
        )

    def test_emits_multiply_instruction(self):
        """The emitted Module contains an SMulI32 (stride * loopCounter)."""
        from rocisa.instruction import SMulI32

        kwa = _build_minimal_kwa(_MULTISUM_KERNEL, globalReadIncsUseVgpr=False)
        result = kwa.graIncrements(_MULTISUM_KERNEL, 0, _TP_B)
        items = list(result.items())
        mul_items = [it for it in items if isinstance(it, SMulI32)]
        assert len(mul_items) >= 1, (
            f"Expected >=1 SMulI32 in SGPR path module; items: {[type(x).__name__ for x in items]}"
        )

    def test_emits_subtract_instruction(self):
        """The no-mirror else-branch (lines 5082-5086) emits an SSubI32."""
        from rocisa.instruction import SSubI32

        kwa = _build_minimal_kwa(_MULTISUM_KERNEL, globalReadIncsUseVgpr=False)
        result = kwa.graIncrements(_MULTISUM_KERNEL, 0, _TP_B)
        items = list(result.items())
        sub_items = [it for it in items if isinstance(it, SSubI32)]
        assert len(sub_items) >= 1, (
            f"Expected >=1 SSubI32 (no-mirror else-branch); "
            f"items: {[type(x).__name__ for x in items]}"
        )

    def test_emits_shift_instruction(self):
        """Line 5088: the bpeGR scale left-shift is emitted."""
        from rocisa.instruction import SLShiftLeftB32

        kwa = _build_minimal_kwa(_MULTISUM_KERNEL, globalReadIncsUseVgpr=False)
        result = kwa.graIncrements(_MULTISUM_KERNEL, 0, _TP_B)
        items = list(result.items())
        shift_items = [it for it in items if isinstance(it, SLShiftLeftB32)]
        assert len(shift_items) >= 1, (
            f"Expected >=1 SLShiftLeftB32 (line 5088); "
            f"items: {[type(x).__name__ for x in items]}"
        )

    def test_gra_inc_references_b_tensor(self):
        """The emitted code references the B-tensor global-read-increment SGPR name."""
        kwa = _build_minimal_kwa(_MULTISUM_KERNEL, globalReadIncsUseVgpr=False)
        result = kwa.graIncrements(_MULTISUM_KERNEL, 0, _TP_B)
        # Render items to strings and check for the B SGPR name.
        text = " ".join(str(it) for it in result.items())
        assert "GlobalReadIncsB" in text, (
            f"Expected 'GlobalReadIncsB' in emitted text; got:\n{text[:400]}"
        )

    def test_multiple_calls_return_modules(self):
        """Calling graIncrements twice (back-to-back) both return non-empty Modules,
        confirming the SGPR-pool checkin/checkout cycle is clean."""
        kwa = _build_minimal_kwa(_MULTISUM_KERNEL, globalReadIncsUseVgpr=False)
        result1 = kwa.graIncrements(_MULTISUM_KERNEL, 0, _TP_B)
        # allocTmpSgpr uses a context-manager that checks registers back in,
        # so the pool is already restored for a second call.
        result2 = kwa.graIncrements(_MULTISUM_KERNEL, 0, _TP_B)
        assert len(list(result1.items())) >= 1, "First call returned empty Module"
        assert len(list(result2.items())) >= 1, "Second call returned empty Module"


class TestMultiSumGraIncrementsVgprPrintExit:
    """VGPR path: BufferLoad=False → globalReadIncsUseVgpr=True.

    When globalReadIncsUseVgpr is True and loopIdx != unrollIdx, the code at
    lines 5023-5025 calls printExit() (which raises SystemExit(-1)).
    The test pins this validity-reject behaviour.
    """

    def setup_method(self):
        _init_rocisa_gfx942()

    def test_printexit_raises_systemexit(self):
        """VGPR other-summation arm calls printExit → SystemExit(-1)."""
        kwa = _build_minimal_kwa(_MULTISUM_KERNEL, globalReadIncsUseVgpr=True)
        with pytest.raises(SystemExit) as exc_info:
            kwa.graIncrements(_MULTISUM_KERNEL, 0, _TP_B)
        assert exc_info.value.code == -1, (
            f"Expected SystemExit(-1) from printExit; got code={exc_info.value.code}"
        )

    def test_printexit_message_contains_numsummation(self):
        """The printExit message includes NumIndicesSummation value."""
        import io
        import sys

        kwa = _build_minimal_kwa(_MULTISUM_KERNEL, globalReadIncsUseVgpr=True)
        captured = io.StringIO()
        with pytest.raises(SystemExit):
            with pytest.MonkeyPatch().context() as mp:
                mp.setattr(sys, "stdout", captured)
                kwa.graIncrements(_MULTISUM_KERNEL, 0, _TP_B)
        output = captured.getvalue()
        assert "2" in output, (
            f"Expected NumIndicesSummation=2 in printExit message; got: {output!r}"
        )
