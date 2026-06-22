################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R7 — GSU remaining-reduction / workspace arms characterization (CPU-only).

Exercises the GSU.py lines that are still uncovered after the existing test_r2
and test_r3 sweeps:

  GSUOff.graIncrements  — vgpr path   (lines 225-233)  globalReadIncsUseVgpr=True
  GSUOff.calculateIncrementMetadata   (lines 263-266)  direct call
  GSUOn.computeLoadSrd  — Sparse path  (lines 388-400)  Sparse=2, isB=True
  GSUOn.graIncrements   — vgpr path   (lines 442-461)  globalReadIncsUseVgpr=True
  GSUOn.graIncrements   — else branch (line 480)        SwizzleTensorB path
  GSUOn.graIncrements   — Mirror path (lines 506-507)   isMirrorIdx=True
  GSUOn.graIncrementsRestore           (lines 520-529)  direct call
  GSUOn.calculateLoopNumIterGsu        (lines 558-578)  gsuComponent dispatch
  GSUOn.calculateIncrementMetadata     (lines 580-588)  gsuComponent dispatch

All tests are CPU-only (no GPU, no compile).  Uses the rocisa singleton for
rocIsa init + a real RegisterPool for SGPR/VGPR allocation.  Pattern mirrors
test_r4_multisum_char.py (direct-driver style, pure-assert, no snapshots).

Target ranges for measurement: 170-233, 388-461, 442-588.
"""

import shutil

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Shared rocisa init
# ---------------------------------------------------------------------------


def _init_rocisa_gfx942():
    """Initialise the rocisa singleton for gfx942 / wave64.  Idempotent."""
    import rocisa

    ri = rocisa.rocIsa.getInstance()
    asmpath = shutil.which("amdclang++") or "/usr/bin/amdclang++"
    ri.init((9, 4, 2), asmpath)
    ri.setKernel((9, 4, 2), 64)
    return ri


# ---------------------------------------------------------------------------
# Minimal KWA builder
# ---------------------------------------------------------------------------


def _build_kwa(kernel, *, globalReadIncsUseVgpr: bool, labels=None):
    """Build a minimally-initialised KernelWriterAssembly for direct method calls.

    Sets only the state required by the GSU component methods under test.
    Follows the same pattern as test_r4_multisum_char._build_minimal_kwa.
    """
    from unittest.mock import MagicMock

    import rocisa as _rocisa_mod
    from rocisa.enum import RegisterType
    from rocisa.register import RegisterPool

    from Tensile.Common.Types import DebugConfig
    from Tensile.KernelWriter import KernelWriter, StateValues
    from Tensile.KernelWriterAssembly import KernelWriterAssembly, GlobalReadGprRecord

    ri = _rocisa_mod.rocIsa.getInstance()

    mock_asm = MagicMock()
    mock_asm.rocm_version = MagicMock()
    mock_asm.rocm_version.major = 6

    kwa = object.__new__(KernelWriterAssembly)
    KernelWriter.__init__(kwa, mock_asm, DebugConfig())
    kwa.globalread_gpr_record = GlobalReadGprRecord()

    # SGPR pool — 128 registers, all available.
    spool = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False, printRP=False)
    spool.add(0, 128, "r7-gsu-test")
    kwa.sgprPool = spool
    kwa.sgprs = {}
    kwa.dontAppendCode = False

    # VGPR pool — 64 registers, all available.
    vpool = RegisterPool(0, RegisterType.Vgpr, defaultPreventOverflow=False, printRP=False)
    vpool.add(0, 64, "r7-gsu-test-vgpr")
    kwa.vgprPool = vpool

    version = (9, 4, 2)
    kwa.states = StateValues(version=version, kernel=kernel, kernelName="test_r7_gsu")
    kwa.states.unrollIdx = kernel["ProblemType"]["NumIndicesSummation"] - 1
    kwa.states.globalReadIncsUseVgpr = globalReadIncsUseVgpr
    kwa.states.indexChars = list("IJKLMNOPQRSTUVWXYZ")
    kwa.states.archCaps = ri.getArchCaps()
    kwa.states.asmCaps = ri.getAsmCaps()
    kwa.states.regCaps = ri.getRegCaps()
    kwa.states.overflowedResources = 0

    # Labels instance — use real one from KWA if available, else from kernel name.
    if labels is not None:
        kwa.states.labels = labels

    return kwa


# ---------------------------------------------------------------------------
# Shared kernel dicts
# ---------------------------------------------------------------------------

# Minimal kernel for GSUOff (GSU=1) with a single summation index.
_KERNEL_GSU_OFF = {
    "ProblemType": {
        "NumIndicesSummation": 1,
        "IndicesSummation": [2],
        "MirrorDimsA": [],
        "MirrorDimsB": [],
        "UseInitialStridesAB": False,
        "IndexAssignmentsA": [0, 2],
        "IndexAssignmentsB": [2, 1],
        "Sparse": 0,
        "SwizzleTensorA": False,
        "SwizzleTensorB": False,
    },
    "DepthU": 16,
    "_DepthU": 16,
    "_DepthUA": 16,
    "_DepthUB": 16,
    "GlobalSplitU": 1,
    "_GlobalAccumulation": None,
    "GlobalSplitUAlgorithm": "MultipleBuffer",
    "WavefrontSize": 64,
    "MatrixInstM": 16,
    "MatrixInstN": 16,
    "enableTDMMetadata": False,
}

# Minimal kernel for GSUOn (GSU=4) with a single summation index.
_KERNEL_GSU_ON = {
    "ProblemType": {
        "NumIndicesSummation": 1,
        "IndicesSummation": [2],
        "MirrorDimsA": [],
        "MirrorDimsB": [],
        "MirrorDimsD": [],
        "UseInitialStridesAB": False,
        "IndexAssignmentsA": [0, 2],
        "IndexAssignmentsB": [2, 1],
        "Sparse": 0,
        "SwizzleTensorA": False,
        "SwizzleTensorB": False,
    },
    "DepthU": 16,
    "_DepthU": 16,
    "_DepthUA": 16,
    "_DepthUB": 16,
    "GlobalSplitU": 4,
    "_GlobalAccumulation": "MultipleBuffer",
    "GlobalSplitUAlgorithm": "MultipleBuffer",
    "WavefrontSize": 64,
    "MatrixInstM": 16,
    "MatrixInstN": 16,
    "ISA": (9, 4, 2),
    "enableTDMMetadata": False,
    "AdaptiveGemmGSUA": 0,
}

# Kernel with Sparse=2 for computeLoadSrd sparse path (lines 388-400).
_KERNEL_GSU_ON_SPARSE = dict(_KERNEL_GSU_ON)
_KERNEL_GSU_ON_SPARSE = {**_KERNEL_GSU_ON, "ProblemType": {
    **_KERNEL_GSU_ON["ProblemType"],
    "Sparse": 2,
}}

# tP for A tensor
_TP_A = {
    "tensorChar": "A",
    "isA": True,
    "isB": False,
    "isM": False,
    "idx": 0,
    "bpeGR": 2.0,
    "ia": [0, 2],
    "tlu": False,
    "isSwizzled": False,
}

# tP for B tensor
_TP_B = {
    "tensorChar": "B",
    "isA": False,
    "isB": True,
    "isM": False,
    "idx": 1,
    "bpeGR": 2.0,
    "ia": [2, 1],
    "tlu": False,
    "isSwizzled": False,
}


# ---------------------------------------------------------------------------
# Helper — get GSU component instances
# ---------------------------------------------------------------------------


def _get_gsu_off():
    """Return the GSUOff component instance (matches GSU==0 kernel)."""
    from Tensile.Components.GSU import GSUOff

    return GSUOff()


def _get_gsu_on():
    """Return the GSUOn component instance (matches GSU>0 kernel)."""
    from Tensile.Components.GSU import GSUOn

    return GSUOn()


# ---------------------------------------------------------------------------
# Tests: GSUOff — vgpr path (lines 225-233)
# ---------------------------------------------------------------------------


class TestGSUOffGraIncrementsVgprPath:
    """GSUOff.graIncrements with globalReadIncsUseVgpr=True (lines 225-233).

    The condition is: writer.states.globalReadIncsUseVgpr=True, which arises
    when kernel["BufferLoad"]=False (flat-load path).  We set the state flag
    directly without needing a full kernel emit.
    """

    def setup_method(self):
        _init_rocisa_gfx942()

    def test_vgpr_path_emits_module(self):
        """GSUOff.graIncrements vgpr path returns a non-empty Module."""
        kwa = _build_kwa(_KERNEL_GSU_OFF, globalReadIncsUseVgpr=True)
        gsu = _get_gsu_off()
        result = gsu.graIncrements(kwa, _KERNEL_GSU_OFF, 0, _TP_A)
        items = list(result.items())
        assert len(items) >= 1, (
            "GSUOff.graIncrements vgpr path should emit >=1 instructions"
        )

    def test_vgpr_path_emits_vmov(self):
        """GSUOff.graIncrements vgpr path emits VMovB32 instructions."""
        from rocisa.instruction import VMovB32

        kwa = _build_kwa(_KERNEL_GSU_OFF, globalReadIncsUseVgpr=True)
        gsu = _get_gsu_off()
        result = gsu.graIncrements(kwa, _KERNEL_GSU_OFF, 0, _TP_A)
        items = list(result.items())
        vmov_items = [it for it in items if isinstance(it, VMovB32)]
        assert len(vmov_items) >= 1, (
            f"Expected >=1 VMovB32 in vgpr path; items: {[type(x).__name__ for x in items]}"
        )

    def test_vgpr_path_references_tensor_A(self):
        """GSUOff.graIncrements vgpr path references GlobalReadIncsA SGPRs."""
        kwa = _build_kwa(_KERNEL_GSU_OFF, globalReadIncsUseVgpr=True)
        gsu = _get_gsu_off()
        result = gsu.graIncrements(kwa, _KERNEL_GSU_OFF, 0, _TP_A)
        text = " ".join(str(it) for it in result.items())
        assert "GlobalReadIncsA" in text, (
            f"Expected GlobalReadIncsA in vgpr-path output; got:\n{text[:400]}"
        )

    def test_vgpr_path_b_tensor(self):
        """GSUOff.graIncrements vgpr path works for B tensor too."""
        from rocisa.instruction import VMovB32

        kwa = _build_kwa(_KERNEL_GSU_OFF, globalReadIncsUseVgpr=True)
        gsu = _get_gsu_off()
        result = gsu.graIncrements(kwa, _KERNEL_GSU_OFF, 0, _TP_B)
        items = list(result.items())
        vmov_items = [it for it in items if isinstance(it, VMovB32)]
        assert len(vmov_items) >= 1, "Expected VMovB32 in B-tensor vgpr path"


# ---------------------------------------------------------------------------
# Tests: GSUOff — calculateIncrementMetadata (lines 263-266)
# ---------------------------------------------------------------------------


class TestGSUOffCalculateIncrementMetadata:
    """GSUOff.calculateIncrementMetadata (lines 263-266).

    This method emits a simple right-shift, reachable via
    KernelWriterAssembly.calculateIncrementMetadata dispatch to the GSUOff
    instance.  We call it directly on the GSUOff instance with a minimal writer.
    """

    def setup_method(self):
        _init_rocisa_gfx942()

    def test_emits_module(self):
        """GSUOff.calculateIncrementMetadata returns a non-empty Module."""
        kwa = _build_kwa(_KERNEL_GSU_OFF, globalReadIncsUseVgpr=False)
        gsu = _get_gsu_off()
        result = gsu.calculateIncrementMetadata(kwa, _KERNEL_GSU_OFF, "LoopCounterL")
        items = list(result.items())
        assert len(items) >= 1, "Expected non-empty Module from calculateIncrementMetadata"

    def test_emits_smov_and_shift(self):
        """GSUOff.calculateIncrementMetadata emits SMovB32 and SLShiftRightB32."""
        from rocisa.instruction import SLShiftRightB32, SMovB32

        kwa = _build_kwa(_KERNEL_GSU_OFF, globalReadIncsUseVgpr=False)
        gsu = _get_gsu_off()
        result = gsu.calculateIncrementMetadata(kwa, _KERNEL_GSU_OFF, "LoopCounterL")
        items = list(result.items())
        mov_items = [it for it in items if isinstance(it, SMovB32)]
        shr_items = [it for it in items if isinstance(it, SLShiftRightB32)]
        assert len(mov_items) >= 1, (
            f"Expected SMovB32; items: {[type(x).__name__ for x in items]}"
        )
        assert len(shr_items) >= 1, (
            f"Expected SLShiftRightB32; items: {[type(x).__name__ for x in items]}"
        )


# ---------------------------------------------------------------------------
# Tests: GSUOn — computeLoadSrd Sparse path (lines 388-400)
# ---------------------------------------------------------------------------


class TestGSUOnComputeLoadSrdSparsePath:
    """GSUOn.computeLoadSrd with Sparse=2, isB=True (lines 388-400).

    When Sparse==2 and tP["isB"] is True, the code sets divider=2 and
    adjusts depthUDiv, exercising lines 393-400.
    """

    def setup_method(self):
        _init_rocisa_gfx942()

    def _make_tP_sparse_B(self):
        """tP dict for the sparse B tensor."""
        return {
            "tensorChar": "B",
            "isA": False,
            "isB": True,
            "isM": False,
            "idx": 1,
            "bpeGR": 2.0,
            "ia": [2, 1],
            "tlu": False,
            "isSwizzled": False,
        }

    def test_sparse_path_emits_module(self):
        """GSUOn.computeLoadSrd with Sparse=2 isB=True returns non-empty Module."""
        kwa = _build_kwa(_KERNEL_GSU_ON_SPARSE, globalReadIncsUseVgpr=False)
        gsu = _get_gsu_on()
        tP = self._make_tP_sparse_B()
        # stmp and tileStart are SGPR index placeholders
        result = gsu.computeLoadSrd(kwa, _KERNEL_GSU_ON_SPARSE, tP, 10, 12)
        items = list(result.items())
        assert len(items) >= 1, (
            "GSUOn.computeLoadSrd Sparse path should emit >=1 instructions"
        )

    def test_sparse_divider_2_branch(self):
        """Sparse=2 with isB=True takes the divider=2 path — emits S instructions."""
        from rocisa.instruction import SAndB32

        kwa = _build_kwa(_KERNEL_GSU_ON_SPARSE, globalReadIncsUseVgpr=False)
        gsu = _get_gsu_on()
        tP = self._make_tP_sparse_B()
        result = gsu.computeLoadSrd(kwa, _KERNEL_GSU_ON_SPARSE, tP, 10, 12)
        items = list(result.items())
        and_items = [it for it in items if isinstance(it, SAndB32)]
        assert len(and_items) >= 1, (
            f"Expected >=1 SAndB32 in sparse-B computeLoadSrd; "
            f"items: {[type(x).__name__ for x in items][:10]}"
        )

    def test_sparse_1_isA_divider_2(self):
        """Sparse=1 with isA=True takes the divider=2 path (lines 394-400)."""
        # Sparse=1 means the A tensor is sparse; isA=True picks up divider=2.
        kernel_sparse1 = {
            **_KERNEL_GSU_ON,
            "ProblemType": {
                **_KERNEL_GSU_ON["ProblemType"],
                "Sparse": 1,
            },
        }
        tP_sparse_A = {
            "tensorChar": "A",
            "isA": True,
            "isB": False,
            "isM": False,
            "idx": 0,
            "bpeGR": 2.0,
            "ia": [0, 2],
            "tlu": False,
            "isSwizzled": False,
        }
        kwa = _build_kwa(kernel_sparse1, globalReadIncsUseVgpr=False)
        gsu = _get_gsu_on()
        result = gsu.computeLoadSrd(kwa, kernel_sparse1, tP_sparse_A, 10, 12)
        items = list(result.items())
        assert len(items) >= 1, "Expected non-empty module for Sparse=1 isA path"


# ---------------------------------------------------------------------------
# Tests: GSUOn — graIncrements vgpr path (lines 442-461)
# ---------------------------------------------------------------------------


class TestGSUOnGraIncrementsVgprPath:
    """GSUOn.graIncrements with globalReadIncsUseVgpr=True (lines 442-461).

    This path runs when kernel["BufferLoad"]=False (flat-load kernels).
    We set the state flag directly and call the method on a GSUOn instance.
    """

    def setup_method(self):
        _init_rocisa_gfx942()

    def test_vgpr_path_emits_module(self):
        """GSUOn.graIncrements vgpr path returns non-empty Module."""
        kwa = _build_kwa(_KERNEL_GSU_ON, globalReadIncsUseVgpr=True)
        gsu = _get_gsu_on()
        result = gsu.graIncrements(kwa, _KERNEL_GSU_ON, 0, _TP_A)
        items = list(result.items())
        assert len(items) >= 1, (
            "GSUOn.graIncrements vgpr path should emit >=1 instructions"
        )

    def test_vgpr_path_emits_vmov(self):
        """GSUOn.graIncrements vgpr path emits VMovB32 instructions."""
        from rocisa.instruction import VMovB32

        kwa = _build_kwa(_KERNEL_GSU_ON, globalReadIncsUseVgpr=True)
        gsu = _get_gsu_on()
        result = gsu.graIncrements(kwa, _KERNEL_GSU_ON, 0, _TP_A)
        items = list(result.items())
        vmov_items = [it for it in items if isinstance(it, VMovB32)]
        assert len(vmov_items) >= 1, (
            f"Expected >=1 VMovB32 in GSUOn vgpr path; items: {[type(x).__name__ for x in items]}"
        )

    def test_vgpr_path_emits_gsu_and(self):
        """GSUOn.graIncrements vgpr path emits SAndB32 to restore GSU bits."""
        from rocisa.instruction import SAndB32

        kwa = _build_kwa(_KERNEL_GSU_ON, globalReadIncsUseVgpr=True)
        gsu = _get_gsu_on()
        result = gsu.graIncrements(kwa, _KERNEL_GSU_ON, 0, _TP_A)
        items = list(result.items())
        and_items = [it for it in items if isinstance(it, SAndB32)]
        assert len(and_items) >= 1, (
            f"Expected >=1 SAndB32 in vgpr path (GSU bit restore); "
            f"items: {[type(x).__name__ for x in items]}"
        )

    def test_vgpr_path_b_tensor(self):
        """GSUOn.graIncrements vgpr path works for B tensor too."""
        from rocisa.instruction import VMovB32

        kwa = _build_kwa(_KERNEL_GSU_ON, globalReadIncsUseVgpr=True)
        gsu = _get_gsu_on()
        result = gsu.graIncrements(kwa, _KERNEL_GSU_ON, 0, _TP_B)
        items = list(result.items())
        vmov_items = [it for it in items if isinstance(it, VMovB32)]
        assert len(vmov_items) >= 1, "Expected VMovB32 in B-tensor GSUOn vgpr path"


# ---------------------------------------------------------------------------
# Tests: GSUOn — graIncrements else-branch SwizzleTensorB path (line 480)
# ---------------------------------------------------------------------------


class TestGSUOnGraIncrementsSwizzlePath:
    """GSUOn.graIncrements SGPR-path with SwizzleTensorB=True (line 480).

    When tc=="B" and kernel["ProblemType"]["SwizzleTensorB"]=True, the code
    sets mi_dim = kernel["MatrixInstN"] (line 480), exercising that branch.
    """

    def setup_method(self):
        _init_rocisa_gfx942()

    def test_swizzle_b_path_emits_module(self):
        """SwizzleTensorB=True path emits a non-empty Module."""
        kernel_swizzle = {
            **_KERNEL_GSU_ON,
            "ProblemType": {
                **_KERNEL_GSU_ON["ProblemType"],
                "SwizzleTensorB": True,
            },
        }
        tP_swizzle_B = {**_TP_B, "isSwizzled": True}
        kwa = _build_kwa(kernel_swizzle, globalReadIncsUseVgpr=False)
        gsu = _get_gsu_on()
        result = gsu.graIncrements(kwa, kernel_swizzle, 0, tP_swizzle_B)
        items = list(result.items())
        assert len(items) >= 1, (
            "SwizzleTensorB path should emit >=1 instructions"
        )


# ---------------------------------------------------------------------------
# Tests: GSUOn — graIncrements MirrorIdx path (line 507)
# ---------------------------------------------------------------------------


class TestGSUOnGraIncrementsMirrorPath:
    """GSUOn.graIncrements SGPR-path with isMirrorIdx=True (line 507).

    When dimIdx is in MirrorDimsA, isMirrorIdx=True and line 507
    (m.setMinus(True)) is executed.
    """

    def setup_method(self):
        _init_rocisa_gfx942()

    def test_mirror_a_path_emits_module(self):
        """MirrorDimsA path (isMirrorIdx=True) emits a non-empty Module."""
        kernel_mirror = {
            **_KERNEL_GSU_ON,
            "ProblemType": {
                **_KERNEL_GSU_ON["ProblemType"],
                "MirrorDimsA": [2],  # dimIdx=2 is the summation index
                "MirrorDimsB": [],
            },
        }
        kwa = _build_kwa(kernel_mirror, globalReadIncsUseVgpr=False)
        gsu = _get_gsu_on()
        result = gsu.graIncrements(kwa, kernel_mirror, 0, _TP_A)
        items = list(result.items())
        assert len(items) >= 1, "MirrorDimsA path should emit >=1 instructions"

    def test_mirror_b_path_emits_module(self):
        """MirrorDimsB path (isMirrorIdx=True for B) emits a non-empty Module."""
        kernel_mirror = {
            **_KERNEL_GSU_ON,
            "ProblemType": {
                **_KERNEL_GSU_ON["ProblemType"],
                "MirrorDimsA": [],
                "MirrorDimsB": [2],  # dimIdx=2 is the summation index
            },
        }
        kwa = _build_kwa(kernel_mirror, globalReadIncsUseVgpr=False)
        gsu = _get_gsu_on()
        result = gsu.graIncrements(kwa, kernel_mirror, 0, _TP_B)
        items = list(result.items())
        assert len(items) >= 1, "MirrorDimsB path should emit >=1 instructions"


# ---------------------------------------------------------------------------
# Tests: GSUOn — graIncrementsRestore (lines 520-529)
# ---------------------------------------------------------------------------


class TestGSUOnGraIncrementsRestore:
    """GSUOn.graIncrementsRestore (lines 520-529).

    Exercises the restore path that adjusts the loop counter by GSU*DepthU.
    """

    def setup_method(self):
        _init_rocisa_gfx942()

    def test_restore_emits_module(self):
        """GSUOn.graIncrementsRestore returns a non-empty Module."""
        kwa = _build_kwa(_KERNEL_GSU_ON, globalReadIncsUseVgpr=False)
        gsu = _get_gsu_on()
        result = gsu.graIncrementsRestore(kwa, _KERNEL_GSU_ON, "LoopCounterL")
        items = list(result.items())
        assert len(items) >= 1, "graIncrementsRestore should emit >=1 instructions"

    def test_restore_emits_and_and_mul(self):
        """GSUOn.graIncrementsRestore emits SAndB32 and SMulI32."""
        from rocisa.instruction import SAndB32, SMulI32

        kwa = _build_kwa(_KERNEL_GSU_ON, globalReadIncsUseVgpr=False)
        gsu = _get_gsu_on()
        result = gsu.graIncrementsRestore(kwa, _KERNEL_GSU_ON, "LoopCounterL")
        items = list(result.items())
        and_items = [it for it in items if isinstance(it, SAndB32)]
        mul_items = [it for it in items if isinstance(it, SMulI32)]
        assert len(and_items) >= 1, (
            f"Expected SAndB32 in graIncrementsRestore; items: {[type(x).__name__ for x in items]}"
        )
        assert len(mul_items) >= 1, (
            f"Expected SMulI32 in graIncrementsRestore; items: {[type(x).__name__ for x in items]}"
        )


# ---------------------------------------------------------------------------
# Tests: GSUOn — calculateLoopNumIterGsu (lines 558-578)
# ---------------------------------------------------------------------------


class TestGSUOnCalculateLoopNumIterGsu:
    """GSUOn.calculateLoopNumIterGsu (lines 558-578).

    This is called via the gsuComponent dispatch path in KernelWriterAssembly
    when NumIndicesSummation>1 AND GlobalSplitU>1 AND tP["isA"]=True.
    We call the method directly on the GSUOn instance.
    """

    def setup_method(self):
        _init_rocisa_gfx942()

    def _make_tmpSgprRes(self, kwa, size=3):
        from rocisa.container import ContinuousRegister

        idx = kwa.sgprPool.checkOut(size, "calc-gsu-test")
        return ContinuousRegister(idx=idx, size=size)

    def test_emits_module(self):
        """GSUOn.calculateLoopNumIterGsu returns a non-empty Module."""
        from rocisa.container import ContinuousRegister

        kwa = _build_kwa(_KERNEL_GSU_ON, globalReadIncsUseVgpr=False)
        gsu = _get_gsu_on()
        tmpRes = self._make_tmpSgprRes(kwa)
        result = gsu.calculateLoopNumIterGsu(kwa, _KERNEL_GSU_ON, "LoopCounterL", tmpRes)
        items = list(result.items())
        assert len(items) >= 1, "calculateLoopNumIterGsu should emit >=1 instructions"

    def test_emits_divide_and_remainder(self):
        """GSUOn.calculateLoopNumIterGsu emits a divide-remainder sequence."""
        from rocisa.instruction import SCmpLtU32

        kwa = _build_kwa(_KERNEL_GSU_ON, globalReadIncsUseVgpr=False)
        gsu = _get_gsu_on()
        tmpRes = self._make_tmpSgprRes(kwa)
        result = gsu.calculateLoopNumIterGsu(kwa, _KERNEL_GSU_ON, "LoopCounterL", tmpRes)
        items = list(result.items())
        # The function uses scalarUInt32DivideAndRemainder which emits multiple
        # instructions + SCmpLtU32 for the remainder check.
        cmp_items = [it for it in items if isinstance(it, SCmpLtU32)]
        assert len(cmp_items) >= 1, (
            f"Expected SCmpLtU32 in calculateLoopNumIterGsu; "
            f"items: {[type(x).__name__ for x in items][:10]}"
        )

    def test_emits_scmov(self):
        """GSUOn.calculateLoopNumIterGsu emits SCMovB32 for numIter increment."""
        from rocisa.instruction import SCMovB32

        kwa = _build_kwa(_KERNEL_GSU_ON, globalReadIncsUseVgpr=False)
        gsu = _get_gsu_on()
        tmpRes = self._make_tmpSgprRes(kwa)
        result = gsu.calculateLoopNumIterGsu(kwa, _KERNEL_GSU_ON, "LoopCounterL", tmpRes)
        items = list(result.items())
        scmov_items = [it for it in items if isinstance(it, SCMovB32)]
        assert len(scmov_items) >= 1, (
            f"Expected SCMovB32 in calculateLoopNumIterGsu; "
            f"items: {[type(x).__name__ for x in items][:10]}"
        )


# ---------------------------------------------------------------------------
# Tests: GSUOn — calculateIncrementMetadata (lines 580-588)
# ---------------------------------------------------------------------------


class TestGSUOnCalculateIncrementMetadata:
    """GSUOn.calculateIncrementMetadata (lines 580-588).

    Called via KernelWriterAssembly.calculateIncrementMetadata when a Sparse
    kernel needs to compute the metadata stride increment.  We call the method
    directly on the GSUOn instance.
    """

    def setup_method(self):
        _init_rocisa_gfx942()

    def test_emits_module(self):
        """GSUOn.calculateIncrementMetadata returns a non-empty Module."""
        kwa = _build_kwa(_KERNEL_GSU_ON, globalReadIncsUseVgpr=False)
        gsu = _get_gsu_on()
        result = gsu.calculateIncrementMetadata(kwa, _KERNEL_GSU_ON, "LoopCounterL")
        items = list(result.items())
        assert len(items) >= 1, "calculateIncrementMetadata should emit >=1 instructions"

    def test_emits_and_mul_shift(self):
        """GSUOn.calculateIncrementMetadata emits SAndB32, SMulI32, SLShiftRightB32."""
        from rocisa.instruction import SAndB32, SLShiftRightB32, SMulI32

        kwa = _build_kwa(_KERNEL_GSU_ON, globalReadIncsUseVgpr=False)
        gsu = _get_gsu_on()
        result = gsu.calculateIncrementMetadata(kwa, _KERNEL_GSU_ON, "LoopCounterL")
        items = list(result.items())
        and_items = [it for it in items if isinstance(it, SAndB32)]
        mul_items = [it for it in items if isinstance(it, SMulI32)]
        shr_items = [it for it in items if isinstance(it, SLShiftRightB32)]
        assert len(and_items) >= 1, (
            f"Expected SAndB32; items: {[type(x).__name__ for x in items]}"
        )
        assert len(mul_items) >= 1, (
            f"Expected SMulI32; items: {[type(x).__name__ for x in items]}"
        )
        assert len(shr_items) >= 1, (
            f"Expected SLShiftRightB32; items: {[type(x).__name__ for x in items]}"
        )

    def test_emits_scmov_for_gsuc_path(self):
        """GSUOn.calculateIncrementMetadata emits SCMovB32 for GSUC==1 path."""
        from rocisa.instruction import SCMovB32

        kwa = _build_kwa(_KERNEL_GSU_ON, globalReadIncsUseVgpr=False)
        gsu = _get_gsu_on()
        result = gsu.calculateIncrementMetadata(kwa, _KERNEL_GSU_ON, "LoopCounterL")
        items = list(result.items())
        scmov_items = [it for it in items if isinstance(it, SCMovB32)]
        assert len(scmov_items) >= 1, (
            f"Expected SCMovB32 (GSUC==1 path); items: {[type(x).__name__ for x in items]}"
        )
