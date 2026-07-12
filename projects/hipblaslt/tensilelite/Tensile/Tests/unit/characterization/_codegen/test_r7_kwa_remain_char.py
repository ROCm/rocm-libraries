################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R7 — KWA remaining mid-cluster characterization.

Exercises three uncovered clusters in Tensile/KernelWriterAssembly.py via
direct component invocation (CPU-only, no GPU, no compile).

Target missing ranges:
  9921-9990 : ``globalReadIncrement`` — BufferLoad=True non-stagger path
              (lines 9921-9933) and flat-addressing / VGPR-increment path
              (lines 9956-9975, requires BufferLoad=False).
              Gate: ``kernel["BufferLoad"]`` and ``states.staggerUCode`` (falsy).

  10055-10108: ``globalReadGuardK`` isTr path — DirectToVgpr + TLUA + gfx1201
               (HasGLTr16B128=True).  Gate: ``isTr = enableGLTrA=True``.

  2970-3017 : ``extractPackedCoord1ToRowStart`` — multi-dim packed free index.
              Gate: called from ComputeStoreVgprs when ``len(PackedC1IndicesX) > 1``.
              Function iterates over ``packedC1[:-1]``, so >=2 elements needed.

All tests are pure-assert (no syrupy snapshots) and CPU-only.
"""

import shutil

import pytest

pytestmark = pytest.mark.unit

# Force full Tensile package init before component imports.
import rocisa  # noqa: F401
import Tensile.KernelWriter  # noqa: F401


# ---------------------------------------------------------------------------
# Shared setup helpers
# ---------------------------------------------------------------------------

def _init_rocisa(version=(9, 4, 2), wave64=True):
    """Initialise rocisa singleton for *version*.  Idempotent."""
    ri = rocisa.rocIsa.getInstance()
    asmpath = shutil.which("amdclang++") or "/usr/bin/amdclang++"
    ri.init(version, asmpath)
    ri.setKernel(version, 64 if wave64 else 32)
    return ri


def _build_kwa_for_global_read_inc(kernel, *, version=(9, 4, 2), bufferLoad=True):
    """Build a minimal KernelWriterAssembly for globalReadIncrement tests.

    Sets up the minimum required state:
      - sgprPool (80 registers)
      - states.unrollIdx
      - states.globalReadIncsUseVgpr  (False when BufferLoad, True otherwise)
      - states.use64bShadowLimit / use64bShadowLimitMX (False to simplify incrementSrd)
      - states.staggerUCode (None/falsy -> non-stagger path)
      - states.a / states.b defaults (useConstSgprGlobalReadIncs=False)
      - states.indexChars
    """
    from unittest.mock import MagicMock

    import rocisa as _rocisa_mod
    from rocisa.enum import RegisterType
    from rocisa.register import RegisterPool

    from Tensile.Common.Types import DebugConfig
    from Tensile.KernelWriter import KernelWriter, StateValues
    from Tensile.KernelWriterAssembly import KernelWriterAssembly, GlobalReadGprRecord

    ri = _init_rocisa(version)

    mock_asm = MagicMock()
    mock_asm.rocm_version = MagicMock()
    mock_asm.rocm_version.major = 6

    kwa = object.__new__(KernelWriterAssembly)
    KernelWriter.__init__(kwa, mock_asm, DebugConfig())
    kwa.globalread_gpr_record = GlobalReadGprRecord()

    # SGPR pool.
    pool = RegisterPool(
        0, RegisterType.Sgpr, defaultPreventOverflow=False, printRP=False
    )
    pool.add(0, 120, "kwa-gri-test")
    kwa.sgprPool = pool
    kwa.sgprs = {}
    kwa.dontAppendCode = False

    kwa.states = StateValues(
        version=version,
        kernel=kernel,
        kernelName="test_kwa_remain",
    )
    kwa.states.unrollIdx = 0  # single summation; unrollIdx=0
    kwa.states.globalReadIncsUseVgpr = not bufferLoad
    kwa.states.use64bShadowLimit = False
    kwa.states.use64bShadowLimitMX = False
    kwa.states.staggerUCode = None  # falsy -> non-stagger path at line 9920
    kwa.states.indexChars = list("IJKLMNOPQRSTUVWXYZ")
    kwa.states.archCaps = ri.getArchCaps()
    kwa.states.asmCaps = ri.getAsmCaps()
    kwa.states.regCaps = ri.getRegCaps()
    kwa.states.overflowedResources = 0
    # states.a / states.b are ABMatrixInfo() by default;
    # useConstSgprGlobalReadIncs=False by default.

    return kwa



def _build_kwa_for_packed_coord(*, version=(9, 4, 2)):
    """Build a minimal KWA for extractPackedCoord1ToRowStart.

    Needs a VGPR pool (checkOut(4)) and vgprs.coutRowPtrD set.
    """
    from unittest.mock import MagicMock

    import rocisa as _rocisa_mod
    from rocisa.enum import RegisterType
    from rocisa.register import RegisterPool

    from Tensile.Common.Types import DebugConfig
    from Tensile.KernelWriter import KernelWriter, StateVgprs, StateValues
    from Tensile.KernelWriterAssembly import KernelWriterAssembly, GlobalReadGprRecord

    ri = _init_rocisa(version)

    mock_asm = MagicMock()
    mock_asm.rocm_version = MagicMock()
    mock_asm.rocm_version.major = 6

    kwa = object.__new__(KernelWriterAssembly)
    KernelWriter.__init__(kwa, mock_asm, DebugConfig())
    kwa.globalread_gpr_record = GlobalReadGprRecord()

    sgpr_pool = RegisterPool(
        0, RegisterType.Sgpr, defaultPreventOverflow=False, printRP=False
    )
    sgpr_pool.add(0, 120, "packed-sgpr")
    kwa.sgprPool = sgpr_pool
    kwa.sgprs = {}

    vgpr_pool = RegisterPool(
        0, RegisterType.Vgpr, defaultPreventOverflow=False, printRP=False
    )
    vgpr_pool.add(0, 256, "packed-vgpr")
    kwa.vgprPool = vgpr_pool
    kwa.dontAppendCode = False

    kernel = _make_packed_coord_kernel()
    kwa.states = StateValues(
        version=version,
        kernel=kernel,
        kernelName="test_packed_coord",
    )
    kwa.states.unrollIdx = 0
    kwa.states.globalReadIncsUseVgpr = False
    kwa.states.use64bShadowLimit = False
    kwa.states.use64bShadowLimitMX = False
    kwa.states.staggerUCode = None
    kwa.states.indexChars = list("IJKLMNOPQRSTUVWXYZ")
    kwa.states.archCaps = ri.getArchCaps()
    kwa.states.asmCaps = ri.getAsmCaps()
    kwa.states.regCaps = ri.getRegCaps()
    kwa.states.overflowedResources = 0

    # Allocate coutRowPtrD from the VGPR pool so extractPackedCoord1ToRowStart
    # can write into it at line 3013.
    kwa.vgprs = StateVgprs()
    kwa.vgprs.coutRowPtrD = vgpr_pool.checkOut(1, "coutRowPtrD")

    return kwa, kernel


# ---------------------------------------------------------------------------
# Kernel dict builders
# ---------------------------------------------------------------------------

def _make_gri_kernel_buffer_load():
    """Kernel dict for globalReadIncrement BufferLoad=True, single summation."""
    return {
        "ProblemType": {
            "NumIndicesSummation": 1,
            "IndicesSummation": [2],
            "MirrorDimsA": [],
            "MirrorDimsB": [],
            "Sparse": 0,
        },
        "BufferLoad": True,
        "DepthU": 16,
        "GlobalSplitU": 1,
        "_GlobalAccumulation": "SingleBuffer",
        "GlobalSplitUAlgorithm": "SingleBuffer",
        # For the non-stagger incrementSrd path, DirectToVgprA/B must be
        # present in the kernel dict (accessed at line 9884 via tcOther).
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "PrefetchGlobalRead": 1,
    }


def _make_gri_kernel_flat():
    """Kernel dict for globalReadIncrement BufferLoad=False (flat addressing)."""
    return {
        "ProblemType": {
            "NumIndicesSummation": 1,
            "IndicesSummation": [2],
            "MirrorDimsA": [],
            "MirrorDimsB": [],
            "Sparse": 0,
        },
        "BufferLoad": False,
        "DepthU": 16,
        "GlobalSplitU": 1,
        "_GlobalAccumulation": "SingleBuffer",
        "GlobalSplitUAlgorithm": "SingleBuffer",
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "PrefetchGlobalRead": 1,
    }



def _make_packed_coord_kernel():
    """Kernel dict for extractPackedCoord1ToRowStart.

    The function is gated on UseE/GSU at line 2970 (must be False/0/non-SK).
    It reads kernel["MagicDivAlg"] (int) and uses self.sizeRef(idx) /
    self.strideRef('D', idx) — these resolve via the sgpr-name helpers.
    strideRef needs ProblemType["UseInitialStridesCD"] (line 314 of KWA).
    """
    return {
        "ProblemType": {
            "NumIndicesC": 3,
            "NumIndicesSummation": 1,
            "IndicesSummation": [3],
            "MirrorDimsA": [],
            "MirrorDimsB": [],
            "Sparse": 0,
            # UseE=False ensures no printExit at line 2970.
            "UseE": False,
            # strideRef('D', dim) uses UseInitialStridesCD at line 314.
            "UseInitialStridesCD": False,
            # strideRef also checks IndexAssignmentsA/B for A/B tensors only;
            # but D/C/E use UseInitialStridesCD only. Provide minimal A/B
            # assignments in case anything else needs them.
            "IndexAssignmentsA": [0, 2, 3],
            "IndexAssignmentsB": [1, 2, 3],
        },
        "GlobalSplitU": 0,  # not 1 or -1 -> no printExit at 2970
        "StreamK": 0,       # StreamK=0 -> no printExit at 2970
        "MagicDivAlg": 2,   # algorithm branch at line 2988/2992
        # PackedC1IndicesX: two free indices in dim-1 -> triggers the loop
        # in extractPackedCoord1ToRowStart (packedC1[:-1] = [0]).
        # With packedC1 = [0, 1], the loop iterates once (i=0, idx=0)
        # then processes the final element (idx=1).
        "PackedC1IndicesX": [0, 1],
        "DepthU": 16,
    }


# ---------------------------------------------------------------------------
# tP dicts
# ---------------------------------------------------------------------------

def _make_tP_A_buffer():
    """tP for the A tensor, BufferLoad, non-stagger, no mirror dims."""
    return {
        "tensorChar": "A",
        "isA": True,
        "isB": False,
        "idx": 0,
    }


def _make_tP_B_flat():
    """tP for the B tensor, flat addressing (BufferLoad=False).

    Needs nrp/nrpv/nrc/nrcv/nrcvpi for the loop at lines 9958-9990.
    rpga is on states.
    """
    return {
        "tensorChar": "B",
        "isA": False,
        "isB": True,
        "idx": 1,
        # Number-of-reads parameters: one read per thread.
        "nrp": 1,
        "nrpv": 1,
        "nrc": 1,
        "nrcv": 1,
        "nrcvpi": 1,
    }



# ---------------------------------------------------------------------------
# Tests — globalReadIncrement non-stagger BufferLoad=True  (lines 9921-9933)
# ---------------------------------------------------------------------------

class TestGlobalReadIncrementNonStagger:
    """globalReadIncrement BufferLoad=True, no stagger, loopIdx==unrollIdx.

    This is the ``else`` branch of ``if loopIdx == self.states.unrollIdx and
    self.states.staggerUCode`` (line 9875) combined with the ``else`` at
    line 9928 (loopIdx==unrollIdx, no MirrorDims).

    Lines exercised: 9921 (grIncTcs assignment), 9922 (if condition, False),
    9928 (else), 9929-9933 (incUpper=0 + srcGRInc + incrementSrd).
    """

    def setup_method(self):
        _init_rocisa()

    def test_non_stagger_bufferload_emits_module(self):
        """Non-stagger BufferLoad=True path returns non-empty Module."""
        from rocisa.code import Module

        kernel = _make_gri_kernel_buffer_load()
        kwa = _build_kwa_for_global_read_inc(kernel, bufferLoad=True)
        tP = _make_tP_A_buffer()

        imod = Module("test_gri")
        kwa.globalReadIncrement(kernel, imod, 0, tP, 0)
        items = list(imod.items())
        assert len(items) >= 1, (
            f"Expected non-empty Module from non-stagger BufferLoad path; "
            f"got {len(items)} items"
        )

    def test_non_stagger_bufferload_comment_present(self):
        """Non-stagger path emits 'global read inc A' comment."""
        from rocisa.code import Module

        kernel = _make_gri_kernel_buffer_load()
        kwa = _build_kwa_for_global_read_inc(kernel, bufferLoad=True)
        tP = _make_tP_A_buffer()

        imod = Module("test_gri_comment")
        kwa.globalReadIncrement(kernel, imod, 0, tP, 0)
        flat = imod.prettyPrint("")
        assert "global read inc" in flat, (
            f"Expected 'global read inc' comment in emitted module; got:\n{flat[:400]}"
        )

    def test_non_stagger_bufferload_srd_inc_present(self):
        """Non-stagger path emits SAddU32 (from incrementSrd) for SRD increment."""
        from rocisa.code import Module
        from rocisa.instruction import SAddU32, SAddCU32

        kernel = _make_gri_kernel_buffer_load()
        kwa = _build_kwa_for_global_read_inc(kernel, bufferLoad=True)
        tP = _make_tP_A_buffer()

        imod = Module("test_gri_srd")
        kwa.globalReadIncrement(kernel, imod, 0, tP, 0)
        items = list(imod.items())
        adds = [it for it in items if isinstance(it, (SAddU32, SAddCU32))]
        assert len(adds) >= 1, (
            f"Expected SAddU32/SAddCU32 from incrementSrd; "
            f"found: {[type(x).__name__ for x in items]}"
        )


# ---------------------------------------------------------------------------
# Tests — globalReadIncrement flat addressing BufferLoad=False  (lines 9956-9975)
# ---------------------------------------------------------------------------

class TestGlobalReadIncrementFlat:
    """globalReadIncrement BufferLoad=False, globalReadIncsUseVgpr=True path.

    The ``else`` branch at line 9956 iterates over nrp/nrpv/nrc/nrcv and
    emits VAddCOU32 + VAddCCOU32 pairs using VGPR increments (lines 9963-9975).
    """

    def setup_method(self):
        _init_rocisa()

    def test_flat_path_emits_module(self):
        """Flat-addressing path returns non-empty Module (isTr=False, BufferLoad=False)."""
        from rocisa.code import Module

        kernel = _make_gri_kernel_flat()
        kwa = _build_kwa_for_global_read_inc(kernel, bufferLoad=False)
        # rpga: registers per global address (2 for 64-bit flat addresses).
        kwa.states.rpga = 2
        tP = _make_tP_B_flat()

        imod = Module("test_flat")
        kwa.globalReadIncrement(kernel, imod, 0, tP, 0)
        items = list(imod.items())
        assert len(items) >= 1, (
            f"Expected non-empty Module from flat-address path; got {len(items)}"
        )

    def test_flat_path_emits_vadd_cou(self):
        """Flat-addressing path emits VAddCOU32 (lower 32-bit addr increment)."""
        from rocisa.code import Module
        from rocisa.instruction import VAddCOU32

        kernel = _make_gri_kernel_flat()
        kwa = _build_kwa_for_global_read_inc(kernel, bufferLoad=False)
        kwa.states.rpga = 2
        tP = _make_tP_B_flat()

        imod = Module("test_flat_vadd")
        kwa.globalReadIncrement(kernel, imod, 0, tP, 0)
        items = list(imod.items())
        vadd = [it for it in items if isinstance(it, VAddCOU32)]
        assert len(vadd) >= 1, (
            f"Expected VAddCOU32 from flat VGPR-inc path (line 9963); "
            f"items: {[type(x).__name__ for x in items]}"
        )

    def test_flat_path_emits_vadd_ccou(self):
        """Flat-addressing path emits VAddCCOU32 (upper 32-bit carry propagation)."""
        from rocisa.code import Module
        from rocisa.instruction import VAddCCOU32

        kernel = _make_gri_kernel_flat()
        kwa = _build_kwa_for_global_read_inc(kernel, bufferLoad=False)
        kwa.states.rpga = 2
        tP = _make_tP_B_flat()

        imod = Module("test_flat_vaccou")
        kwa.globalReadIncrement(kernel, imod, 0, tP, 0)
        items = list(imod.items())
        vaccou = [it for it in items if isinstance(it, VAddCCOU32)]
        assert len(vaccou) >= 1, (
            f"Expected VAddCCOU32 from flat VGPR-inc path (line 9969); "
            f"items: {[type(x).__name__ for x in items]}"
        )


# ---------------------------------------------------------------------------
# P5 ceiling evidence — globalReadGuardK isTr path (lines 10055-10108)
# ---------------------------------------------------------------------------
# Lines 10055-10108 live inside globalReadGuardK.  The isTr block at 10053
# only computes the max-address VGPR (lines 10055-10108), but control flow
# falls through immediately into the nested ``globalReadGuardKBody`` closure
# (line 10155) which is called at line 10718.  That body requires
# ``tP["globalReadInstruction"].totalWidth`` (line 10187), ``tP["NonTemporal"]``
# (line 10194), and dozens of other fully-derived tensor parameters that only
# exist after a complete kernel-derivation pass (Solution + KernelWriter.__init__
# + all the prepareCode / makeSchedule passes).
#
# A direct-call approach cannot fake these without re-implementing the full
# derivation pipeline.  The config-driven emit approach (emit_kernels_from_config)
# with a gfx1201 + DirectToVgprA + TLUA=True + FP16 config would reach these
# lines but multiprocessing coverage jitter means a single-shard isolated test
# is not reliable.
#
# P5 ceiling verdict: 10055-10108 requires a real end-to-end kernel emit.
# Kept=False; measured_marginal for this sub-cluster is 0 in isolation.
# (The driver's full-suite run via coverage combine may reach these lines if
# a gfx1201 + GLTr config is added to the curated emit set.)


# ---------------------------------------------------------------------------
# Tests — extractPackedCoord1ToRowStart (lines 2970-3017)
# ---------------------------------------------------------------------------

class TestExtractPackedCoord1ToRowStart:
    """extractPackedCoord1ToRowStart with two packed free indices.

    Gate: called from ComputeStoreVgprs when len(PackedC1IndicesX) > 1.
    We call it directly with packedC1=[0, 1] (two elements):
      - Loop at line 2985 iterates once (i=0, idx=0): emits div + mul + sub
      - Final element at line 3010 (idx=1): emits VMulLOU32 + VAddU32

    Line 2988: HasVgprMSB=0 on gfx942 -> MacroInstruction V_MAGIC_DIV (else at 2991).
    Line 2997: i==0 -> VMulLOU32 into tmpV3.
    Line 3010: extract final idx (element 1).
    """

    def setup_method(self):
        _init_rocisa()

    def test_packed_coord_emits_module(self):
        """extractPackedCoord1ToRowStart returns a non-empty Module."""
        from rocisa.code import Module

        kwa, kernel = _build_kwa_for_packed_coord()
        # packedCoordVgpr: allocate one VGPR as the "incoming packed coord1"
        packed_vgpr = kwa.vgprPool.checkOut(1, "packed_coord_test")
        packedC1 = [0, 1]  # two free indices in dim-1

        module = kwa.extractPackedCoord1ToRowStart(kernel, packedC1, packed_vgpr, "D")
        kwa.vgprPool.checkIn(packed_vgpr)

        assert isinstance(module, Module), (
            f"Expected Module from extractPackedCoord1ToRowStart; got {type(module)}"
        )
        items = list(module.items())
        assert len(items) >= 2, (
            f"Expected >=2 instructions; got {len(items)}: "
            f"{[type(x).__name__ for x in items]}"
        )

    def test_packed_coord_vmov_copy(self):
        """Line 2984: VMovB32 copy of incoming packed coordinate into tmpV0."""
        from rocisa.code import Module
        from rocisa.instruction import VMovB32

        kwa, kernel = _build_kwa_for_packed_coord()
        packed_vgpr = kwa.vgprPool.checkOut(1, "packed_coord_vmov")
        packedC1 = [0, 1]

        module = kwa.extractPackedCoord1ToRowStart(kernel, packedC1, packed_vgpr, "D")
        kwa.vgprPool.checkIn(packed_vgpr)

        items = list(module.items())
        vmovs = [it for it in items if isinstance(it, VMovB32)]
        assert len(vmovs) >= 1, (
            f"Expected VMovB32 copy (line 2984); items: {[type(x).__name__ for x in items]}"
        )

    def test_packed_coord_vmullo_scale(self):
        """Lines 2995/2998/3011: VMulLOU32 for remainder and stride scaling."""
        from rocisa.code import Module
        from rocisa.instruction import VMulLOU32

        kwa, kernel = _build_kwa_for_packed_coord()
        packed_vgpr = kwa.vgprPool.checkOut(1, "packed_coord_mul")
        packedC1 = [0, 1]

        module = kwa.extractPackedCoord1ToRowStart(kernel, packedC1, packed_vgpr, "D")
        kwa.vgprPool.checkIn(packed_vgpr)

        items = list(module.items())
        muls = [it for it in items if isinstance(it, VMulLOU32)]
        assert len(muls) >= 2, (
            f"Expected >=2 VMulLOU32 (remainder-part-1 + stride-scale); "
            f"items: {[type(x).__name__ for x in items]}"
        )

    def test_packed_coord_vsub_remainder(self):
        """Line 2996: VSubU32 for remainder part 2 (coord mod size)."""
        from rocisa.code import Module
        from rocisa.instruction import VSubU32

        kwa, kernel = _build_kwa_for_packed_coord()
        packed_vgpr = kwa.vgprPool.checkOut(1, "packed_coord_sub")
        packedC1 = [0, 1]

        module = kwa.extractPackedCoord1ToRowStart(kernel, packedC1, packed_vgpr, "D")
        kwa.vgprPool.checkIn(packed_vgpr)

        items = list(module.items())
        subs = [it for it in items if isinstance(it, VSubU32)]
        assert len(subs) >= 1, (
            f"Expected VSubU32 remainder (line 2996); items: {[type(x).__name__ for x in items]}"
        )

    def test_packed_coord_vadd_row_accumulate(self):
        """Line 3013: VAddU32 accumulates rowStart = tmpV3 + scaled final dim."""
        from rocisa.code import Module
        from rocisa.instruction import VAddU32

        kwa, kernel = _build_kwa_for_packed_coord()
        packed_vgpr = kwa.vgprPool.checkOut(1, "packed_coord_add")
        packedC1 = [0, 1]

        module = kwa.extractPackedCoord1ToRowStart(kernel, packedC1, packed_vgpr, "D")
        kwa.vgprPool.checkIn(packed_vgpr)

        items = list(module.items())
        adds = [it for it in items if isinstance(it, VAddU32)]
        assert len(adds) >= 1, (
            f"Expected VAddU32 rowStart accumulation (line 3013); "
            f"items: {[type(x).__name__ for x in items]}"
        )
