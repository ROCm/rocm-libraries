"""Shared helpers for subtile tail-loop scaffold unit tests.

`test_subtile_tailloop_emit.py` and `test_SubtileBasedLogicalScheduler.py`
both drive `KernelWriter._emitTailLoopScaffoldSubtile` directly against
the same minimal `KernelWriterAssembly` state (sgpr/vgpr pools, register
caps, tile infos, problem dict). Before this module the setup code was
copy-pasted between the two files; this module consolidates it so a
future state-shape change (new sgpr, new tile-info field, new arch-cap
dependency) only has to land in one place.

The kernel dict is still built per-test in the calling file because the
two test suites have different kernel-builder helpers
(`_create_kernel` in `test_subtile_tailloop_emit.py` vs `create_kernel`
in `gpu_test_helpers`). Once a caller has a kernel dict it passes it
to `build_minimal_subtile_kwa(kernel)` to get a fully-wired
`KernelWriterAssembly` ready for a single scaffold emit.
"""

import shutil
from unittest.mock import MagicMock


def init_rocisa_for_gfx950():
    """Initialise the rocisa singleton against gfx950 with wavesize 64.

    Other test imports may have initialised the singleton against a
    different ISA, leaving the per-kernel caps un-set for our target.
    Always re-init to make this fixture self-contained.
    """
    from rocisa import rocIsa

    asmpath = shutil.which("amdclang++") or "/usr/bin/amdclang++"
    ri = rocIsa.getInstance()
    ri.init((9, 5, 0), asmpath)
    ri.setKernel((9, 5, 0), 64)
    return ri


def setdefault_tail_scaffold_kernel_keys(kernel, pgr, *, asem=32):
    """Populate the kernel-dict keys the scaffold reaches into.

    Idempotent: uses `setdefault` so callers can override any key
    upstream and this helper only fills the gaps. `asem` overrides
    `AssertSummationElementMultiple` (default 32 preserves the
    aligned-K test callers).
    """
    kernel["PrefetchGlobalRead"] = pgr
    kernel.setdefault("SuppressNoLoadLoop", False)
    kernel.setdefault("LocalSplitU", 1)
    kernel.setdefault("GlobalSplitU", 0)
    kernel.setdefault("StreamK", 0)
    kernel.setdefault("InnerUnroll", 1)
    kernel.setdefault("UseDotInstruction", False)
    kernel.setdefault("EnableMatrixInstruction", True)
    kernel.setdefault("NumDotElements", 1)
    kernel.setdefault("NumWaveSplitK", 1)
    kernel.setdefault("ExpertSchedulingMode", 0)
    # Must overwrite (not setdefault): `_create_kernel` pre-bakes
    # `AssertSummationElementMultiple: 32`, which would otherwise mask
    # a caller-supplied asem.
    kernel["AssertSummationElementMultiple"] = asem
    kernel.setdefault("MatrixInstB", 1)
    miInputDefault = (kernel["MatrixInstK"] * kernel["MatrixInstM"]
                      // kernel["WavefrontSize"])
    kernel.setdefault("MIInputPerThreadA", miInputDefault)
    kernel.setdefault("MIInputPerThreadB", miInputDefault)
    pt = kernel["ProblemType"]
    pt.setdefault("IndicesSummation", [3])
    pt.setdefault("Sparse", 0)
    pt.setdefault("DirectToVgprSparseMetadata", False)
    pt.setdefault("MXBlockA", 0)
    pt.setdefault("MXBlockB", 0)
    return kernel


def build_minimal_subtile_kwa(kernel):
    """Build a `KernelWriterAssembly` ready for a single
    `_emitTailLoopScaffoldSubtile` call against `kernel`.

    Includes register pools, arch/asm/reg caps, sgpr definitions for
    the scaffold's named references (LoopCounterL, OrigLoopCounter,
    SizesSum, Srd{A,B,MXSA,MXSB}, LocalWriteBaseAddr{A,B,MXSA,MXSB},
    Swap{A,B,MXSA,MXSB} for PGR>0), and real `TileInfo` instances on
    `states.{a,b,mxsa,mxsb,d}.tileInfo` with allocated `vgprTiles`
    and offset registers.

    Caller must have already populated `kernel` via
    `setdefault_tail_scaffold_kernel_keys(kernel, pgr)` (or
    equivalent inline setdefaults).
    """
    from rocisa.register import RegisterPool
    from rocisa.enum import RegisterType

    from Tensile.Common.Types import DebugConfig
    from Tensile.KernelWriter import (
        CodeModules, KernelWriter, StateValues, StateVgprs,
    )
    from Tensile.KernelWriterAssembly import (
        GlobalReadGprRecord, KernelWriterAssembly,
    )

    ri = init_rocisa_for_gfx950()

    mock_assembler = MagicMock()
    mock_assembler.rocm_version = MagicMock()
    mock_assembler.rocm_version.major = 6

    kwa = object.__new__(KernelWriterAssembly)
    KernelWriter.__init__(kwa, mock_assembler, DebugConfig())
    kwa.globalread_gpr_record = GlobalReadGprRecord()

    kwa.sgprPool = RegisterPool(0, RegisterType.Sgpr, False)
    kwa.vgprPool = RegisterPool(0, RegisterType.Vgpr, False)
    kwa.agprPool = RegisterPool(0, RegisterType.Accvgpr, False)
    kwa.sgprs = {}
    kwa.codes = CodeModules()

    kwa.states = StateValues(version=(9, 5, 0), kernel=kernel,
                             kernelName="test_tail_loop_scaffold")
    kwa.vgprs = StateVgprs()

    kwa.states.archCaps = ri.getArchCaps()
    kwa.states.asmCaps = ri.getAsmCaps()
    kwa.states.regCaps = ri.getRegCaps()
    kwa.states.indexChars = list("IJKLMNOPQRSTUVWXYZ")
    kwa.states.unrollIdx = 0
    kwa.states.unrollChar = "L"
    kwa.states.numReadsIterCoalescedA = 1
    kwa.states.numReadsIterCoalescedB = 1

    # Sgprs that `calculateLoopNumIter(-1)` and `closeLoop(-1)` reference
    # by name. `defineSgprIdx` registers each into `kwa.sgprs` so
    # `sgpr(name)` resolves to a concrete register index in the emitted
    # asm.
    kwa.defineSgprIdx("LoopCounterL", 1)
    kwa.defineSgprIdx("OrigLoopCounter", 1)
    kwa.defineSgprIdx("SizesSum", 1)

    # 4-sgpr SRD descriptors. The scaffold's PGR>0 SRD add/sub touches
    # only the low two sgprs, but the full 4-sgpr alloc matches the
    # production layout.
    kwa.defineSgprIdx("SrdA", 4, 4)
    kwa.defineSgprIdx("SrdB", 4, 4)
    if kernel["ProblemType"].get("MXBlockA", 0) > 0:
        kwa.defineSgprIdx("SrdMXSA", 4, 4)
    if kernel["ProblemType"].get("MXBlockB", 0) > 0:
        kwa.defineSgprIdx("SrdMXSB", 4, 4)

    # GR/LR M0-base sgprs referenced by `globalReadDoSubtile` and
    # `globalReadDoScaleSubtile` per load. Production sets these in
    # `globalReadDTLInitCommonSgpr` and the scale equivalent.
    kwa.defineSgprIdx("LocalWriteBaseAddrA", 1)
    kwa.defineSgprIdx("LocalWriteBaseAddrB", 1)
    if kernel["ProblemType"].get("MXBlockA", 0) > 0:
        kwa.defineSgprIdx("LocalWriteBaseAddrMXSA", 1)
    if kernel["ProblemType"].get("MXBlockB", 0) > 0:
        kwa.defineSgprIdx("LocalWriteBaseAddrMXSB", 1)

    # PGR>0 entry-gate emits `s_xor LWA<tc>, Swap<tc>` (the
    # double-buffer toggle mask). PGR=0 never reaches it.
    if kernel.get("PrefetchGlobalRead", 0) > 0:
        kwa.defineSgprIdx("SwapA", 1)
        kwa.defineSgprIdx("SwapB", 1)
        if kernel["ProblemType"].get("MXBlockA", 0) > 0:
            kwa.defineSgprIdx("SwapMXSA", 1)
        if kernel["ProblemType"].get("MXBlockB", 0) > 0:
            kwa.defineSgprIdx("SwapMXSB", 1)

    populate_subtile_tile_infos(kwa, kernel)
    return kwa


def populate_subtile_tile_infos(kwa, kernel):
    """Populate `kwa.states.{a,b,mxsa,mxsb,d}.tileInfo` with real
    `TileInfo` instances, allocate `vgprTiles` from the writer's
    vgprPool, and run `allocOffsetRegisters` so the tail GR/LR
    re-issue has the `sharedVgprGROffset` / `sharedVgprLROffset` /
    `localSubtilesRegister` state that production sets up in
    `kernelBodySubtile`. Real (non-mock) `regList.indices` are
    required so the lane-mask `v_cndmask_b32` and MFMA emits receive
    concrete VGPR operands.
    """
    from Tensile.Components.Subtile.Kernel import (
        TileInfo, AB_B16, AB_B4, MXSA_B4, MXSB_B4, CD_F32,
    )

    fp4 = bool(kernel["ProblemType"].get("MXBlockA", 0) > 0)
    abGeo = AB_B4 if fp4 else AB_B16

    tiA = TileInfo(abGeo, 'A', None, kernel)
    tiB = TileInfo(abGeo, 'B', None, kernel)
    tiA.allocVgprTileRegisters_legacy(kwa, kernel)
    tiB.allocVgprTileRegisters_legacy(kwa, kernel)
    tiA.allocOffsetRegisters(kwa, kernel)
    tiB.allocOffsetRegisters(kwa, kernel)
    kwa.states.a.tileInfo = tiA
    kwa.states.b.tileInfo = tiB

    if fp4:
        tiMXSA = TileInfo(MXSA_B4, 'MXSA', None, kernel)
        tiMXSB = TileInfo(MXSB_B4, 'MXSB', None, kernel)
        tiMXSA.allocVgprTileRegisters_legacy(kwa, kernel)
        tiMXSB.allocVgprTileRegisters_legacy(kwa, kernel)
        tiMXSA.allocOffsetRegisters(kwa, kernel)
        tiMXSB.allocOffsetRegisters(kwa, kernel)
        kwa.states.mxsa.tileInfo = tiMXSA
        kwa.states.mxsb.tileInfo = tiMXSB

    tiD = TileInfo(CD_F32, 'D', None, kernel)
    tiD.allocVgprTileRegisters_legacy(kwa, kernel)
    kwa.states.d.tileInfo = tiD


def wrap_with_skiptoend(module) -> str:
    """Append a synthetic `SkipToEnd:` label after `module` and return
    the flat asm string.

    Simulates the surrounding emit order: in production the
    `kernelBodySubtile` call-site places the scaffold's output between
    `mainLoop()` and the `SkipToEnd:` join-point label that
    `LogicalScheduler.emitAllLoops` emits at the end of the loop
    block. Tests that assert `tail-before-SkipToEnd` ordering depend on
    this synthetic terminator to verify the scaffold leaves SkipToEnd
    reachable.
    """
    from rocisa.code import Module as _Module, Label as _Label

    wrapped = _Module("scaffold_with_skiptoend")
    wrapped.add(module)
    wrapped.add(_Label("SkipToEnd", ""))
    return str(wrapped)
