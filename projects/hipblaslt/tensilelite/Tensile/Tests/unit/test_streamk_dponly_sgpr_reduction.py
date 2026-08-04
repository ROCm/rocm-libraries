# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the StreamKForceDPOnly SGPR-reduction changes (AIHPBLAS-4145).

StreamKForceDPOnly is the SK3 DP-first path on gfx1250. Because every workgroup
processes complete tiles and the reduction is always the single-kernel tree path,
a family of StreamK SGPRs become dead / compile-time constants:

  * ``StreamKLocalStart`` == 0 and ``StreamKLocalEnd`` == ItersPerTile always,
    so both persistent SGPRs are dropped and their readers are constant-folded.
  * ``AddressWS`` / ``AddressFlags`` / ``SrdWS`` (the workspace + synchronizer-flag
    kernarg pointers / SRD) are never dereferenced, so they are dropped from the
    kernarg SGPR define, the ``.kd`` signature metadata, and the host kernarg
    builder (ContractionSolution.cpp, host-side, not unit-tested here).

Each folded reader is gated on ``kernel["StreamKForceDPOnly"]``. These tests pin
that behaviour by driving the individual emitter methods with both DP-only and
non-DP-only kernels and asserting the DP-only path drops the dead SGPR reads while
the non-DP-only path is unchanged.

The component-level harness (SimpleNamespace / mock-writer + rocisa RegisterPool,
introspecting emitted items) is reused from ``test_PrefetchAcrossPersistent``; the
real-Solution gfx1250 fixtures (auto-skip without an amdclang++ that can target
gfx1250) are reused from ``test_prefetchgl2_streamk_guard``.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from rocisa.code import Label, Module
from rocisa.container import sgpr, vgpr
from rocisa.enum import RegisterType
from rocisa.instruction import (
    SBranch,
    SCmpEQU32,
    SCmpEQU64,
    SCmpGeU32,
    SMovB32,
    SMulI32,
    SSubU32,
    VMovB32,
)
from rocisa.register import RegisterPool

import Tensile.KernelWriter as kw_module
import Tensile.KernelWriterAssembly as kwa_module
from Tensile.Common.DataType import DataType
from Tensile.Components.StreamK import StreamKTwoTileDPFirst
from Tensile.Components.Subtile.SubtileGREmit import tdmApplyStreamKOffsetSubtile
from Tensile.Contractions import SizeMapping

# Reuse the established component-level harness helpers rather than reinventing
# them (see test_segment_interleave_state / cms_validation_base for the in-repo
# cross-test-module import idiom).
from test_PrefetchAcrossPersistent import (
    _StubLabels,
    _StubStreamK,
    _instruction_indices,
    _module_items,
    _prefetch_across_persistent,
    _tensor_parameters,
)

# Real-Solution gfx1250 toolchain fixtures. Imported so pytest resolves them as
# fixtures in this module (they auto-skip when amdclang++ cannot target gfx1250).
from test_prefetchgl2_streamk_guard import (  # noqa: F401  (fixtures used by name)
    _gp_gfx1250,
    assembler,
    gfx1250_iim,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Shared minimal mock writer for the StreamK component / TDM emitter methods.
# The DP-only branches of the methods under test early-return or emit a tiny
# constant sequence, so only a small surface of the real writer is exercised.
# ---------------------------------------------------------------------------
class _SimpleSgprPool:
    def __init__(self):
        self._next = 40
        self.checked_out = []
        self.checked_in = []

    def checkOut(self, size, tag="", *args, **kwargs):
        base = self._next
        self._next += size
        self.checked_out.append((base, size, tag))
        return base

    def checkIn(self, idx):
        self.checked_in.append(idx)


class _SKWriter:
    """Mock writer sufficient to drive the StreamK.py Common methods and the
    KernelWriterAssembly TDM StreamK-offset helpers, for both DP-only (constant
    fold / early return) and non-DP-only (partial-tile) paths."""

    def __init__(self):
        self.labels = _StubLabels()
        self.sgprPool = _SimpleSgprPool()
        self.states = SimpleNamespace(
            unrollIdx=0,
            indexChars=["I", "J", "K", "L", "M"],
            skConstVgprs={},
            a=SimpleNamespace(tileInfo=SimpleNamespace(depthUBytes=256)),
            b=SimpleNamespace(tileInfo=SimpleNamespace(depthUBytes=256)),
        )
        self._next_tmp = 90

    @contextmanager
    def allocTmpSgpr(self, size, alignment=1, tag=""):
        base = self._next_tmp
        self._next_tmp += size + alignment
        yield SimpleNamespace(idx=base, size=size)

    # StreamK constant SGPRs are kept in-place (returns the symbolic name) in this
    # mock; isStreamKConstantsToVgprEnabled=False so no VGPR readfirstlane fires.
    def acquireStreamKConstSgpr(self, kernel, name):
        return name

    def releaseStreamKConstSgpr(self, nameOrIdx):
        pass

    def isStreamKConstantsToVgprEnabled(self, kernel):
        return False

    def strideRef(self, tc, idx):
        return 1

    def s_mul_u64_u32(self, *args, **kwargs):
        return Module("s_mul_u64_u32 stub")


def _sk_common_kernel(dp_only):
    return {
        "StreamKForceDPOnly": 1 if dp_only else 0,
        "NoTailLoop": True,
        "DepthU": 64,
        "ProblemType": {
            "Sparse": 0,
            "IndicesSummation": [0],
            "ComputeDataType": DataType("s"),
        },
    }


def _sk():
    return StreamKTwoTileDPFirst()


# ---------------------------------------------------------------------------
# 1. classic PAP: the AddressFlags "parallel reduction: skip PAP" compare is
#    folded out under DP-only; the StreamKIter >= StreamKIterEnd check remains.
# ---------------------------------------------------------------------------
def test_pap_addressflags_compare_folded_under_dp_only(monkeypatch):
    _, dp_items = _prefetch_across_persistent(monkeypatch, StreamKForceDPOnly=1)
    _, nodp_items = _prefetch_across_persistent(monkeypatch, StreamKForceDPOnly=0)

    # DP-only: no AddressFlags synchronizer compare ...
    assert not _instruction_indices(dp_items, SCmpEQU64, src_contains="AddressFlags")
    # ... but the last-tile StreamKIter/StreamKIterEnd check is still emitted.
    assert _instruction_indices(dp_items, SCmpGeU32, src_contains="StreamKIter")

    # non-DP-only: the AddressFlags compare is present (path unchanged).
    assert _instruction_indices(nodp_items, SCmpEQU64, src_contains="AddressFlags")
    assert _instruction_indices(nodp_items, SCmpGeU32, src_contains="StreamKIter")


# ---------------------------------------------------------------------------
# 2. Subtile PAP (KernelWriter.prefetchAcrossPersistentSubtile): same fold.
# ---------------------------------------------------------------------------
class _SubtilePapWriter:
    def __init__(self):
        self.labels = _StubLabels()
        self.vgprPool = SimpleNamespace(
            checkOutAligned=lambda *a, **k: 300,
            checkIn=lambda *a, **k: None,
        )

    def isPrefetchAcrossPersistentEnabled(self, kernel):
        return True

    def papTileIdentityNames(self, kernel):
        return []

    def papCheckpointCurrentTileIdentityVgprs(self, kernel, prevTile):
        return Module("papCheckpointCurrentTileIdentityVgprs")

    def papRestoreCurrentTileIdentityVgprs(self, kernel, prevTile):
        return Module("papRestoreCurrentTileIdentityVgprs")

    def setupPrefetchAcrossPersistentSubtileLoads(self, kernel, tpa, tpb, preloopGrModule=None):
        return Module("setupPrefetchAcrossPersistentSubtileLoads")


def _subtile_pap_items(monkeypatch, dp_only):
    monkeypatch.setattr(kw_module.Component.StreamK, "find", lambda writer: _StubStreamK())
    writer = _SubtilePapWriter()
    kernel = {"UseSubtileImpl": True, "StreamKForceDPOnly": 1 if dp_only else 0}
    tpa, tpb = _tensor_parameters()
    module = kw_module.KernelWriter.prefetchAcrossPersistentSubtile(writer, kernel, tpa, tpb)
    return _module_items(module)


def test_subtile_pap_addressflags_compare_folded_under_dp_only(monkeypatch):
    dp_items = _subtile_pap_items(monkeypatch, dp_only=True)
    nodp_items = _subtile_pap_items(monkeypatch, dp_only=False)

    assert not _instruction_indices(dp_items, SCmpEQU64, src_contains="AddressFlags")
    assert _instruction_indices(dp_items, SCmpGeU32, src_contains="StreamKIter")

    assert _instruction_indices(nodp_items, SCmpEQU64, src_contains="AddressFlags")
    assert _instruction_indices(nodp_items, SCmpGeU32, src_contains="StreamKIter")


# ---------------------------------------------------------------------------
# 3. general-batched flag check (StreamK.stridedBatchOrGeneralBatch): DP-only
#    folds the AddressFlags synchronizer compare into an unconditional branch to
#    the general-batched target.
# ---------------------------------------------------------------------------
def _strided_or_general_items(dp_only):
    strided = Label("StridedBatchedGemmLoad", "")
    general = Label("GeneralBatchedGemmLoad", "")
    kernel = {
        "StreamKForceDPOnly": 1 if dp_only else 0,
        "ProblemType": {"SupportUserArgs": True},
    }
    module = _sk().stridedBatchOrGeneralBatch(strided, general, kernel)
    return _module_items(module), general.getLabelName()


def test_general_batched_flag_check_folded_under_dp_only():
    dp_items, general_name = _strided_or_general_items(dp_only=True)
    nodp_items, _ = _strided_or_general_items(dp_only=False)

    # DP-only: no AddressFlags compare, and an unconditional branch to general.
    assert not _instruction_indices(dp_items, SCmpEQU64, src_contains="AddressFlags")
    assert any(isinstance(i, SBranch) and general_name in str(i) for i in dp_items)
    # The ArgType==3 dispatch compare is retained in both variants.
    assert _instruction_indices(dp_items, SCmpEQU32, src_contains="ArgType")

    # non-DP-only: the AddressFlags synchronizer compare is present.
    assert _instruction_indices(nodp_items, SCmpEQU64, src_contains="AddressFlags")
    assert _instruction_indices(nodp_items, SCmpEQU32, src_contains="ArgType")


# ---------------------------------------------------------------------------
# 4. calculateLoopNumIterCommon: DP-only loop count folds to ItersPerTile (an
#    SMovB32) instead of StreamKLocalEnd - StreamKLocalStart (an SSubU32); and
#    computeLoadSrd / declareStaggerParms / tailLoopNumIter become empty.
# ---------------------------------------------------------------------------
def _calc_loop_num_iter_items(dp_only):
    writer = _SKWriter()
    kernel = _sk_common_kernel(dp_only)
    module = _sk().calculateLoopNumIterCommon(
        writer, kernel, "LoopCounterL", 0, SimpleNamespace(idx=50)
    )
    return _module_items(module)


def test_calculate_loop_num_iter_folds_to_iterspertile_under_dp_only():
    dp_items = _calc_loop_num_iter_items(dp_only=True)
    nodp_items = _calc_loop_num_iter_items(dp_only=False)

    # DP-only: loop count comes from an SMovB32 (ItersPerTile), never from an
    # SSubU32 reading the (now-absent) StreamKLocalStart/End SGPRs.
    assert not _instruction_indices(dp_items, SSubU32, src_contains="StreamKLocalStart")
    assert not _instruction_indices(dp_items, SSubU32, src_contains="StreamKLocalEnd")
    assert _instruction_indices(dp_items, SMovB32, dst_contains="LoopCounterL")

    # non-DP-only: loop count = StreamKLocalEnd - StreamKLocalStart.
    ssub = _instruction_indices(nodp_items, SSubU32, dst_contains="LoopCounterL")
    assert ssub
    ssub_item = nodp_items[ssub[0]]
    srcs = [str(s) for s in ssub_item.srcs]
    assert any("StreamKLocalEnd" in s for s in srcs)
    assert any("StreamKLocalStart" in s for s in srcs)


def test_dp_only_streamk_common_helpers_emit_empty_modules():
    writer = _SKWriter()
    kernel = _sk_common_kernel(dp_only=True)
    tp = {"tensorChar": "A", "bpe": 2}

    assert _sk().computeLoadSrdCommon(writer, kernel, tp, 60).itemsSize() == 0
    assert _sk().declareStaggerParmsCommon(writer, kernel).itemsSize() == 0
    assert _sk().tailLoopNumIterCommon(writer, kernel, sgpr("LoopCounterL")).itemsSize() == 0


# ---------------------------------------------------------------------------
# 5. graAddressesCommon: DP-only emits only VMovB32 of Address{tc}; no
#    StreamKLocalStart-scaled partial-tile offset.
# ---------------------------------------------------------------------------
def _gra_addresses_items(dp_only):
    writer = _SKWriter()
    kernel = _sk_common_kernel(dp_only)
    tp = {"tensorChar": "A", "bpe": 2}
    module = _sk().graAddressesCommon(writer, kernel, tp, 10)
    return _module_items(module)


def test_gra_addresses_dp_only_moves_only_base_address():
    dp_items = _gra_addresses_items(dp_only=True)
    nodp_items = _gra_addresses_items(dp_only=False)

    # DP-only: two VMovB32 of AddressA lo/hi, nothing scaled by StreamKLocalStart.
    assert len(_instruction_indices(dp_items, VMovB32, src_contains="AddressA")) == 2
    assert not any("StreamKLocalStart" in str(i) for i in dp_items)

    # non-DP-only: the partial-tile start offset (StreamKLocalStart * depthU) is
    # computed via SMulI32.
    assert _instruction_indices(nodp_items, SMulI32, src_contains="StreamKLocalStart")


# ---------------------------------------------------------------------------
# 6. TDM StreamK-offset appliers: DP-only makes the wave-separated K-offset and
#    the subtile K-offset no-ops, and the tail applier reads no StreamKLocalEnd.
# ---------------------------------------------------------------------------
def test_tdm_apply_streamk_offset_wave_separated_noop_under_dp_only():
    writer = _SKWriter()
    tpa, tpb = _tensor_parameters()

    dp_mod = kwa_module.KernelWriterAssembly.tdmApplyStreamKOffsetWaveSeparated(
        writer, {"StreamKForceDPOnly": 1}, tpa, tpb
    )
    nodp_mod = kwa_module.KernelWriterAssembly.tdmApplyStreamKOffsetWaveSeparated(
        writer, {"StreamKForceDPOnly": 0}, tpa, tpb
    )

    assert dp_mod.itemsSize() == 0
    assert _instruction_indices(_module_items(nodp_mod), SMulI32, src_contains="StreamKLocalStart")


def test_tdm_apply_streamk_offset_subtile_noop_under_dp_only():
    writer = _SKWriter()
    tp = {"tensorChar": "A"}

    dp_mod = tdmApplyStreamKOffsetSubtile(writer, {"StreamKForceDPOnly": 1}, tp)
    nodp_mod = tdmApplyStreamKOffsetSubtile(writer, {"StreamKForceDPOnly": 0}, tp)

    assert dp_mod.itemsSize() == 0
    assert _instruction_indices(_module_items(nodp_mod), SMulI32, src_contains="StreamKLocalStart")


def test_tdm_apply_streamk_tail_offset_derives_iterspertile_under_dp_only():
    writer = _SKWriter()
    tpa, tpb = _tensor_parameters()

    dp_items = _module_items(
        kwa_module.KernelWriterAssembly.tdmApplyStreamKTailOffsetWaveSeparated(
            writer, _sk_common_kernel(dp_only=True), tpa, tpb
        )
    )
    nodp_items = _module_items(
        kwa_module.KernelWriterAssembly.tdmApplyStreamKTailOffsetWaveSeparated(
            writer, _sk_common_kernel(dp_only=False), tpa, tpb
        )
    )

    # DP-only: tail index derived from ItersPerTile, never reads StreamKLocalEnd.
    assert not _instruction_indices(dp_items, SSubU32, src_contains="StreamKLocalEnd")
    assert _instruction_indices(dp_items, SSubU32, src_contains="ItersPerTile")
    # non-DP-only: tail index = StreamKLocalEnd - 1.
    assert _instruction_indices(nodp_items, SSubU32, src_contains="StreamKLocalEnd")


# ---------------------------------------------------------------------------
# 9. SizeMapping seam: streamKForceDPOnly round-trips from the solution state
#    (the host contract that keeps ContractionSolution.cpp in sync).
# ---------------------------------------------------------------------------
def _minimal_size_mapping_state():
    from test_streamk_force_dp_only import minimal_size_mapping_state

    return minimal_size_mapping_state()


def test_size_mapping_streamk_force_dp_only_round_trips():
    assert "streamKForceDPOnly" in SizeMapping.StateKeys

    state = _minimal_size_mapping_state()
    state["StreamK"] = 3
    state["StreamKForceDPOnly"] = 1
    assert SizeMapping.FromOriginalState(state).streamKForceDPOnly == 1

    state_off = _minimal_size_mapping_state()
    state_off["StreamK"] = 3
    state_off.pop("StreamKForceDPOnly", None)
    assert SizeMapping.FromOriginalState(state_off).streamKForceDPOnly == 0


# ---------------------------------------------------------------------------
# 7 + 8 + 10. Real gfx1250 SK3 kernel emit (CPU-only assembly text). Auto-skips
# when amdclang++ cannot target gfx1250 (via the gfx1250_iim fixture). Proves the
# dead workspace/flag SGPRs and the StreamK local-bound SGPRs are absent from the
# emitted assembly (SGPR defines, .kd signature metadata) under DP-only, and
# present under non-DP-only.
# ---------------------------------------------------------------------------
_DEAD_SGPR_SYMBOLS = ["AddressWS", "AddressFlags", "SrdWS", "StreamKLocalStart", "StreamKLocalEnd"]


def _sk3_gfx1250_params(gfx1250_iim, dp_only):
    """A minimal, valid F16 TN SK3 gfx1250 solution config.

    Kept intentionally SGPR-light (no MX, PrefetchAcrossPersistent=0,
    PrefetchGL2=0) so the *non*-DP-only variant, which still defines
    AddressWS/AddressFlags/SrdWS/StreamKLocalStart/StreamKLocalEnd, fits under the
    gfx1250 SGPR budget and emits cleanly -- otherwise the non-DP baseline
    overflows (which is exactly the pressure the DP-only reduction relieves)."""
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    isa = gfxToIsa("gfx1250")
    mi = [16, 16, 32, 1, 1, 2, 2, 2, 2]
    problem_type = {
        "OperationType": "GEMM",
        "DataType": "H",
        "DestDataType": "H",
        "ComputeDataType": "s",
        "HighPrecisionAccumulate": True,
        "TransposeA": True,
        "TransposeB": False,
        "UseBeta": True,
        "Batched": True,
        "StridedBatched": True,
    }
    params = {
        "ProblemType": problem_type,
        "ISA": isa,
        "MatrixInstruction": mi,
        "WorkGroup": [16, 16, 1],
        "WavefrontSize": 32,
        "DepthU": 64,
        "KernelLanguage": "Assembly",
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 1,
        "ScheduleIterAlg": 0,
        "StaggerU": 0,
        "GlobalSplitU": 0,
        "InnerUnroll": 1,
        "TransposeLDS": -1,
        "LdsPadA": -1,
        "LdsPadB": -1,
        "LdsBlockSizePerPadA": -1,
        "LdsBlockSizePerPadB": -1,
        "1LDSBuffer": 0,
        "VectorWidthA": -1,
        "VectorWidthB": -1,
        "StoreVectorWidth": -1,
        "GlobalReadVectorWidthA": -1,
        "GlobalReadVectorWidthB": -1,
        "LocalReadVectorWidth": -1,
        "SourceSwap": False,
        "ExpandPointerSwap": False,
        "GlobalSplitUAlgorithm": "MultipleBuffer",
        "StreamK": 3,
        "StreamKForceDPOnly": 1 if dp_only else 0,
        "PrefetchAcrossPersistent": 0,
        "PrefetchGL2": 0,
        "UseSubtileImpl": False,
        "StoreRemapVectorWidth": 0,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "DirectToVgprSparseMetadata": False,
        "WorkGroupMapping": 1,
        "ClusterLocalRead": 0,
    }
    params.update(
        matrixInstructionToMIParameters(
            mi, isa, params["WavefrontSize"], problem_type, params["WorkGroup"], gfx1250_iim
        )
    )
    return params


def _emit_sk3_kernel_asm(gfx1250_iim, assembler, capsys, dp_only):
    import shutil

    import rocisa
    from Tensile.Common.Types import DebugConfig
    from Tensile.KernelWriterAssembly import KernelWriterAssembly
    from Tensile.SolutionStructs.Naming import getKernelFileBase
    from Tensile.SolutionStructs.Solution import Solution
    from Tensile.TensileCreateLibrary.Run import (
        generateKernelObjectsFromSolutions,
        processKernelSource,
    )

    params = _sk3_gfx1250_params(gfx1250_iim, dp_only)
    sol = Solution(params, False, True, False, assembler, gfx1250_iim)
    capsys.readouterr()
    assert sol.get("Valid") is True, "base SK3 gfx1250 solution must derive cleanly"

    kernels = generateKernelObjectsFromSolutions([sol])
    assert kernels, "solution produced no kernels"
    kernel = kernels[0]

    isa = tuple(kernel["ISA"])
    asmpath = shutil.which("amdclang++") or "/usr/bin/amdclang++"
    ri = rocisa.rocIsa.getInstance()
    ri.init(isa, asmpath)
    ri.setKernel(isa, kernel["WavefrontSize"])

    kwa = KernelWriterAssembly(assembler, DebugConfig())
    kernel.duplicate = False
    kernel["BaseName"] = getKernelFileBase(False, kernel)
    res = processKernelSource(kwa, ri.getData(), ri.getOutputOptions(), False, kernel)
    src = res.src
    if isinstance(src, (bytes, bytearray)):
        src = src.decode(errors="replace")
    assert res.err == 0, "kernel emit returned nonzero err=%s" % res.err
    return src


def test_dp_only_kernel_asm_omits_workspace_and_local_sgpr_symbols(
    _gp_gfx1250, gfx1250_iim, assembler, capsys
):
    src = _emit_sk3_kernel_asm(gfx1250_iim, assembler, capsys, dp_only=True)
    for symbol in _DEAD_SGPR_SYMBOLS:
        assert symbol not in src, "DP-only asm unexpectedly references %s" % symbol


def test_non_dp_only_kernel_asm_retains_workspace_and_local_sgpr_symbols(
    _gp_gfx1250, gfx1250_iim, assembler, capsys
):
    src = _emit_sk3_kernel_asm(gfx1250_iim, assembler, capsys, dp_only=False)
    for symbol in _DEAD_SGPR_SYMBOLS:
        assert symbol in src, "non-DP-only asm unexpectedly missing %s" % symbol
