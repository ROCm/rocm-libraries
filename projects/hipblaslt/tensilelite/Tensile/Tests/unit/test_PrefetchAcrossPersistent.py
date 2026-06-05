################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
################################################################################

from contextlib import contextmanager
from types import SimpleNamespace

from rocisa.code import Module
from rocisa.enum import RegisterType
from rocisa.register import RegisterPool

from Tensile.KernelWriter import KernelWriter
import Tensile.KernelWriterAssembly as kwa_module
from Tensile.Components.StreamK import StreamKTwoTileDPFirst
from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.Common.RequiredParameters import getRequiredParametersMin
from Tensile.Common.ValidParameters import validParameters
from Tensile.Contractions import SizeMapping
from Tensile.SolutionStructs.Solution import validateParameterTypes


def _module_with_comment(name, comment):
    module = Module(name)
    module.addComment0(comment)
    return module


def _tensor_module(name, comment, tensor_parameters):
    tc = tensor_parameters["tensorChar"]
    return _module_with_comment("%s_%s" % (name, tc), "%s %s" % (comment, tc))


class _ClassicPapWriter:
    def __init__(self, *, version=(9, 5, 0), use64b_shadow=False, use64b_shadow_mx=False):
        self.states = SimpleNamespace(
            a=SimpleNamespace(numVgprGlobalReadOffsets=2),
            b=SimpleNamespace(numVgprGlobalReadOffsets=2),
            ldsTensorTokenIdx=0,
            memTokenLdsBuffer0=0,
            memTokenLdsBuffer1=1,
            staggerUCode=False,
            unrollIdx=0,
            use64bShadowLimit=use64b_shadow,
            use64bShadowLimitMX=use64b_shadow_mx,
            version=version,
        )
        self._next_tmp_sgpr = 100
        self.vgprPool = _TrackingRegisterPool(RegisterType.Vgpr)
        self.startVgprGlobalReadOffsetA = 200
        self.startVgprGlobalReadOffsetB = 210

    @contextmanager
    def allocTmpSgpr(self, size, alignment=1, tag=""):
        base = self._next_tmp_sgpr
        self._next_tmp_sgpr += size + alignment
        yield SimpleNamespace(idx=base, size=size)

    def isSwapGlobalReadOrderForDtvOrDtl(self, kernel, prefetch1=False):
        return False

    def isPrefetchAcrossPersistentEnabled(self, kernel):
        return True

    def openSumAtLeastUnroll(self, kernel, prefetch=False, isOptNLL=True):
        return _module_with_comment("openSumAtLeastUnroll", "unit: open sum")

    def declareStaggerParms(self, kernel):
        return _module_with_comment("declareStaggerParms", "unit: declare stagger")

    def lwaTileAssignment(self, kernel, tensor_parameters):
        return _tensor_module("lwaTileAssignment", "unit: LWA tile", tensor_parameters)

    def graTileAssignment(self, kernel, tensor_parameters):
        return _tensor_module("graTileAssignment", "unit: tile assignment", tensor_parameters)

    def graUnrollAssignment(self, kernel, tensor_parameters):
        return _tensor_module("graUnrollAssignment", "unit: unroll assignment", tensor_parameters)

    def graTileOffsets(self, kernel, tensor_parameters):
        return _tensor_module("graTileOffsets", "unit: tile offsets", tensor_parameters)

    def graUnrollOffsets(self, kernel, tensor_parameters):
        return _tensor_module("graUnrollOffsets", "unit: unroll offsets", tensor_parameters)

    def graShift(self, kernel, tensor_parameters):
        return _tensor_module("graShift", "unit: shift", tensor_parameters)

    def graAddresses(self, kernel, tensor_parameters):
        return _tensor_module("graAddresses", "unit: GRA", tensor_parameters)

    def graFinalOffsets(self, kernel, tensor_parameters):
        return _tensor_module("graFinalOffsets", "unit: final offsets", tensor_parameters)

    def calculateStagger(self, kernel, tensor_parameters):
        return _tensor_module("calculateStagger", "unit: stagger", tensor_parameters)

    def directToLdsM0Update(self, kernel, offset, tensor_parameters, skipWait=False):
        return _tensor_module(
            "directToLdsM0Update",
            "unit: M0 %s skipWait=%s" % (tensor_parameters["tensorChar"], skipWait),
            tensor_parameters,
        )

    def globalReadDo(self, kernel, offset, tensor_parameters):
        return _tensor_module("globalReadDo", "unit: GR", tensor_parameters)

    def papDtlSaveLdsBank(self, kernel, tensor_parameters_a, tensor_parameters_b):
        return _module_with_comment("papDtlSaveLdsBank", "unit: save DTL LDS bank")


_ClassicPapWriter.setupPrefetchAcrossPersistentLoads = KernelWriter.setupPrefetchAcrossPersistentLoads


class _TrackingRegisterPool(RegisterPool):
    def __init__(self, register_type):
        super().__init__(0, register_type, defaultPreventOverflow=False, printRP=False)
        self.checked_out = []
        self.checked_in = []

    def checkOutAligned(self, size, alignment=1, tag="", *args, **kwargs):
        base = super().checkOutAligned(size, alignment, tag, *args, **kwargs)
        self.checked_out.append((base, size, tag))
        return base

    def checkIn(self, vgpr):
        super().checkIn(vgpr)
        self.checked_in.append(vgpr)


class _StubLabels:
    def __init__(self):
        self._count = 0

    def getNameInc(self, name):
        self._count += 1
        return "%s_%u" % (name, self._count)


class _ClassicPapWrapperWriter:
    def __init__(self):
        self.labels = _StubLabels()
        self.states = SimpleNamespace(unrollIdx=0)
        self.vgprPool = _TrackingRegisterPool(RegisterType.Vgpr)

    def isPrefetchAcrossPersistentEnabled(self, kernel):
        return True

    @contextmanager
    def allocPapTileIdentitySgprs(self, kernel):
        yield {
            "WorkGroup0": 100,
            "WorkGroup1": 101,
            "WorkGroup2": 102,
            "StreamKLocalStart": 103,
            "StreamKLocalEnd": 104,
        }

    def papCheckpointCurrentTileIdentity(self, kernel, prev_tile):
        return _module_with_comment("papCheckpointCurrentTileIdentity", "unit: checkpoint tile")

    def loopCounterName(self, kernel, loop_idx):
        return "LoopCounterL"

    def calculateLoopNumIter(self, kernel, tpa, tpb, loop_idx):
        return _module_with_comment("calculateLoopNumIter", "unit: calculate loop num iter")

    def setupPrefetchAcrossPersistentLoads(self, kernel, tpa, tpb, isOptNLL=True):
        return _module_with_comment("setupPrefetchAcrossPersistentLoads", "unit: setup PAP loads")

    def papRestoreCurrentTileIdentity(self, kernel, prev_tile):
        return _module_with_comment("papRestoreCurrentTileIdentity", "unit: restore tile")


class _StubStreamK:
    def prefetchAcrossPersistentSetupNextTile(self, writer, kernel, tpa, tpb, skipLroReset=False):
        return _module_with_comment("prefetchAcrossPersistentSetupNextTile", "unit: setup next tile")


def _classic_kernel(**overrides):
    kernel = {
        "BufferLoad": True,
        "DirectToLdsA": False,
        "DirectToLdsB": False,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "PrefetchGlobalRead": 2,
        "ProblemType": {
            "MXBlockA": 0,
            "MXBlockB": 0,
            "Sparse": 0,
        },
        "EdgeType": "None",
        "GuaranteeNoPartialA": False,
        "GuaranteeNoPartialB": False,
        "UseGeneralizedNLCOneA": False,
        "UseGeneralizedNLCOneB": False,
        "_UseSgprForGRO": False,
        "enableTDMA": False,
        "enableTDMB": False,
    }
    kernel.update(overrides)
    return kernel


def _tensor_parameters(with_mx=False):
    tpa = {"tensorChar": "A", "isSwizzled": False}
    tpb = {"tensorChar": "B", "isSwizzled": False}
    if with_mx:
        tpa["MX"] = {"tensorChar": "MXSA", "isSwizzled": False}
        tpb["MX"] = {"tensorChar": "MXSB", "isSwizzled": False}
    return tpa, tpb


def _module_items(module):
    return [module.getItem(i) for i in range(module.itemsSize())]


def _module_index(items, name):
    return next(i for i, item in enumerate(items) if isinstance(item, Module) and item.name == name)


def _instruction_indices(items, instruction_type, *, dst_contains=None, src_contains=None):
    indices = []
    for i, item in enumerate(items):
        if not isinstance(item, instruction_type):
            continue
        dst = str(getattr(item, "dst", ""))
        srcs = [str(item_src) for item_src in getattr(item, "srcs", [])]
        if dst_contains is not None and dst_contains not in dst:
            continue
        if src_contains is not None and not any(src_contains in src for src in srcs):
            continue
        indices.append(i)
    return indices


def _instruction_index(items, instruction_type, dst, src):
    return next(
        i
        for i, item in enumerate(items)
        if isinstance(item, instruction_type)
        and str(item.dst) == dst
        and [str(item_src) for item_src in item.srcs] == [src]
    )


def test_pap_is_valid_solution_parameter():
    assert validParameters["PrefetchAcrossPersistent"] == [0, 1]
    assert defaultSolution["PrefetchAcrossPersistent"] == 0
    assert "PrefetchAcrossPersistent" in getRequiredParametersMin()
    assert "prefetchAcrossPersistent" in SizeMapping.StateKeys
    validateParameterTypes({"PrefetchAcrossPersistent": 1})


def test_classic_pap_primes_mx_first_pgr_group_before_marking_primed():
    writer = _ClassicPapWriter(version=(9, 5, 0))
    kernel = _classic_kernel(ProblemType={"MXBlockA": 32, "MXBlockB": 32, "Sparse": 0})
    tpa, tpb = _tensor_parameters(with_mx=True)

    module = writer.setupPrefetchAcrossPersistentLoads(kernel, tpa, tpb)
    items = _module_items(module)

    gr_a = _module_index(items, "globalReadDo_A")
    gr_mxsa = _module_index(items, "globalReadDo_MXSA")
    gr_mxsb = _module_index(items, "globalReadDo_MXSB")
    gr_b = _module_index(items, "globalReadDo_B")
    primed = _instruction_index(items, kwa_module.SMovB32, "s[sgprSkPrefetchPrimed]", "1")

    assert gr_a < gr_mxsa
    assert gr_mxsa < gr_mxsb
    assert gr_mxsb < gr_b
    assert gr_b < primed


def test_classic_pap_restores_gfx1250_shadow_limit_descriptor_encoding():
    writer = _ClassicPapWriter(version=(12, 5, 0), use64b_shadow=True)
    kernel = _classic_kernel()
    tpa, tpb = _tensor_parameters()

    module = writer.setupPrefetchAcrossPersistentLoads(kernel, tpa, tpb)
    items = _module_items(module)

    assert _instruction_indices(items, kwa_module.SMovB64, src_contains="ShadowLimitA+0")
    assert _instruction_indices(items, kwa_module.SMovB64, dst_contains="ShadowLimitA+0")
    assert len(_instruction_indices(items, kwa_module.SLShiftRightB32, dst_contains="Srd")) == 2


def test_classic_pap_shiftptr_refreshes_and_restores_gro_for_next_tile_loads():
    writer = _ClassicPapWriter()
    kernel = _classic_kernel(EdgeType="ShiftPtr")
    tpa, tpb = _tensor_parameters()

    module = writer.setupPrefetchAcrossPersistentLoads(kernel, tpa, tpb)
    items = _module_items(module)
    gro_snapshot_bases = {
        tag: base for base, _, tag in writer.vgprPool.checked_out if tag.endswith("GROSnapshot")
    }

    for tc in ("A", "B"):
        snapshot_base = gro_snapshot_bases["PAP%sGROSnapshot" % tc]
        checkpoint = _instruction_index(
            items,
            kwa_module.VMovB32,
            "v%u" % snapshot_base,
            "v[vgprGlobalReadOffset%s+0]" % tc,
        )
        refresh = _module_index(items, "lwaTileAssignment_%s" % tc)
        first_load = _module_index(items, "globalReadDo_%s" % tc)
        restore = _instruction_index(
            items,
            kwa_module.VMovB32,
            "v[vgprGlobalReadOffset%s+0]" % tc,
            "v%u" % snapshot_base,
        )

        assert checkpoint < refresh
        assert refresh < first_load
        assert first_load < restore

    gro_snapshots = [tag for _, _, tag in writer.vgprPool.checked_out if tag.endswith("GROSnapshot")]
    assert gro_snapshots == ["PAPAGROSnapshot", "PAPBGROSnapshot"]
    assert len(writer.vgprPool.checked_in) == len(gro_snapshots)


def test_classic_pap_saves_direct_to_lds_bank_state_after_priming():
    writer = _ClassicPapWriter(version=(9, 5, 0))
    kernel = _classic_kernel(DirectToLdsA=True)
    tpa, tpb = _tensor_parameters()

    module = writer.setupPrefetchAcrossPersistentLoads(kernel, tpa, tpb)
    items = _module_items(module)
    primed = _instruction_index(items, kwa_module.SMovB32, "s[sgprSkPrefetchPrimed]", "1")
    save_lds_bank = _module_index(items, "papDtlSaveLdsBank")

    assert primed < save_lds_bank
    assert writer.states.ldsTensorTokenIdx == writer.states.memTokenLdsBuffer1


def test_classic_pap_checkpoints_loop_counters_in_vgprs_around_next_tile_recount():
    original_find = kwa_module.Component.StreamK.find
    kwa_module.Component.StreamK.find = lambda writer: _StubStreamK()
    try:
        writer = _ClassicPapWrapperWriter()
        kernel = _classic_kernel(
            PrefetchAcrossPersistent=1,
            StreamK=3,
            SpaceFillingAlgo=[],
            ProblemType={"MXBlockA": 0, "MXBlockB": 0, "Sparse": 0},
        )
        tpa, tpb = _tensor_parameters()

        module = kwa_module.KernelWriterAssembly.prefetchAcrossPersistent(writer, kernel, tpa, tpb)
        items = _module_items(module)

        loop_vgpr = next(base for base, _, tag in writer.vgprPool.checked_out if tag == "PAP loop counters")
        orig_loop_vgpr = loop_vgpr + 1
        loop_checkpoint = _instruction_index(items, kwa_module.VMovB32, "v%u" % loop_vgpr, "s[sgprLoopCounterL]")
        orig_loop_checkpoint = _instruction_index(items, kwa_module.VMovB32, "v%u" % orig_loop_vgpr, "s[sgprOrigLoopCounter]")
        calculate_loop_num_iter = _module_index(items, "calculateLoopNumIter")
        setup_pap_loads = _module_index(items, "setupPrefetchAcrossPersistentLoads")
        loop_restore = _instruction_index(items, kwa_module.VReadfirstlaneB32, "s[sgprLoopCounterL]", "v%u" % loop_vgpr)
        orig_loop_restore = _instruction_index(items, kwa_module.VReadfirstlaneB32, "s[sgprOrigLoopCounter]", "v%u" % orig_loop_vgpr)

        assert loop_checkpoint < calculate_loop_num_iter
        assert orig_loop_checkpoint < calculate_loop_num_iter
        assert calculate_loop_num_iter < setup_pap_loads
        assert setup_pap_loads < loop_restore
        assert loop_restore < orig_loop_restore
        assert writer.vgprPool.checked_in == [loop_vgpr]
    finally:
        kwa_module.Component.StreamK.find = original_find


def test_classic_pap_can_skip_internal_barrier_after_caller_sync():
    original_find = kwa_module.Component.StreamK.find
    kwa_module.Component.StreamK.find = lambda writer: _StubStreamK()
    try:
        writer = _ClassicPapWrapperWriter()
        kernel = _classic_kernel(
            PrefetchAcrossPersistent=1,
            StreamK=3,
            SpaceFillingAlgo=[],
            ProblemType={"MXBlockA": 0, "MXBlockB": 0, "Sparse": 0},
        )
        tpa, tpb = _tensor_parameters()

        with_barrier = kwa_module.KernelWriterAssembly.prefetchAcrossPersistent(
            writer, kernel, tpa, tpb, skipBarrier=False)
        without_barrier = kwa_module.KernelWriterAssembly.prefetchAcrossPersistent(
            writer, kernel, tpa, tpb, skipBarrier=True)

        assert _instruction_indices(_module_items(with_barrier), kwa_module.SBarrier)
        assert not _instruction_indices(_module_items(without_barrier), kwa_module.SBarrier)
    finally:
        kwa_module.Component.StreamK.find = original_find


def test_streamk_pap_next_tile_setup_applies_default_wgm_remap():
    import Tensile.Components.WorkGroupMappingAlgos as wgm_algos

    original_default_wgm = wgm_algos.DefaultWGM
    try:
        wgm_algos.DefaultWGM = lambda writer, kernel, sgpr_wgm: _module_with_comment(
            "DefaultWGM", "unit: default WGM remap"
        )

        streamk = StreamKTwoTileDPFirst()
        streamk.skTileIndex = lambda writer, kernel, s_tmp, tpa, tpb, skipLroReset=False: _module_with_comment(
            "skTileIndex", "unit: tile index"
        )
        streamk.skIndexToWG = lambda writer, kernel, s_tmp: _module_with_comment(
            "skIndexToWG", "unit: index to WG"
        )

        writer = SimpleNamespace(
            sgprPool=RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False, printRP=False),
            states=SimpleNamespace(WGMTransformLevels=-1),
        )
        kernel = {"SpaceFillingAlgo": []}

        module = streamk.prefetchAcrossPersistentSetupNextTile(writer, kernel, {"tensorChar": "A"}, {"tensorChar": "B"})
        items = _module_items(module)

        tile_index = _module_index(items, "skTileIndex")
        index_to_wg = _module_index(items, "skIndexToWG")
        default_wgm = _module_index(items, "DefaultWGM")
        assert tile_index < index_to_wg
        assert index_to_wg < default_wgm
    finally:
        wgm_algos.DefaultWGM = original_default_wgm


def test_streamk_pap_next_tile_setup_applies_space_filling_wgm_remap():
    import Tensile.Components.WorkGroupMappingAlgos as wgm_algos

    original_space_filling = wgm_algos.SpaceFillingCurveWalk
    try:
        wgm_algos.SpaceFillingCurveWalk = lambda writer, kernel, sgpr_wgm: _module_with_comment(
            "SpaceFillingCurveWalk", "unit: space-filling WGM remap"
        )

        streamk = StreamKTwoTileDPFirst()
        streamk.skTileIndex = lambda writer, kernel, s_tmp, tpa, tpb, skipLroReset=False: _module_with_comment(
            "skTileIndex", "unit: tile index"
        )
        streamk.skIndexToWG = lambda writer, kernel, s_tmp: _module_with_comment(
            "skIndexToWG", "unit: index to WG"
        )

        writer = SimpleNamespace(
            sgprPool=RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False, printRP=False),
            states=SimpleNamespace(WGMTransformLevels=-1),
        )
        kernel = {"SpaceFillingAlgo": [{"foo": "bar"}]}

        module = streamk.prefetchAcrossPersistentSetupNextTile(writer, kernel, {"tensorChar": "A"}, {"tensorChar": "B"})
        items = _module_items(module)

        tile_index = _module_index(items, "skTileIndex")
        index_to_wg = _module_index(items, "skIndexToWG")
        space_filling_wgm = _module_index(items, "SpaceFillingCurveWalk")
        assert tile_index < index_to_wg
        assert index_to_wg < space_filling_wgm
        assert writer.states.WGMTransformLevels == 1
    finally:
        wgm_algos.SpaceFillingCurveWalk = original_space_filling
