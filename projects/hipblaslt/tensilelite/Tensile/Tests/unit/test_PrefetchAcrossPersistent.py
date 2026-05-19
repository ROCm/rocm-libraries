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

from Tensile.KernelWriter import KernelWriter
import Tensile.KernelWriterAssembly as kwa_module
from Tensile.Components.StreamK import StreamKTwoTileDPFirst


def _module_with_comment(name, comment):
    module = Module(name)
    module.addComment0(comment)
    return module


class _ClassicPapWriter:
    def __init__(self, *, version=(9, 5, 0), use64b_shadow=False, use64b_shadow_mx=False):
        self.states = SimpleNamespace(
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

    @contextmanager
    def allocTmpSgpr(self, size, alignment=1, tag=""):
        base = self._next_tmp_sgpr
        self._next_tmp_sgpr += size + alignment
        yield SimpleNamespace(idx=base, size=size)

    def isSwapGlobalReadOrderForDtvOrDtl(self, kernel, prefetch1=False):
        return False

    def openSumAtLeastUnroll(self, kernel, prefetch=False, isOptNLL=True):
        return _module_with_comment("openSumAtLeastUnroll", "unit: open sum")

    def declareStaggerParms(self, kernel):
        return _module_with_comment("declareStaggerParms", "unit: declare stagger")

    def graAddresses(self, kernel, tensor_parameters):
        return _module_with_comment("graAddresses", "unit: GRA %s" % tensor_parameters["tensorChar"])

    def calculateStagger(self, kernel, tensor_parameters):
        return _module_with_comment("calculateStagger", "unit: stagger %s" % tensor_parameters["tensorChar"])

    def directToLdsM0Update(self, kernel, offset, tensor_parameters, skipWait=False):
        return _module_with_comment(
            "directToLdsM0Update",
            "unit: M0 %s skipWait=%s" % (tensor_parameters["tensorChar"], skipWait),
        )

    def globalReadDo(self, kernel, offset, tensor_parameters):
        return _module_with_comment("globalReadDo", "unit: GR %s" % tensor_parameters["tensorChar"])

    def papDtlSaveLdsBank(self, kernel, tensor_parameters_a, tensor_parameters_b):
        return _module_with_comment("papDtlSaveLdsBank", "unit: save DTL LDS bank")


_ClassicPapWriter.setupPrefetchAcrossPersistentLoads = KernelWriter.setupPrefetchAcrossPersistentLoads


class _StubSgprPool:
    def checkOutAligned(self, *args, **kwargs):
        return 40

    def checkIn(self, *args, **kwargs):
        pass


class _StubVgprPool:
    def __init__(self):
        self.checked_in = []

    def checkOutAligned(self, *args, **kwargs):
        return 70

    def checkIn(self, vgpr):
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
        self.vgprPool = _StubVgprPool()

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
        "enableTDMA": False,
        "enableTDMB": False,
    }
    kernel.update(overrides)
    return kernel


def _tensor_parameters(with_mx=False):
    tpa = {"tensorChar": "A"}
    tpb = {"tensorChar": "B"}
    if with_mx:
        tpa["MX"] = {"tensorChar": "MXSA"}
        tpb["MX"] = {"tensorChar": "MXSB"}
    return tpa, tpb


def _module_items(module):
    return [module.getItem(i) for i in range(module.itemsSize())]


def _module_index(items, name):
    return next(i for i, item in enumerate(items) if isinstance(item, Module) and item.name == name)


def _instruction_index(items, instruction_type, dst, src):
    return next(
        i
        for i, item in enumerate(items)
        if isinstance(item, instruction_type)
        and str(item.dst) == dst
        and [str(item_src) for item_src in item.srcs] == [src]
    )


def test_classic_pap_primes_mx_first_pgr_group_before_marking_primed():
    writer = _ClassicPapWriter(version=(9, 5, 0))
    kernel = _classic_kernel(ProblemType={"MXBlockA": 32, "MXBlockB": 32, "Sparse": 0})
    tpa, tpb = _tensor_parameters(with_mx=True)

    asm = str(writer.setupPrefetchAcrossPersistentLoads(kernel, tpa, tpb))

    assert "unit: GR A" in asm
    assert "unit: GR MXSA" in asm
    assert "unit: GR MXSB" in asm
    assert "unit: GR B" in asm
    assert asm.index("unit: GR A") < asm.index("unit: GR MXSA")
    assert asm.index("unit: GR MXSA") < asm.index("unit: GR MXSB")
    assert asm.index("unit: GR MXSB") < asm.index("unit: GR B")
    assert asm.index("unit: GR B") < asm.index("first PGR group for next persistent iter prefetched")


def test_classic_pap_restores_gfx1250_shadow_limit_descriptor_encoding():
    writer = _ClassicPapWriter(version=(12, 5, 0), use64b_shadow=True, use64b_shadow_mx=True)
    kernel = _classic_kernel(ProblemType={"MXBlockA": 32, "MXBlockB": 32, "Sparse": 0})
    tpa, tpb = _tensor_parameters(with_mx=True)

    asm = str(writer.setupPrefetchAcrossPersistentLoads(kernel, tpa, tpb))

    assert "checkpoint ShadowLimitA" in asm
    assert "restore ShadowLimitA" in asm
    assert "checkpoint ShadowLimitMXSA" in asm
    assert "restore ShadowLimitMXSA" in asm
    assert asm.count("Shift num records for gfx125x") == 4


def test_classic_pap_saves_direct_to_lds_bank_state_after_priming():
    writer = _ClassicPapWriter(version=(9, 5, 0))
    kernel = _classic_kernel(DirectToLdsA=True)
    tpa, tpb = _tensor_parameters()

    asm = str(writer.setupPrefetchAcrossPersistentLoads(kernel, tpa, tpb))

    assert "first PGR group for next persistent iter prefetched" in asm
    assert "unit: save DTL LDS bank" in asm
    assert asm.index("first PGR group for next persistent iter prefetched") < asm.index("unit: save DTL LDS bank")
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

        loop_checkpoint = _instruction_index(items, kwa_module.VMovB32, "v70", "s[sgprLoopCounterL]")
        orig_loop_checkpoint = _instruction_index(items, kwa_module.VMovB32, "v71", "s[sgprOrigLoopCounter]")
        calculate_loop_num_iter = _module_index(items, "calculateLoopNumIter")
        setup_pap_loads = _module_index(items, "setupPrefetchAcrossPersistentLoads")
        loop_restore = _instruction_index(items, kwa_module.VReadfirstlaneB32, "s[sgprLoopCounterL]", "v70")
        orig_loop_restore = _instruction_index(items, kwa_module.VReadfirstlaneB32, "s[sgprOrigLoopCounter]", "v71")

        assert loop_checkpoint < calculate_loop_num_iter
        assert orig_loop_checkpoint < calculate_loop_num_iter
        assert calculate_loop_num_iter < setup_pap_loads
        assert setup_pap_loads < loop_restore
        assert loop_restore < orig_loop_restore
        assert writer.vgprPool.checked_in == [70]
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
            sgprPool=_StubSgprPool(),
            states=SimpleNamespace(WGMTransformLevels=-1),
        )
        kernel = {"SpaceFillingAlgo": []}

        asm = str(streamk.prefetchAcrossPersistentSetupNextTile(writer, kernel, {"tensorChar": "A"}, {"tensorChar": "B"}))

        assert "unit: tile index" in asm
        assert "unit: index to WG" in asm
        assert "unit: default WGM remap" in asm
        assert asm.index("unit: index to WG") < asm.index("unit: default WGM remap")
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
            sgprPool=_StubSgprPool(),
            states=SimpleNamespace(WGMTransformLevels=-1),
        )
        kernel = {"SpaceFillingAlgo": [{"foo": "bar"}]}

        asm = str(streamk.prefetchAcrossPersistentSetupNextTile(writer, kernel, {"tensorChar": "A"}, {"tensorChar": "B"}))

        assert "unit: tile index" in asm
        assert "unit: index to WG" in asm
        assert "unit: space-filling WGM remap" in asm
        assert asm.index("unit: index to WG") < asm.index("unit: space-filling WGM remap")
        assert writer.states.WGMTransformLevels == 1
    finally:
        wgm_algos.SpaceFillingCurveWalk = original_space_filling
