################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
################################################################################

from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest
from rocisa.code import Module
from rocisa.enum import RegisterType
from rocisa.register import RegisterPool

import Tensile.KernelWriter as kw_module
from Tensile.KernelWriter import KernelWriter
import Tensile.KernelWriterAssembly as kwa_module
from Tensile.Components.StreamK import StreamKTwoTileDPFirst
from Tensile.Common.GlobalParameters import defaultSolution, globalParameters
from Tensile.Common.RequiredParameters import getRequiredParametersMin
from Tensile.Common.Types import IsaInfo, IsaVersion, SemanticVersion
from Tensile.Common.ValidParameters import validParameters
from Tensile.Contractions import SizeMapping
from Tensile.SolutionStructs.Solution import (
    Solution,
    _disableUnsupportedRuntimeStaggerU,
    validateParameterTypes,
)

pytestmark = pytest.mark.unit


# Tensile keeps process-global, module-level default dicts (`defaultSolution`,
# `globalParameters`) that `Solution.__init__` reads while constructing a solution.
# Some sibling unit tests mutate these in place -- e.g. test_MatrixInstructionConversion
# injects a "ProblemType" key into `defaultSolution`, which makes Solution.__init__'s
# `for key in defaultSolution` loop overwrite the already-converted ProblemType object
# with the raw config dict, leaving DataType a str and crashing
# assignProblemIndependentDerivedParameters. That manifested as order-dependent
# failures of the Solution-validation tests below under pytest-xdist. Snapshot the
# pristine defaults at import time (collection runs before any test executes, so they
# are clean here) and restore them around every test so Solution construction in this
# module is hermetic regardless of suite ordering.
_PRISTINE_DEFAULT_SOLUTION = deepcopy(defaultSolution)
_PRISTINE_GLOBAL_PARAMETERS = deepcopy(globalParameters)


@pytest.fixture(autouse=True)
def _isolate_global_solution_state():
    def _restore(target, pristine):
        target.clear()
        target.update(deepcopy(pristine))

    _restore(defaultSolution, _PRISTINE_DEFAULT_SOLUTION)
    _restore(globalParameters, _PRISTINE_GLOBAL_PARAMETERS)
    yield
    _restore(defaultSolution, _PRISTINE_DEFAULT_SOLUTION)
    _restore(globalParameters, _PRISTINE_GLOBAL_PARAMETERS)


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


class _SetupNewTilePapTdmWriter:
    def __init__(self):
        self.states = SimpleNamespace(
            actualSummationLoops=1,
            doShadowInit=2,
            IncLdsBufSwitch=False,
            ldsTensorTokenIdx=0,
            memTokenLdsBuffer0=0,
            memTokenLdsBuffer1=1,
            staggerUCode=False,
            unrollIdx=0,
        )
        self.do = {"executeToInitEnd": False}
        self.dontAppendCode = False
        self.labels = _StubLabels()

    def _module(self, name):
        return _module_with_comment(name, "unit: %s" % name)

    def graWorkGroup(self, *args):
        return self._module("graWorkGroup")

    def localReadAddresses(self, *args):
        return self._module("localReadAddresses")

    def localWriteAddresses(self, *args):
        return self._module("localWriteAddresses")

    def removeGRSrdVariableSgprsFromPool(self, kernel):
        return self._module("removeGRSrdVariableSgprsFromPool")

    def initTDMDescriptorWaveSeparated(self, kernel, tpa, tpb):
        return self._module("initTDMDescriptorWaveSeparated_%s_%s" % (tpa["tensorChar"], tpb["tensorChar"]))

    def tdmGlobalOffsetWaveSeparated(self, kernel, tpa, tpb):
        return self._module("tdmGlobalOffsetWaveSeparated_%s_%s" % (tpa["tensorChar"], tpb["tensorChar"]))

    def removeGROffsetsVariableSgprsFromPool(self, kernel):
        return self._module("removeGROffsetsVariableSgprsFromPool")

    def tdmSetupIncrementWaveSeparated(self, kernel, tpa, tpb):
        return self._module("tdmSetupIncrementWaveSeparated_%s_%s" % (tpa["tensorChar"], tpb["tensorChar"]))

    def tdmApplyStreamKOffsetWaveSeparated(self, kernel, tpa, tpb):
        return self._module("tdmApplyStreamKOffsetWaveSeparated_%s_%s" % (tpa["tensorChar"], tpb["tensorChar"]))

    def releaseGlobalReadIncsSgprsAfterTdmWaveSep(self, kernel):
        return self._module("releaseGlobalReadIncsSgprsAfterTdmWaveSep")

    def undefineSgpr(self, name):
        return self._module("undefineSgpr_%s" % name)

    def initC(self, kernel):
        return self._module("initC")

    def calculateLoopNumIter(self, kernel, tpa, tpb, loop_idx):
        return self._module("calculateLoopNumIter")

    def localReadInitPointers(self, kernel, tpa, tpb):
        return self._module("localReadInitPointers_%s" % tpb["tensorChar"])

    def isPrefetchAcrossPersistentEnabled(self, kernel):
        return KernelWriter.isPrefetchAcrossPersistentEnabled(self, kernel)

    def papTdmRestoreLdsBank(self, kernel, tpa, tpb):
        return self._module("papTdmRestoreLdsBank")

    def isSwapGlobalReadOrderForDtvOrDtl(self, kernel, prefetch1=False):
        return False

    def openSumAtLeastUnroll(self, kernel, prefetch=False, isOptNLL=True):
        return self._module("openSumAtLeastUnroll")

    def directToLdsM0Update(self, kernel, offset, tensor_parameters, skipWait=False):
        return self._module("directToLdsM0Update_%s" % tensor_parameters["tensorChar"])

    def globalReadDo(self, kernel, offset, tensor_parameters, **kwargs):
        return self._module("globalReadDo_%s" % tensor_parameters["tensorChar"])

    def globalReadIncrementAB(self, kernel, tpa, tpb, loop_idx, prefetch_index):
        return self._module("globalReadIncrementAB")


class _StubGsu:
    def setupNewTile(self, writer, kernel, tpa, tpb, tpm):
        return _module_with_comment("gsuSetupNewTile", "unit: GSU setup")


class _StubTdmComp:
    def getLdsAddrSgprName(self, group_name):
        return "%s+1" % group_name


class _PapTdmDescriptorRefreshWriter:
    def __init__(self):
        self._next_tmp_sgpr = 300
        self.recomputed_waveidx = []
        self.init_waveidx = []
        self.global_offset_waveidx = []

    @contextmanager
    def allocTmpSgpr(self, size, alignment=1, tag=""):
        base = self._next_tmp_sgpr
        self._next_tmp_sgpr += size + alignment
        yield SimpleNamespace(idx=base, size=size)

    def papTdmRecomputeWaveIdx(self, kernel, wave_idx_sgpr):
        self.recomputed_waveidx.append(wave_idx_sgpr)
        return _module_with_comment("papTdmRecomputeWaveIdx", "unit: recompute WaveIdx")

    def initTDMDescriptorWaveSeparated(self, kernel, tpa, tpb, wave_idx_sgpr="WaveIdx"):
        self.init_waveidx.append(wave_idx_sgpr)
        return _module_with_comment("initTDMDescriptorWaveSeparated", "unit: init TDM descriptor")

    def tdmGlobalOffsetWaveSeparated(self, kernel, tpa, tpb, wave_idx_sgpr="WaveIdx"):
        self.global_offset_waveidx.append(wave_idx_sgpr)
        return _module_with_comment("tdmGlobalOffsetWaveSeparated", "unit: global offset")

    def tdmApplyStreamKOffsetWaveSeparated(self, kernel, tpa, tpb):
        return _module_with_comment("tdmApplyStreamKOffsetWaveSeparated", "unit: StreamK offset")


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


def _problem_type(**overrides):
    problem_type = {
        "BiasSrc": "D",
        "Gradient": False,
        "MXBlockA": 0,
        "MXBlockB": 0,
        "NumIndicesSummation": 1,
        "Sparse": 0,
        "UseBias": False,
    }
    problem_type.update(overrides)
    return problem_type


def _kernel_from(base, **overrides):
    kernel = deepcopy(base)
    problem_type_overrides = overrides.pop("ProblemType", None)
    kernel.update(overrides)
    if problem_type_overrides is not None:
        kernel["ProblemType"].update(problem_type_overrides)
    return kernel


_CLASSIC_KERNEL_BASE = {
    "BufferLoad": True,
    "DirectToLdsA": False,
    "DirectToLdsB": False,
    "DirectToVgprA": False,
    "DirectToVgprB": False,
    "EdgeType": "None",
    "GuaranteeNoPartialA": False,
    "GuaranteeNoPartialB": False,
    "PrefetchGlobalRead": 2,
    "PrefetchGL2": 0,
    "ProblemType": _problem_type(),
    "StreamKForceDPOnly": 0,
    "UseGeneralizedNLCOneA": False,
    "UseGeneralizedNLCOneB": False,
    "_UseSgprForGRO": False,
    "enableTDMA": False,
    "enableTDMB": False,
}


_SETUP_NEW_TILE_TDM_BASE = _kernel_from(
    _CLASSIC_KERNEL_BASE,
    ClusterBarrier=False,
    DirectToVgprSparseMetadata=False,
    enableTDMA=True,
    enableTDMB=True,
    enableTDMMetadata=False,
    GuaranteeNoPartialA=True,
    GuaranteeNoPartialB=True,
    GuaranteeNoPartialMetadata=False,
    MIWaveGroup=[2, 1],
    Multicast=False,
    NumWaves=2,
    PrefetchAcrossPersistent=1,
    PrefetchGlobalRead=1,
    StreamK=3,
    SuppressNoLoadLoop=False,
    TDMInst=3,
    UseCustomMainLoopSchedule=0,
    UseSubtileImpl=False,
)


def _classic_kernel(**overrides):
    return _kernel_from(_CLASSIC_KERNEL_BASE, **overrides)


def _setup_new_tile_tdm_kernel(prefetch_across_persistent=1, **overrides):
    return _kernel_from(
        _SETUP_NEW_TILE_TDM_BASE,
        PrefetchAcrossPersistent=prefetch_across_persistent,
        **overrides,
    )


def _pap_wrapper_kernel(**overrides):
    return _classic_kernel(
        PrefetchAcrossPersistent=1,
        StreamK=3,
        SpaceFillingAlgo=[],
        **overrides,
    )


def _tensor_parameters(*, with_mx=False, with_metadata=False):
    tpa = {"tensorChar": "A", "isSwizzled": False}
    tpb = {"tensorChar": "B", "isSwizzled": False}
    if with_mx:
        tpa["MX"] = {"tensorChar": "MXSA", "isSwizzled": False}
        tpb["MX"] = {"tensorChar": "MXSB", "isSwizzled": False}
    if with_metadata:
        metadata = {"tensorChar": "Metadata", "isSwizzled": False}
        tpa.update({"is_sparse": False, "tpsMetadata": metadata})
        tpb.update({"is_sparse": False, "tpsMetadata": metadata})
    return tpa, tpb


_RUNTIME_STAGGER_BASE = {
    "PrefetchAcrossPersistent": 0,
    "TDMInst": 0,
    "StaggerU": 32,
    "StaggerUMapping": 2,
    "StaggerUStride": 256,
    "InternalSupportParams": {"SupportCustomStaggerU": True},
    "ProblemType": {"MXBlockA": 0, "MXBlockB": 0},
    "enableTDMA": False,
    "enableTDMB": False,
    "DirectToLdsA": False,
    "DirectToLdsB": False,
}


def _stagger_runtime_state(*, pap=False, tdm=False, **overrides):
    state = deepcopy(_RUNTIME_STAGGER_BASE)
    if pap:
        state["PrefetchAcrossPersistent"] = 1
    if tdm:
        state.update({"TDMInst": 3, "enableTDMA": True, "enableTDMB": True})
    state.update(overrides)
    return state


class _DefaultFalseDict(dict):
    def __missing__(self, key):
        return False


def _pap_solution_isa_info_map():
    isa = IsaVersion(9, 5, 0)
    asm_caps = _DefaultFalseDict(
        {
            "SupportedISA": True,
            "HasMFMA": True,
            "HasTDM": True,
            "HasDirectToLds": True,
            "HasDirectToLdsx4": True,
            "HasNTModifier": True,
            "HasMFMA_f64": True,
            "HasGLTr8B64": True,
            "HasGLTr16B128": True,
            "HasLDSTr": True,
        }
    )
    arch_caps = _DefaultFalseDict(
        {
            "DeviceLDS": 65536,
            "HasEccHalf": True,
            "HasSchedMode": True,
            "HasAccCD": True,
            "HasMXScaleSwizzle": True,
        }
    )
    return {isa: IsaInfo(asm_caps, arch_caps, _DefaultFalseDict(), _DefaultFalseDict())}


def _pap_solution_config(**overrides):
    config = {
        "ISA": [9, 5, 0],
        "EnableMatrixInstruction": True,
        "ProblemType": {
            "OperationType": "GEMM",
            "DataType": "s",
            "DestDataType": "s",
            "ComputeDataType": "s",
            "TransposeA": True,
            "TransposeB": False,
            "UseBeta": True,
            "Batched": True,
            "StridedBatched": True,
        },
        "MatrixInstruction": [32, 32, 8, 1],
        "MIBlock": [32, 32, 8, 1, 1, 1],
        "MIWaveGroup": [2, 1],
        "MIWaveTile": [1, 1],
        "MIInputPerThread": 1,
        "WorkGroup": [32, 2, 1],
        "StreamK": 3,
        "PrefetchAcrossPersistent": 1,
        "PrefetchGlobalRead": 1,
        "ScheduleIterAlg": 0,
        "TDMInst": 3,
        "StaggerU": 0,
        "1LDSBuffer": 0,
        "BufferLoad": True,
        "BufferStore": True,
        "StoreRemapVectorWidth": 0,
        "SuppressNoLoadLoop": False,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "UseSubtileImpl": False,
        "DepthU": 32,
        "LocalSplitU": 1,
    }
    problem_type_overrides = overrides.pop("ProblemType", None)
    config.update(overrides)
    if problem_type_overrides is not None:
        config["ProblemType"].update(problem_type_overrides)
    return config


def _pap_solution(**overrides):
    assembler = SimpleNamespace(
        code_object_version="default",
        rocm_version=SemanticVersion(6, 4, 0),
    )
    return Solution(
        _pap_solution_config(**overrides),
        False,
        True,
        False,
        assembler,
        _pap_solution_isa_info_map(),
    )


def _module_items(module):
    return [module.getItem(i) for i in range(module.itemsSize())]


def _module_index(items, name):
    return next(i for i, item in enumerate(items) if isinstance(item, Module) and item.name == name)


def _module_names(module_or_items):
    items = _module_items(module_or_items) if isinstance(module_or_items, Module) else module_or_items
    return [item.name for item in items if isinstance(item, Module)]


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


def _setup_new_tile_module_names(prefetch_across_persistent):
    tpa, tpb = _tensor_parameters(with_metadata=True)
    module = KernelWriter.setupNewTile(
        _SetupNewTilePapTdmWriter(),
        _setup_new_tile_tdm_kernel(prefetch_across_persistent=prefetch_across_persistent),
        tpa,
        tpb,
    )
    return _module_names(module)


def _prefetch_across_persistent(monkeypatch, *, skip_barrier=False, **kernel_overrides):
    monkeypatch.setattr(kwa_module.Component.StreamK, "find", lambda writer: _StubStreamK())
    writer = _ClassicPapWrapperWriter()
    module = kwa_module.KernelWriterAssembly.prefetchAcrossPersistent(
        writer,
        _pap_wrapper_kernel(**kernel_overrides),
        *_tensor_parameters(),
        skipBarrier=skip_barrier,
    )
    return writer, _module_items(module)


def _streamk_with_stubbed_tile_indexing():
    streamk = StreamKTwoTileDPFirst()
    streamk.skTileIndex = lambda writer, kernel, s_tmp, tpa, tpb, skipLroReset=False: (
        _module_with_comment("skTileIndex", "unit: tile index")
    )
    streamk.skIndexToWG = lambda writer, kernel, s_tmp: _module_with_comment(
        "skIndexToWG", "unit: index to WG"
    )
    return streamk


def _streamk_wgm_writer():
    writer = SimpleNamespace(
        sgprPool=RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False, printRP=False),
        states=SimpleNamespace(WGMTransformLevels=-1),
    )

    # prefetchAcrossPersistentSetupNextTile now takes its SKPrefetchTemp through the
    # growable allocTmpSgpr path (see StreamK.py); provide a counter-based stub
    # matching the other PAP mock writers (skTileIndex/skIndexToWG are stubbed, so
    # the actual register index is irrelevant here).
    next_tmp = [100]

    @contextmanager
    def _alloc_tmp_sgpr(size, alignment=1, tag=""):
        base = next_tmp[0]
        next_tmp[0] += size + alignment
        yield SimpleNamespace(idx=base, size=size)

    writer.allocTmpSgpr = _alloc_tmp_sgpr
    return writer


def test_pap_is_valid_solution_parameter():
    assert validParameters["PrefetchAcrossPersistent"] == [0, 1]
    assert defaultSolution["PrefetchAcrossPersistent"] == 0
    assert "PrefetchAcrossPersistent" in getRequiredParametersMin()
    assert "prefetchAcrossPersistent" in SizeMapping.StateKeys
    validateParameterTypes({"PrefetchAcrossPersistent": 1})


def test_solution_validation_accepts_minimal_pap_tdm_contract():
    assert _pap_solution()["Valid"] is True


@pytest.mark.parametrize(
    "overrides, reason",
    [
        pytest.param(
            {"1LDSBuffer": 1},
            "PrefetchAcrossPersistent requires 1LDSBuffer != 1",
            id="rejects_single_lds_buffer",
        ),
        pytest.param(
            {"StaggerU": 32},
            "TDM + PrefetchAcrossPersistent with StaggerU is not implemented",
            id="rejects_nonzero_staggeru",
        ),
        pytest.param(
            # ScheduleIterAlg 3 so the SIA0+StreamK rule (which already caps PGR at 2)
            # doesn't reject first; this exercises the PAP PGR<=2 contract directly.
            {"PrefetchGlobalRead": 3, "ScheduleIterAlg": 3},
            "PrefetchAcrossPersistent requires PrefetchGlobalRead in [1, 2]",
            id="rejects_pgr_above_two",
        ),
    ],
)
def test_solution_validation_rejects_unsupported_pap_tdm_contracts(capsys, overrides, reason):
    solution = _pap_solution(**overrides)

    assert solution["Valid"] is False
    assert reason in capsys.readouterr().out


@pytest.mark.parametrize(
    "pap, tdm, expected",
    [
        pytest.param(
            True,
            True,
            (0, 0, 0, False),
            id="pap_tdm_disables_runtime_custom_staggeru",
        ),
        pytest.param(
            False,
            True,
            (32, 2, 256, True),
            id="non_pap_tdm_keeps_runtime_custom_staggeru",
        ),
    ],
)
def test_runtime_staggeru_controls_for_tdm_pap(pap, tdm, expected):
    state = _stagger_runtime_state(pap=pap, tdm=tdm, StaggerU=0 if pap else 32)

    _disableUnsupportedRuntimeStaggerU(state)

    stagger_u, mapping, stride, support_custom = expected
    assert state["StaggerU"] == stagger_u
    assert state["StaggerUMapping"] == mapping
    assert state["StaggerUStride"] == stride
    assert state["InternalSupportParams"]["SupportCustomStaggerU"] is support_custom


def test_setup_new_tile_releases_waveidx_for_pap_wave_separated_tdm(monkeypatch):
    monkeypatch.setattr(kw_module.Component.GSU, "find", lambda writer: _StubGsu())

    pap_module_names = _setup_new_tile_module_names(prefetch_across_persistent=1)
    non_pap_module_names = _setup_new_tile_module_names(prefetch_across_persistent=0)

    assert "papTdmRestoreLdsBank" in pap_module_names
    assert "undefineSgpr_WaveIdx" in pap_module_names
    assert "undefineSgpr_WaveIdx" in non_pap_module_names
    assert pap_module_names.index("undefineSgpr_WaveIdx") < pap_module_names.index("papTdmRestoreLdsBank")


def test_pap_tdm_descriptor_refresh_threads_temporary_waveidx(monkeypatch):
    monkeypatch.setattr(kwa_module.TensorDataMoverLoad, "find", lambda writer: _StubTdmComp())
    writer = _PapTdmDescriptorRefreshWriter()
    kernel = {"LdsOffsetA_Blk": 0, "StreamK": 3}
    tpa, tpb = {"tensorChar": "A"}, {"tensorChar": "B"}

    module = kwa_module.KernelWriterAssembly.papTdmUpdateDescriptor(writer, kernel, tpa, tpb)

    assert _module_names(module) == [
        "papTdmRecomputeWaveIdx",
        "initTDMDescriptorWaveSeparated",
        "tdmGlobalOffsetWaveSeparated",
        "tdmApplyStreamKOffsetWaveSeparated",
    ]
    assert writer.recomputed_waveidx == [300]
    assert writer.init_waveidx == [300]
    assert writer.global_offset_waveidx == [300]


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


def test_classic_pap_checkpoints_loop_counters_in_vgprs_around_next_tile_recount(monkeypatch):
    writer, items = _prefetch_across_persistent(monkeypatch)

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


def test_dp_only_pap_skips_loop_counter_checkpoint(monkeypatch):
    # DP-only StreamK keeps LoopCounter/OrigLoopCounter constant (idempotent
    # recompute, PAP never runs on the last tile), so prefetchAcrossPersistent
    # skips the 2-VGPR checkpoint/restore entirely
    # (KernelWriterAssembly: snapshotLoopCounter = not StreamKForceDPOnly).
    writer, items = _prefetch_across_persistent(monkeypatch, StreamKForceDPOnly=1)

    assert not any(tag == "PAP loop counters" for _, _, tag in writer.vgprPool.checked_out)
    assert not _instruction_indices(items, kwa_module.VReadfirstlaneB32, dst_contains="sgprLoopCounterL")
    assert not _instruction_indices(items, kwa_module.VReadfirstlaneB32, dst_contains="sgprOrigLoopCounter")
    # the next-tile prefetch setup still runs; only the loop-counter snapshot is skipped
    _module_index(items, "setupPrefetchAcrossPersistentLoads")


def test_classic_pap_can_skip_internal_barrier_after_caller_sync(monkeypatch):
    _, with_barrier = _prefetch_across_persistent(monkeypatch, skip_barrier=False)
    _, without_barrier = _prefetch_across_persistent(monkeypatch, skip_barrier=True)

    assert _instruction_indices(with_barrier, kwa_module.SBarrier)
    assert not _instruction_indices(without_barrier, kwa_module.SBarrier)


@pytest.mark.parametrize(
    "space_filling_algo, remap_name, expected_transform_levels",
    [
        pytest.param([], "DefaultWGM", -1, id="default_wgm"),
        pytest.param([{"foo": "bar"}], "SpaceFillingCurveWalk", 1, id="space_filling_wgm"),
    ],
)
def test_streamk_pap_next_tile_setup_applies_wgm_remap(
    monkeypatch,
    space_filling_algo,
    remap_name,
    expected_transform_levels,
):
    import Tensile.Components.WorkGroupMappingAlgos as wgm_algos

    monkeypatch.setattr(
        wgm_algos,
        remap_name,
        lambda writer, kernel, sgpr_wgm: _module_with_comment(
            remap_name, "unit: %s remap" % remap_name
        ),
    )

    writer = _streamk_wgm_writer()
    module = _streamk_with_stubbed_tile_indexing().prefetchAcrossPersistentSetupNextTile(
        writer,
        {"SpaceFillingAlgo": space_filling_algo},
        {"tensorChar": "A"},
        {"tensorChar": "B"},
    )
    items = _module_items(module)

    tile_index = _module_index(items, "skTileIndex")
    index_to_wg = _module_index(items, "skIndexToWG")
    wgm_remap = _module_index(items, remap_name)
    assert tile_index < index_to_wg
    assert index_to_wg < wgm_remap
    assert writer.states.WGMTransformLevels == expected_transform_levels
