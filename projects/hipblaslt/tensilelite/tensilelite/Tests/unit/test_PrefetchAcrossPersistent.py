################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
################################################################################

import copy
import re
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest
from rocisa.code import Module
from rocisa.enum import RegisterType
from rocisa.register import RegisterPool

import tensilelite.KernelWriter as kw_module
from tensilelite.KernelWriter import KernelWriter
import tensilelite.KernelWriterAssembly as kwa_module
from tensilelite.Components.StreamK import StreamKDynamic, StreamKHybrid, StreamKTwoTileDPFirst
from tensilelite.Components.TDMFuse import TDM_GROUPS, tdmGrouping, tdmPapRejectReason, tdmScaleSharesDataSet
from tensilelite.Components.TileProcessingStrategy import DataParallel
from tensilelite.Components.PersistentLoop import PersistentLoopOn
from tensilelite.Components.WorkAssignment import StaticGrid, DynamicWorkQueue, Hybrid
from tensilelite.Common.GlobalParameters import defaultSolution, globalParameters
from tensilelite.Common.RequiredParameters import getRequiredParametersMin
from tensilelite.Common.Types import IsaInfo, IsaVersion, SemanticVersion
from tensilelite.Common.ValidParameters import validParameters
from tensilelite.Contractions import SizeMapping
from tensilelite.SolutionStructs.Solution import (
    Solution,
    _disableUnsupportedRuntimeStaggerU,
    validateParameterTypes,
)

pytestmark = pytest.mark.unit

# TensileLite keeps process-global, module-level default dicts (`defaultSolution`,
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
            dcpTokenGate=False,
            kernel={"TDMPlusLdsBuf": 0},
            ldsTensorTokenIdx=0,
            memTokenLdsBuffer0=0,
            memTokenLdsBuffer1=1,
            numLDSBlk=2,
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
_ClassicPapWriter._nextLdsToken = KernelWriter._nextLdsToken
_ClassicPapWriter._dcpDivergent = kwa_module.KernelWriterAssembly._dcpDivergent
_ClassicPapWriter._dcpThickThinIssueOrder = KernelWriter._dcpThickThinIssueOrder


class _SetupNewTilePapTdmWriter:
    def __init__(self):
        self.states = SimpleNamespace(
            actualSummationLoops=1,
            dcpTokenGate=False,
            doShadowInit=2,
            IncLdsBufSwitch=False,
            ldsTensorTokenIdx=0,
            memTokenLdsBuffer0=0,
            memTokenLdsBuffer1=1,
            numLDSBlk=2,
            staggerUCode=False,
            unrollIdx=0,
            # Capability/kernel state consumed by ClusterLoadTDM.find()'s
            # PartialMatch (asmCaps HasTDM + kernel TDMInst==3), mirroring the
            # real writer.states on a gfx1250 TDM path so the component matches.
            asmCaps={"HasTDM": True},
            kernel={"TDMInst": 3, "TDMPlusLdsBuf": 0},
            waveIdxReleasedAfterStagger=False,
            tdmParityPackedInArgType=False,
        )
        self.do = {"executeToInitEnd": False}
        self.dontAppendCode = False
        self.labels = _StubLabels()
        # releaseWaveIdxAfterStagger runs for real here (it asserts the pool slot is
        # still checked out and latches against a second check-in), so back the mock
        # with a real RegisterPool rather than stubbing the release out.
        self.sgprPool = RegisterPool(
            8, RegisterType.Sgpr, defaultPreventOverflow=False, printRP=False
        )
        self.sgprPool.add(0, 8, "unit")
        self.sgprs = {"WaveIdx": self.sgprPool.checkOut(1, "WaveIdx")}

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

    def tdmApplyTileKOffsetWaveSeparated(self, kernel, tpa, tpb):
        return self._module("tdmApplyTileKOffsetWaveSeparated_%s_%s" % (tpa["tensorChar"], tpb["tensorChar"]))

    def releaseGlobalReadIncsSgprsAfterTdmWaveSep(self, kernel):
        return self._module("releaseGlobalReadIncsSgprsAfterTdmWaveSep")

    def isTdmWaveSeparated(self, kernel):
        return kwa_module.KernelWriterAssembly.isTdmWaveSeparated(self, kernel)

    def tdmFusePaired(self, kernel):
        return kwa_module.KernelWriterAssembly.tdmFusePaired(self, kernel)

    def _tdmPairedParityOrder(self, kernel, tpa, tpb):
        return kwa_module.KernelWriterAssembly._tdmPairedParityOrder(self, kernel, tpa, tpb)

    def tdmSeparateABDescriptors(self, kernel):
        return kwa_module.KernelWriterAssembly.tdmSeparateABDescriptors(self, kernel)

    def _dcpDivergent(self, kernel):
        return kwa_module.KernelWriterAssembly._dcpDivergent(self, kernel)

    def tdmWaveIdxReadAfterPrologue(self, kernel):
        return kwa_module.KernelWriterAssembly.tdmWaveIdxReadAfterPrologue(self, kernel)

    def isTdmWaveIdxLive(self, kernel):
        return kwa_module.KernelWriterAssembly.isTdmWaveIdxLive(self, kernel)

    def undefineSgpr(self, name):
        # Mirror the real undefineSgpr: return the slot to the pool but keep the name
        # in self.sgprs, so the latch stays the only guard against a double check-in.
        if name in self.sgprs:
            self.sgprPool.checkIn(self.sgprs[name])
        return self._module("undefineSgpr_%s" % name)

    def releaseWaveIdxAfterStagger(self, kernel):
        return kwa_module.KernelWriterAssembly.releaseWaveIdxAfterStagger(self, kernel)

    def hoistWaveParityWrapUSel(self, kernel, tpa, tpb):
        return self._module("hoistWaveParityWrapUSel")

    def packTdmParityIntoArgType(self, kernel):
        return self._module("packTdmParityIntoArgType")

    def declareStaggerParms(self, kernel):
        return self._module("declareStaggerParms")

    def calculateStagger(self, kernel, tensor_parameters):
        return self._module("calculateStagger_%s" % tensor_parameters["tensorChar"])

    def initC(self, kernel):
        return self._module("initC")

    def calculateLoopNumIter(self, kernel, tpa, tpb, loop_idx):
        return self._module("calculateLoopNumIter")

    def localReadInitPointers(self, kernel, tpa, tpb):
        return self._module("localReadInitPointers_%s" % tpb["tensorChar"])

    def isPrefetchAcrossPersistentEnabled(self, kernel):
        return KernelWriter.isPrefetchAcrossPersistentEnabled(self, kernel)

    def _nextLdsToken(self, idx):
        return KernelWriter._nextLdsToken(self, idx)

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

    def tdmApplyTileKOffsetWaveSeparated(self, kernel, tpa, tpb):
        return _module_with_comment("tdmApplyTileKOffsetWaveSeparated", "unit: StreamK offset")

    def papTdmRefreshPairs(self, kernel, tpa, tpb):
        return kwa_module.KernelWriterAssembly.papTdmRefreshPairs(self, kernel, tpa, tpb)


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
        # rapInPapNextTilePrefetch: prefetchAcrossPersistent raises it over the
        # window where WorkGroup* names the next tile, so RAP's A silencing can
        # tell a next-tile load from an in-loop one.
        self.states = SimpleNamespace(unrollIdx=0, rapInPapNextTilePrefetch=False)
        self.vgprPool = _TrackingRegisterPool(RegisterType.Vgpr)

    def isPrefetchAcrossPersistentEnabled(self, kernel):
        return True

    @contextmanager
    def allocPapTileIdentity(self, kernel, subtile=False):
        yield {
            "WorkGroup0": 100,
            "WorkGroup1": 101,
            "WorkGroup2": 102,
            "StreamKLocalStart": 103,
            "StreamKLocalEnd": 104,
        }

    def papCheckpointCurrentTileIdentity(self, kernel, prev_tile, subtile=False):
        return _module_with_comment("papCheckpointCurrentTileIdentity", "unit: checkpoint tile")

    def loopCounterName(self, kernel, loop_idx):
        return "LoopCounterL"

    def calculateLoopNumIter(self, kernel, tpa, tpb, loop_idx):
        return _module_with_comment("calculateLoopNumIter", "unit: calculate loop num iter")

    def setupPrefetchAcrossPersistentLoads(self, kernel, tpa, tpb, isOptNLL=True):
        return _module_with_comment("setupPrefetchAcrossPersistentLoads", "unit: setup PAP loads")

    def papRestoreCurrentTileIdentity(self, kernel, prev_tile, subtile=False):
        return _module_with_comment("papRestoreCurrentTileIdentity", "unit: restore tile")


class _StubStreamK:
    def prefetchEligibility(self, writer, kernel, skipLabel):
        return Module("prefetchEligibility")

    def prefetchAcrossPersistentSetupNextTile(self, writer, kernel, tpa, tpb, skipLroReset=False):
        return _module_with_comment("prefetchAcrossPersistentSetupNextTile", "unit: setup next tile")


class _StubWorkAssignment:
    def reserveNext(self, writer, kernel, skipLabel):
        return _module_with_comment("reserveNext", "unit: reserve next persistent assignment")


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
    "HalfPLR": 0,
    "NoTailLoop": False,
    "PrefetchGlobalRead": 2,
    "PrefetchGL2": 0,
    "ProblemType": _problem_type(),
    "ReuseAcrossPersistent": 0,
    "TileProcessingStrategy": "None",
    "WorkAssignment": "StaticGrid",
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
    TileProcessingStrategy="StreamK",
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
    overrides.setdefault("TileProcessingStrategy", "StreamK")
    return _classic_kernel(
        PrefetchAcrossPersistent=1,
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
    "ClusterDim": [1, 1],
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
        "TileProcessingStrategy": "StreamK",
        "WorkAssignment": "StaticGrid",
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


def _pap_solution(*, rocm_version=SemanticVersion(6, 4, 0), **overrides):
    assembler = SimpleNamespace(
        code_object_version="default",
        rocm_version=rocm_version,
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


def _setup_new_tile_writer_and_names(prefetch_across_persistent, stagger_u_code=False):
    tpa, tpb = _tensor_parameters(with_metadata=True)
    writer = _SetupNewTilePapTdmWriter()
    writer.states.staggerUCode = stagger_u_code
    module = KernelWriter.setupNewTile(
        writer,
        _setup_new_tile_tdm_kernel(prefetch_across_persistent=prefetch_across_persistent),
        tpa,
        tpb,
    )
    return writer, _module_names(module)


def _setup_new_tile_module_names(prefetch_across_persistent, stagger_u_code=False):
    return _setup_new_tile_writer_and_names(prefetch_across_persistent, stagger_u_code)[1]


def _waveidx_is_in_pool(writer):
    status = writer.sgprPool.getPool()[writer.sgprs["WaveIdx"]].status
    return status == RegisterPool.Status.Available


def _prefetch_across_persistent(monkeypatch, *, skip_barrier=False, **kernel_overrides):
    monkeypatch.setattr(kwa_module.Component.TileProcessingStrategy, "find", lambda writer: _StubStreamK())
    monkeypatch.setattr(kwa_module.Component.WorkAssignment, "find", lambda writer: _StubWorkAssignment())
    monkeypatch.setattr(kwa_module.Component.PersistentLoop, "find", lambda writer: PersistentLoopOn())
    writer = _ClassicPapWrapperWriter()
    kernel = _pap_wrapper_kernel(**kernel_overrides)
    processing = DataParallel() if kernel["TileProcessingStrategy"] == "DataParallel" else StreamKTwoTileDPFirst()
    writer.states.kernel = kernel
    writer.states.currentTileWork = processing.tileWork(kernel)
    module = kwa_module.KernelWriterAssembly.prefetchAcrossPersistent(
        writer,
        kernel,
        *_tensor_parameters(),
        skipBarrier=skip_barrier,
    )
    # The loop now owns the classic handoff as a nested module. Preserve the
    # instruction-order assertions across that ownership boundary.
    def flatten(module):
        for item in _module_items(module):
            yield item
            if isinstance(item, Module):
                yield from flatten(item)
    return writer, list(flatten(module))


def _streamk_with_stubbed_tile_indexing():
    streamk = StreamKTwoTileDPFirst()
    streamk.skTileIndex = lambda writer, kernel, s_tmp, tpa, tpb, skipLroReset=False: (
        _module_with_comment("skTileIndex", "unit: tile index")
    )
    streamk.tileIndexToWorkGroup = lambda writer, kernel, s_tmp: _module_with_comment(
        "skIndexToWG", "unit: index to WG"
    )
    return streamk


def _streamk_wgm_writer():
    writer = SimpleNamespace(
        sgprPool=RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False, printRP=False),
        states=SimpleNamespace(WGMTransformLevels=-1),
        isPersistentConstantsToVgprEnabled=lambda kernel: False,
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
    "rocm_version, expected_preload",
    [
        pytest.param(SemanticVersion(6, 0, 32649), False, id="rocm_6_before_floor"),
        pytest.param(SemanticVersion(6, 0, 32650), True, id="rocm_6_at_floor"),
        pytest.param(SemanticVersion(7, 1, 25424), True, id="rocm_7_low_build"),
    ],
)
def test_solution_applies_preload_gate_from_assembler_version(
    rocm_version, expected_preload
):
    solution = _pap_solution(
        rocm_version=rocm_version,
        PreloadKernArgs=True,
    )

    assert solution["Valid"] is True
    assert solution["PreloadKernArgs"] is expected_preload


def test_solution_validation_accepts_pap_streamk_dynamic():
    # PAP is allowed for StreamK==4 (StreamKDynamic) in addition to StreamK==3.
    # The validation gate accepts StreamK in (3, 4, 5); every other PAP axis
    # restriction is StreamK-agnostic and still applies. TDM is disabled here
    # (TDMInst=0): the TDM+PAP twin gate is intentionally kept SK3-only, so
    # SK4 PAP is supported for the non-TDM path only.
    assert _pap_solution(WorkAssignment="DynamicWorkQueue", TDMInst=0)["Valid"] is True


def test_solution_validation_rejects_pap_streamk_dynamic_with_tdm(capsys):
    # The TDM + PAP twin gate is deliberately NOT relaxed for SK4: TDM+PAP
    # remains StreamK==3 only.
    assert _pap_solution(WorkAssignment="DynamicWorkQueue", TDMInst=3)["Valid"] is False
    assert "TDM + PrefetchAcrossPersistent requires WorkAssignment=StaticGrid" in capsys.readouterr().out


def test_solution_validation_accepts_pap_streamk_hybrid():
    # PAP is allowed for StreamK==5 (StreamKHybrid) in addition to StreamK==3
    # and StreamK==4. The validation gate accepts StreamK in (3, 4, 5). A
    # single PAP-enabled SK5 kernel is correct for BOTH runtime sub-paths
    # (static SK3-like and dynamic SK4-like) via WorkAssignmentMode dispatch.
    # TDM is disabled here (TDMInst=0): the TDM+PAP twin gate is intentionally
    # kept SK3-only, so SK5 PAP is supported for the non-TDM path.
    assert _pap_solution(WorkAssignment="Hybrid", TDMInst=0)["Valid"] is True


def test_solution_validation_rejects_pap_streamk_hybrid_with_tdm(capsys):
    # The TDM + PAP twin gate is deliberately NOT relaxed for SK5: TDM+PAP
    # remains StreamK==3 only (hybrid TDM+PAP deferred).
    assert _pap_solution(WorkAssignment="Hybrid", TDMInst=3)["Valid"] is False
    assert "TDM + PrefetchAcrossPersistent requires WorkAssignment=StaticGrid" in capsys.readouterr().out

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
        pytest.param(
            {"PrefetchGlobalRead": 2, "PrefetchGlobalReadA": 1, "PrefetchGlobalReadB": 2},
            "PrefetchGlobalReadA/B: PrefetchAcrossPersistent is not",
            id="rejects_decoupled_pgr",
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


@pytest.mark.parametrize(
    "cluster_dim, expected",
    [
        pytest.param([1, 1], (32, 2, 256, True), id="no_cluster_keeps_runtime_custom_staggeru"),
        pytest.param([2, 2], (0, 0, 0, False), id="cluster_2x2_disables_runtime_custom_staggeru"),
        pytest.param([4, 4], (0, 0, 0, False), id="cluster_4x4_disables_runtime_custom_staggeru"),
    ],
)
def test_runtime_staggeru_controls_for_cluster(cluster_dim, expected):
    # PAP off + TDM off so only the workgroup-cluster gate can fire.
    state = _stagger_runtime_state(pap=False, tdm=False, ClusterDim=cluster_dim)

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


def test_setup_new_tile_releases_waveidx_after_stagger_for_wave_separated_tdm(monkeypatch):
    """WaveIdx survives the stagger prologue, then dies before the unroll loop.

    The stagger prologue reads wave parity straight out of s[sgprWaveIdx], so the
    release cannot happen at the usual spot above calculateLoopNumIter. It must still
    happen inside setupNewTile: WaveIdx sits at a low physical index and holding it
    across the main loop pushes the tightest gfx1250 StreamK configs over MaxSgpr.
    """
    monkeypatch.setattr(kw_module.Component.GSU, "find", lambda writer: _StubGsu())

    for pap in (0, 1):
        writer, module_names = _setup_new_tile_writer_and_names(
            prefetch_across_persistent=pap, stagger_u_code=True
        )
        # Confirm we really took the stagger path before asserting on its effect.
        assert "calculateStagger_A" in module_names
        assert "calculateStagger_B" in module_names
        # The release is not the plain undefineSgpr emitted on the non-stagger path;
        # it goes through releaseWaveIdxAfterStagger after packing parity into
        # ArgType bit 8. Later sites use that packed bit, not a live WaveIdx.
        assert "undefineSgpr_WaveIdx" not in module_names
        assert "ReleaseWaveIdxAfterStagger" in module_names
        assert "hoistWaveParityWrapUSel" in module_names
        assert "packTdmParityIntoArgType" in module_names
        assert module_names.index("calculateStagger_B") < module_names.index(
            "hoistWaveParityWrapUSel"
        )
        assert module_names.index("hoistWaveParityWrapUSel") < module_names.index(
            "packTdmParityIntoArgType"
        )
        assert module_names.index("packTdmParityIntoArgType") < module_names.index(
            "ReleaseWaveIdxAfterStagger"
        )
        assert writer.states.waveIdxReleasedAfterStagger is True
        assert _waveidx_is_in_pool(writer)


def test_setup_new_tile_releases_waveidx_at_most_once(monkeypatch):
    """A second setupNewTile emit must not check the WaveIdx slot in twice."""
    monkeypatch.setattr(kw_module.Component.GSU, "find", lambda writer: _StubGsu())

    tpa, tpb = _tensor_parameters(with_metadata=True)
    writer = _SetupNewTilePapTdmWriter()
    writer.states.staggerUCode = True
    kernel = _setup_new_tile_tdm_kernel(prefetch_across_persistent=1)

    first = KernelWriter.setupNewTile(writer, kernel, tpa, tpb)
    second = KernelWriter.setupNewTile(writer, kernel, tpa, tpb)

    def release_module(module):
        found = [
            item
            for item in _module_items(module)
            if isinstance(item, Module) and item.name == "ReleaseWaveIdxAfterStagger"
        ]
        assert len(found) == 1, f"expected one ReleaseWaveIdxAfterStagger, got {len(found)}"
        return found[0]

    assert _module_names(release_module(first)) == ["undefineSgpr_WaveIdx"]
    # The second emit still calls the helper, but the latch must make it come back
    # empty -- a second undefineSgpr would check the same pool slot in twice.
    assert _module_names(release_module(second)) == []
    assert writer.states.waveIdxReleasedAfterStagger is True


def test_pap_tdm_descriptor_refresh_threads_temporary_waveidx(monkeypatch):
    monkeypatch.setattr(kwa_module.TensorDataMoverLoad, "find", lambda writer: _StubTdmComp())
    writer = _PapTdmDescriptorRefreshWriter()
    kernel = {"LdsOffsetA_Blk": 0, "TileProcessingStrategy": "StreamK", "WorkAssignment": "StaticGrid"}
    tpa, tpb = {"tensorChar": "A"}, {"tensorChar": "B"}

    module = kwa_module.KernelWriterAssembly.papTdmUpdateDescriptor(writer, kernel, tpa, tpb)

    assert _module_names(module) == [
        "papTdmRecomputeWaveIdx",
        "initTDMDescriptorWaveSeparated",
        "tdmGlobalOffsetWaveSeparated",
        "tdmApplyTileKOffsetWaveSeparated",
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
    primed = _instruction_index(items, kwa_module.SMovB32, "s[sgprPersistentPrefetchState]", "1")

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
    primed = _instruction_index(items, kwa_module.SMovB32, "s[sgprPersistentPrefetchState]", "1")
    save_lds_bank = _module_index(items, "papDtlSaveLdsBank")

    assert primed < save_lds_bank
    assert writer.states.ldsTensorTokenIdx == writer.states.memTokenLdsBuffer1


def _assert_loop_counters_checkpointed_in_vgprs(writer, items):
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


def test_classic_pap_checkpoints_loop_counters_in_vgprs_around_next_tile_recount(monkeypatch):
    writer, items = _prefetch_across_persistent(monkeypatch)

    _assert_loop_counters_checkpointed_in_vgprs(writer, items)


def test_halfplr_pap_checkpoints_loop_counters_even_under_dp_only(monkeypatch):
    # HalfPLR enters PAP while LoopCounter is one, so the counters cannot be
    # recomputed and DP-only has to checkpoint them anyway.
    writer, items = _prefetch_across_persistent(monkeypatch, TileProcessingStrategy="DataParallel", HalfPLR=1)

    _assert_loop_counters_checkpointed_in_vgprs(writer, items)


def test_dp_only_pap_skips_loop_counter_checkpoint(monkeypatch):
    # DataParallel keeps LoopCounter/OrigLoopCounter constant (idempotent
    # recompute, PAP never runs on the last tile), so prefetchAcrossPersistent
    # skips the 2-VGPR checkpoint/restore entirely
    # (PersistentLoop checks HalfPLR and the TileWork local-K capability).
    writer, items = _prefetch_across_persistent(monkeypatch, TileProcessingStrategy="DataParallel")

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
    import tensilelite.Components.WorkGroupMappingAlgos as wgm_algos

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


def test_static_assignment_reservation_uses_streamk_partition_bound(monkeypatch):
    # Assignment owns the reservation; StreamK supplies the cursor and bound.
    # The exhaustion compare must remain after the reservation marker.
    from rocisa.code import Label

    monkeypatch.setattr(kwa_module.Component.TileProcessingStrategy, "find", lambda writer: StreamKTwoTileDPFirst())
    skip_label = Label("SK_SkipNllPAP_unit", "")
    module = StaticGrid().reserveNext(
        writer=None, kernel={}, skipLabel=skip_label
    )
    rendered = str(module)
    assert "s_cmp_ge_u32 s[sgprPersistentIteration], s[sgprPersistentIterationEnd]" in rendered
    assert "No next persistent iteration" in rendered
    assert "s_cbranch_scc1 label_SK_SkipNllPAP_unit" in rendered
    assert rendered.index("s_cmov_b32 s[sgprPersistentPrefetchState]") < rendered.index("s_cmp_ge_u32")


class _PapFetchWriter:
    def __init__(self):
        self.labels = _StubLabels()
        self.sgprPool = RegisterPool(
            0, RegisterType.Sgpr, defaultPreventOverflow=False, printRP=False
        )


def _fake_fetch(self, writer, kernel, preventOverflow=True, uniqueLabels=False):
    sidx = writer.sgprPool.checkOut(1, "workItemIdx")
    return _module_with_comment("fakeFetch", "unit: queue pop"), sidx


def test_queue_reservation_primes_before_drain_check(monkeypatch):
    # SK4 PAP must stash NextWorkItem and set PersistentPrefetchState before the
    # TotalItems drain compare so the back-edge never re-pops a termination token.
    monkeypatch.setattr(DynamicWorkQueue, "fetchAndBroadcast", _fake_fetch)
    monkeypatch.setattr(kwa_module.Component.TileProcessingStrategy, "find", lambda writer: StreamKDynamic())
    from rocisa.code import Label

    skip_label = Label("SK_SkipNllPAP_sk4", "")
    module = DynamicWorkQueue().reserveNext(
        _PapFetchWriter(), {"TileProcessingStrategy": "StreamK", "WorkAssignment": "DynamicWorkQueue"}, skip_label
    )
    rendered = str(module)
    primed = rendered.find("s_mov_b32 s[sgprPersistentPrefetchState], 0x80000000")
    drain = rendered.find("s[sgprTotalItems]")
    assert primed != -1 and drain != -1 and primed < drain
    assert "s[sgprNextWorkItem]" in rendered
    assert rendered.index("Reuse work or exhaustion reservation without another pop") < rendered.index("unit: queue pop")


def test_hybrid_reservation_dispatches_static_and_dynamic(monkeypatch):
    # SK5 PAP is a runtime hybrid: mode==0 keeps the SK3 PersistentIteration compare;
    # mode!=0 reuses the SK4 pop-and-prime handoff.
    monkeypatch.setattr(Hybrid, "fetchAndBroadcast", _fake_fetch)
    monkeypatch.setattr(kwa_module.Component.TileProcessingStrategy, "find", lambda writer: StreamKHybrid())
    from rocisa.code import Label

    skip_label = Label("SK_SkipNllPAP_sk5", "")
    module = Hybrid().reserveNext(
        _PapFetchWriter(), {"TileProcessingStrategy": "StreamK", "WorkAssignment": "Hybrid"}, skip_label
    )
    rendered = str(module)
    assert "s[sgprWorkAssignmentMode]" in rendered
    assert "s[sgprPersistentIteration]" in rendered
    assert "s[sgprPersistentPrefetchState]" in rendered
    assert "s[sgprNextWorkItem]" in rendered


def test_sk4_pap_setup_next_tile_uses_stashed_work_item(monkeypatch):
    monkeypatch.setattr(
        StreamKDynamic,
        "_computeNextTileIdentity",
        lambda self, writer, kernel, sidx, tpa, tpb: _module_with_comment(
            "computeNextTileIdentity", "unit: tile identity"
        ),
    )
    module = StreamKDynamic().prefetchAcrossPersistentSetupNextTile(
        _PapFetchWriter(),
        {"TileProcessingStrategy": "StreamK", "WorkAssignment": "DynamicWorkQueue"},
        {"tensorChar": "A"},
        {"tensorChar": "B"},
    )
    rendered = str(module)
    assert "s[sgprNextWorkItem]" in rendered
    assert "unit: tile identity" in rendered


# ---------------------------------------------------------------------------
# ReuseAcrossPersistent silences A for the whole reuse copy, but one load in it
# belongs to the next tile, and that tile may not be able to reuse A.
# ---------------------------------------------------------------------------
class _RapTdmNullWriter:
    """Just enough writer to render rapNullTdmDescriptorForEvenWaves.

    tdmParityPackedInArgType picks the parity form that needs no scratch, so what
    comes out is the parity compare plus the selects under test and nothing else.
    The real _emitTdmWaveParitySCC is borrowed rather than stubbed: the point of
    the test is how the batch condition composes with parity.
    """

    _emitTdmWaveParitySCC = kwa_module.KernelWriterAssembly._emitTdmWaveParitySCC

    def __init__(self, in_pap_next_tile_prefetch):
        self.states = SimpleNamespace(
            rapDropAResidentLoads=True,
            rapInPapNextTilePrefetch=in_pap_next_tile_prefetch,
            tdmParityPackedInArgType=True,
        )

    def isTdmWaveIdxLive(self, kernel):
        return False

    @contextmanager
    def allocTmpSgpr(self, num, alignment=None, tag=None):
        yield SimpleNamespace(idx=90, size=num)


def _rap_null_tdm(in_pap_next_tile_prefetch):
    return str(
        kwa_module.KernelWriterAssembly.rapNullTdmDescriptorForEvenWaves(
            _RapTdmNullWriter(in_pap_next_tile_prefetch),
            {"WavefrontSize": 32},
            "tdmAGroup0+0",
        )
    )


def test_rap_silences_a_unconditionally_for_the_reuse_copy_own_loads():
    """Every in-loop load in the reuse copy serves the tile A is resident for."""
    rendered = _rap_null_tdm(in_pap_next_tile_prefetch=False)
    assert "s[sgprtdmAGroup0+0]" in rendered
    for batchState in ("sgprRAPResidentBatch", "sgprWorkGroup2"):
        assert batchState not in rendered, (
            "an in-loop load consulted %s; the tile it serves cannot have changed "
            "batch mid-iteration" % batchState
        )


def test_rap_lets_the_next_tile_prefetch_fetch_a_when_the_batch_changes():
    """The last load in the reuse copy is PAP's, and it feeds the next tile.

    Silencing A there is only right while that tile goes on reusing the resident
    registers. When it changes batch the RAP_IterN guard diverts it to the fill
    copy, which computes from whatever this prefetch left behind -- so A has to be
    fetched. Silencing it unconditionally is invisible to any MX test (the client
    generator gives every batch the same A) and to any single-batch test, which is
    how it survived the first round of this fix.

    WorkGroup2 is the next tile's here: prefetchAcrossPersistent raises the flag
    only between setupNextTile and papRestoreCurrentTileIdentity.
    """
    rendered = _rap_null_tdm(in_pap_next_tile_prefetch=True)
    assert "s[sgprWorkGroup2], s[sgprRAPResidentBatch]" in rendered
    assert "s[sgprtdmAGroup0+0]" in rendered


# PrefetchAcrossPersistent against the non-default TDM descriptor groupings.


# Snapshot defaultSolution at import so construction is not order-dependent.
_PRISTINE_TDMFUSE_SOLUTION = copy.deepcopy(dict(defaultSolution))


# MIWaveGroup [2,2] -> NumWaves 4 (TDMFuse=2's split; TDMFuse=1 needs >= 2 waves).
_MI_W4 = [16, 16, 128, 1, 1, 2, 4, 2, 2]


# Canonical PAP grouping-reject phrase: only a lone scale on a data tensor's set is refused.
_PAP_GROUPING_MSG = "refreshes MX scale descriptors only as a pair"


# ---------------------------------------------------------------------------
# Pure predicate: derived from the grouping, not from the TDMFuse integer.
# ---------------------------------------------------------------------------
def _ks(fuse=1, **ov):
    ks = {
        "TDMFuse": fuse,
        "TDMInst": 3,
        "TDMSplit": False,
        "enableTDMA": True,
        "enableTDMB": True,
        "NumWaves": 4,
        "UseSubtileImpl": False,
        "PrefetchGlobalRead": 2,
        "PrefetchGlobalReadA": -1,
        "PrefetchGlobalReadB": -1,
        "ProblemType": {"MXBlockA": 32, "MXBlockB": 32},
    }
    ks.update(ov)
    return ks


@pytest.mark.parametrize("fuse, expected", [(0, ()), (1, (("A", "MXSA"), ("MXSB", "B"))),
                                            (2, (("A", "MXSA", "MXSB"),))])
def test_shared_sets_read_off_the_resolved_grouping(fuse, expected):
    assert tdmScaleSharesDataSet(_ks(fuse=fuse)) == expected


@pytest.mark.parametrize("fuse", [1, 2, 3])
@pytest.mark.parametrize("num_waves", [2, 4])
def test_reason_is_silent_for_scales_on_both_tensors(fuse, num_waves):
    """PAP refreshes and re-banks every set through its owner, scale sets included."""
    ks = _ks(fuse=fuse, NumWaves=num_waves)
    assert tdmScaleSharesDataSet(ks), "the cell must exercise a shared set"
    assert tdmPapRejectReason(ks) is None


@pytest.mark.parametrize("fuse", [1, 2, 3])
@pytest.mark.parametrize("live", [{"MXBlockA": 32, "MXBlockB": 0}, {"MXBlockA": 0, "MXBlockB": 32}])
def test_reason_names_the_grouping_of_a_lone_scale_on_a_data_set(fuse, live):
    reason = tdmPapRejectReason(_ks(fuse=fuse, ProblemType=live))
    assert _PAP_GROUPING_MSG in reason
    assert tdmGrouping(_ks(fuse=fuse, ProblemType=live)).name in reason
    # The old text claimed a missing tdmMXSAMXSBIncs, which TDMFuse=1 allocates.
    assert "tdmMXSAMXSBIncs" not in reason


def test_default_grouping_is_not_rejected():
    assert tdmPapRejectReason(_ks(fuse=0)) is None


@pytest.mark.parametrize("decline", [{"NumWaves": 1}, {"TDMSplit": True},
                                     {"UseSubtileImpl": True}, {"TDMInst": 1}])
def test_declined_grouping_answers_with_the_fallback(decline):
    """A grouping TDMFuse asked for but the predicates declined shares nothing."""
    assert tdmPapRejectReason(_ks(fuse=1, **decline)) is None
    assert tdmPapRejectReason(_ks(fuse=2, **decline)) is None


def test_scale_less_types_share_nothing():
    """No live MXSA/MXSB means no scale to seat on a data tensor's set."""
    assert tdmPapRejectReason(_ks(fuse=1, ProblemType={"MXBlockA": 0, "MXBlockB": 0})) is None


@pytest.mark.parametrize("live", [{"MXBlockA": 32, "MXBlockB": 0}, {"MXBlockA": 0, "MXBlockB": 32}])
def test_a_lone_scale_still_rides_a_data_tensors_set(live):
    """Solution rejects MX on one tensor first; the grouping alone keeps the live scale paired."""
    assert _PAP_GROUPING_MSG in tdmPapRejectReason(_ks(fuse=1, ProblemType=live))


@pytest.mark.parametrize("row, shares", [("MX_AB", False), ("paired", True),
                                         ("A_MX", True), ("B_MX", True)])
def test_every_grouping_row_answers_without_a_new_branch(row, shares):
    assert bool(tdmScaleSharesDataSet(_ks(), TDM_GROUPS[row])) is shares


# ---------------------------------------------------------------------------
# Solution level: real gfx1250 caps + assembler, assignDerivedParameters
# end-to-end. Mirrors test_halfplr_streamk_rejects.py's harness.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gfx1250_iim():
    from tensilelite.Common.Architectures import gfxToIsa
    from tensilelite.Common.Capabilities import makeIsaInfoMap
    from tensilelite.Toolchain.Validators import validateToolchain

    cxx = validateToolchain("amdclang++")
    isa = gfxToIsa("gfx1250")
    iim = makeIsaInfoMap([isa], cxx)
    if not iim[isa].asmCaps["SupportedISA"]:
        pytest.skip("amdclang++ in this environment does not support gfx1250")
    return iim


@pytest.fixture(scope="module")
def assembler():
    from tensilelite.Toolchain.Assembly import makeAssemblyToolchain
    from tensilelite.Toolchain.Validators import validateToolchain, ToolchainDefaults

    cxx = validateToolchain("amdclang++")
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    return makeAssemblyToolchain(cxx, bundler, "default").assembler


@pytest.fixture(scope="module")
def _gp_gfx1250(gfx1250_iim):
    from tensilelite.Common.GlobalParameters import assignGlobalParameters

    saved_gp = copy.deepcopy(dict(globalParameters))
    saved_vp = copy.deepcopy(dict(validParameters))
    saved_ds = copy.deepcopy(dict(defaultSolution))
    defaultSolution.clear()
    defaultSolution.update(copy.deepcopy(_PRISTINE_TDMFUSE_SOLUTION))
    assignGlobalParameters({}, gfx1250_iim)
    yield
    globalParameters.clear()
    globalParameters.update(saved_gp)
    validParameters.clear()
    validParameters.update(saved_vp)
    defaultSolution.clear()
    defaultSolution.update(saved_ds)


def _make_params(gfx1250_iim, mi=None, **overrides):
    """Smallest PAP+TDM shape: TN MXF8F4 StreamK=3, StreamKForceDPOnly=1."""
    from tensilelite.Common.Architectures import gfxToIsa
    from tensilelite.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    isa = gfxToIsa("gfx1250")
    mi = mi or _MI_W4
    pt = overrides.pop("ProblemType", {})
    problem_type = {
        "OperationType": "GEMM", "MacDataTypeA": "F8", "MacDataTypeB": "F4",
        "DataType": "F8", "DestDataType": "s", "ComputeDataType": "s",
        "HighPrecisionAccumulate": True, "TransposeA": True, "TransposeB": False,
        "UseBeta": True, "Batched": True, "MXBlockA": 32, "MXBlockB": 32,
        "DataTypeMXSA": "E8", "DataTypeMXSB": "E8",
    }
    problem_type.update(pt)
    params = {
        "ProblemType": problem_type, "ISA": isa, "MatrixInstruction": mi,
        "WorkGroup": [16, 16, 1], "WavefrontSize": 32, "DepthU": 256,
        "KernelLanguage": "Assembly", "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
        "ScheduleIterAlg": 4, "StaggerU": 0, "GlobalSplitU": 1, "InnerUnroll": 1,
        "TransposeLDS": -1, "LdsPadA": -1, "LdsPadB": -1,
        "LdsBlockSizePerPadA": -1, "LdsBlockSizePerPadB": -1, "1LDSBuffer": 0,
        "VectorWidthA": -1, "VectorWidthB": -1, "StoreVectorWidth": -1,
        "GlobalReadVectorWidthA": -1, "GlobalReadVectorWidthB": -1,
        "LocalReadVectorWidth": -1, "SourceSwap": False, "ExpandPointerSwap": False,
        "GlobalSplitUAlgorithm": "MultipleBuffer", "TDMInst": 3, "LDSTrInst": False,
        "StreamK": 3, "StreamKForceDPOnly": 1, "PrefetchAcrossPersistent": 0,
        "UseSubtileImpl": False, "StoreRemapVectorWidth": 0,
        "DirectToVgprA": False, "DirectToVgprB": False,
        "DirectToVgprSparseMetadata": False, "WorkGroupMapping": 1,
        "TDMFuse": 0, "TDMSplit": False, "InitCIterWmma": 0,
    }
    params.update(overrides)
    params.update(matrixInstructionToMIParameters(
        mi, isa, params["WavefrontSize"], problem_type, params["WorkGroup"], gfx1250_iim))
    return params


def _derive(gfx1250_iim, assembler, capsys, **overrides):
    from tensilelite.SolutionStructs.Solution import Solution
    sol = Solution(_make_params(gfx1250_iim, **overrides), False, True, False,
                   assembler, gfx1250_iim)
    return sol, capsys.readouterr().out


def test_pap_on_the_default_grouping_is_accepted(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    """Default grouping accepts PAP."""
    sol, out = _derive(gfx1250_iim, assembler, capsys, PrefetchAcrossPersistent=1)
    assert sol.get("Valid") is True, "expected accept, rejected with: %r" % out
    assert sol.get("PrefetchAcrossPersistent") == 1, \
        "PrefetchAcrossPersistent was reset"
    assert sol.get("LdsOffsetA_Blk") != 0, \
        "LdsOffsetA_Blk == 0 folds the LDS-bank helpers out entirely"


@pytest.mark.parametrize("fuse", [1, 2])
def test_pap_with_a_shared_scale_set_is_accepted(_gp_gfx1250, gfx1250_iim, assembler, capsys, fuse):
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       PrefetchAcrossPersistent=1, TDMFuse=fuse)
    assert sol.get("Valid") is True, "TDMFuse=%d + PAP was rejected: %r" % (fuse, out)
    assert _PAP_GROUPING_MSG not in out
    # The requested row is the one in effect, not a declined fallback to TDMFuse=0's.
    assert tdmScaleSharesDataSet(sol), "TDMFuse=%d fell back to %s" % (fuse, tdmGrouping(sol).name)


@pytest.mark.parametrize("fuse", [1, 2])
def test_the_same_grouping_without_pap_is_still_accepted(_gp_gfx1250, gfx1250_iim, assembler, capsys, fuse):
    """The rejection is about PAP, not about the grouping."""
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       PrefetchAcrossPersistent=0, TDMFuse=fuse)
    assert sol.get("Valid") is True, "expected accept, rejected with: %r" % out
    assert _PAP_GROUPING_MSG not in out


def test_tdmfuse_own_guards_keep_precedence(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    sol, out = _derive(gfx1250_iim, assembler, capsys, PrefetchAcrossPersistent=1,
                       TDMFuse=2, mi=[16, 16, 128, 1, 1, 2, 4, 2, 4])
    assert sol.get("Valid") is False
    assert "TDMFuse=2 with MX scales on both tensors supports 2 or 4 waves; got NumWaves=8" in out
    assert _PAP_GROUPING_MSG not in out


# Non-idempotent `s_add_u32 dst, dst, x` sites. papTdmSetTailLdsBank is excluded:
# it normalizes to bank 0 first, so applying it more than once is by design.
_NON_IDEMPOTENT_BANK_SITES = {
    "papTdmRestoreLdsBank": ("shift A/B descriptor to PAP bank",
                             "shift MX descriptor to PAP bank",
                             "shift tdmBGroup0 descriptor set to PAP bank",
                             "shift tdmMXSBGroup0 descriptor set to PAP bank"),
    "papTdmUpdateDescriptor": ("restore PAP LDS bank after descriptor refresh",),
}


_LDS_ADDR_SYMBOLS = ("sgprtdmAGroup0+1", "sgprtdmBGroup0+1",
                     "sgprtdmMXSAGroup0+1", "sgprtdmMXSBGroup0+1")


def _resolve_sets(asm):
    """symbol -> physical register, honouring the first .set and skipping UNDEF."""
    raw = {}
    for m in re.finditer(r"^\s*\.set\s+(\S+?),\s*(.+?)\s*$", asm, re.M):
        name, value = m.group(1), m.group(2).strip()
        if name in raw or value == "UNDEF":
            continue
        raw[name] = value

    def resolve(name, depth=0):
        if depth > 40:
            return None
        value = raw.get(name)
        if value is None:
            return None
        if re.fullmatch(r"-?\d+", value):
            return int(value)
        m = re.fullmatch(r"([A-Za-z_]\w*)\s*\+\s*(\d+)", value)
        if m:
            base = resolve(m.group(1), depth + 1)
            return None if base is None else base + int(m.group(2))
        return resolve(value, depth + 1)

    out = {name: resolve(name) for name in raw}

    def phys(symbol):
        m = re.fullmatch(r"(\w+?)\+(\d+)", symbol)
        if m:
            base = out.get(m.group(1))
            return None if base is None else base + int(m.group(2))
        return out.get(symbol)

    return out, phys


def _emit_asm(gfx1250_iim, assembler, **overrides):
    """(solution, assembly text or None). CPU-only; no GPU is touched."""
    import shutil
    import rocisa
    from tensilelite.Common.Types import DebugConfig
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly
    from tensilelite.SolutionStructs.Naming import getKernelFileBase
    from tensilelite.TensileCreateLibrary.Run import (generateKernelObjectsFromSolutions,
                                                  processKernelSource)
    from tensilelite.Tests.rocisa_test_state import preserve_rocisa_kernel_state

    sol = Solution(_make_params(gfx1250_iim, **overrides), False, True, False,
                   assembler, gfx1250_iim)
    if not sol.get("Valid"):
        return sol, None
    with preserve_rocisa_kernel_state():
        kwa = KernelWriterAssembly(assembler, DebugConfig())
        pieces = []
        for kernel in generateKernelObjectsFromSolutions([sol]):
            ri = rocisa.rocIsa.getInstance()
            ri.init(tuple(kernel["ISA"]),
                    shutil.which("amdclang++") or "/usr/bin/amdclang++")
            ri.setKernel(tuple(kernel["ISA"]), kernel["WavefrontSize"])
            kernel.duplicate = False
            kernel["BaseName"] = getKernelFileBase(False, kernel)
            res = processKernelSource(kwa, ri.getData(), ri.getOutputOptions(), False, kernel)
            src = res.src
            if isinstance(src, (bytes, bytearray)):
                src = src.decode(errors="replace")
            pieces.append(src or "")
    return sol, "\n".join(pieces)


@pytest.mark.parametrize("fuse", [0, 1, 2, 3])
def test_pap_shifts_each_descriptor_lds_bank_exactly_once(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse):
    """Every non-idempotent bank shift lands on its own physical register."""
    sol, asm = _emit_asm(gfx1250_iim, assembler, PrefetchAcrossPersistent=1, TDMFuse=fuse)
    capsys.readouterr()
    if asm is None:
        pytest.skip("TDMFuse=%d + PAP is refused, nothing to emit" % fuse)
    _, phys = _resolve_sets(asm)

    shifted = {}
    for line in asm.splitlines():
        m = re.search(r"s_add_u32 s\[(sgprtdm\w+Group0\+1)\], s\[\1\], \S+\s*//\s*(.*)",
                      line)
        if not m:
            continue
        comment = m.group(2).strip()
        for site, comments in _NON_IDEMPOTENT_BANK_SITES.items():
            if comment in comments:
                shifted.setdefault(site, []).append((m.group(1), phys(m.group(1))))

    for site, hits in shifted.items():
        regs = [reg for _, reg in hits]
        duplicated = sorted({r for r in regs if regs.count(r) > 1})
        assert not duplicated, (
            "%s shifts s%s more than once via aliased spellings %s" %
            (site, duplicated, [sym for sym, _ in hits]))

    # Complement: aliasing also drops the sibling's shift entirely.
    if "papTdmRestoreLdsBank" in shifted:
        allocated = {phys(s) for s in _LDS_ADDR_SYMBOLS if phys(s) is not None}
        assert {reg for _, reg in shifted["papTdmRestoreLdsBank"]} == allocated, (
            "papTdmRestoreLdsBank shifted %s but the kernel allocates %s" %
            (sorted({reg for _, reg in shifted["papTdmRestoreLdsBank"]}), sorted(allocated)))


@pytest.mark.parametrize("fuse", [0, 1, 2, 3])
def test_pap_never_names_an_unallocated_tdm_increment(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse):
    """Every tdm*Incs the kernel reads has a defineSgpr behind it."""
    sol, asm = _emit_asm(gfx1250_iim, assembler, PrefetchAcrossPersistent=1, TDMFuse=fuse)
    capsys.readouterr()
    if asm is None:
        pytest.skip("TDMFuse=%d + PAP is refused, nothing to emit" % fuse)
    resolved, _ = _resolve_sets(asm)
    referenced = set(re.findall(r"\bsgprtdm\w*Incs\b", asm))
    undefined = sorted(s for s in referenced if resolved.get(s) is None)
    assert not undefined, "referenced with no .set: %s" % undefined


# Persistent vector epilogues coexist with prefetched compute data in LDS.
_BF16_VECTOR_EPILOGUE = {
    "DataType": "B", "MacDataTypeA": "B", "MacDataTypeB": "B",
    "DestDataType": "B", "MXBlockA": 0, "MXBlockB": 0,
}
_BF16_VECTOR_MI = [16, 16, 32, 1, 1, 8, 8, 2, 2]


@pytest.mark.parametrize("pap", [0, 1])
@pytest.mark.parametrize("bias,sav", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_persistent_vector_scratch_is_separate_only_with_pap(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, pap, bias, sav):
    common = dict(mi=_BF16_VECTOR_MI, DepthU=128,
                  PrefetchAcrossPersistent=pap, SuppressNoLoadLoop=False)
    baseline, reason = _derive(gfx1250_iim, assembler, capsys,
                              ProblemType=_BF16_VECTOR_EPILOGUE, **common)
    assert baseline.get("Valid"), reason
    sol, reason = _derive(gfx1250_iim, assembler, capsys,
                         ProblemType=dict(_BF16_VECTOR_EPILOGUE,
                                          UseBias=bias, UseScaleAlphaVec=sav), **common)
    assert sol.get("Valid"), reason
    assert sol["_PersistentVectorEpilogueLds"] == bool(bias or sav)
    if pap and (bias or sav):
        assert sol["_SeparateEpilogueLds"]
        # Start immediately after BOTH compute banks, rounded to 16 bytes.
        assert sol["LdsOffsetBias"] == (baseline["LdsNumBytes"] + 15) // 16 * 16
        assert sol["LdsOffsetBiasNonGSU"] == sol["LdsOffsetBias"]
        assert sol["LdsOffsetBiasGSU"] == sol["LdsOffsetBias"]
        assert sol["LdsNumBytes"] == sol["LdsOffsetBias"] + 256 * 4 * (bias + sav)
    else:
        assert not sol["_SeparateEpilogueLds"]
        assert sol["LdsOffsetBias"] == 0
        assert sol["LdsNumBytes"] == baseline["LdsNumBytes"]


def _assert_lds_sync_at_barrier(asm, comment):
    """Comments locate boundaries; the emitted instructions prove the handoff."""
    instructions = []
    boundaries = 0
    for line in asm.splitlines():
        code, _, annotation = line.partition("//")
        if code.strip():
            instructions.append(" ".join(code.split()))
        if annotation.strip() == comment:
            assert instructions[-3:] == [
                "s_wait_dscnt 0", "s_barrier_signal -1", "s_barrier_wait -1",
            ], "incomplete or out-of-order LDS synchronization at %s" % comment
            boundaries += 1
    assert boundaries, "missing LDS synchronization boundary: %s" % comment


@pytest.mark.parametrize("pap,sia", [(0, 0), (0, 4), (1, 0), (1, 4)])
def test_vector_lds_wait_is_late_and_does_not_drain_pap(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, pap, sia):
    sol, asm = _emit_asm(gfx1250_iim, assembler, mi=[16, 16, 32, 1, 1, 4, 4, 2, 2],
                        DepthU=128, ScheduleIterAlg=sia,
                        PrefetchAcrossPersistent=pap, SuppressNoLoadLoop=False,
                        ProblemType=dict(_BF16_VECTOR_EPILOGUE,
                                         UseBias=1, UseScaleAlphaVec=1))
    assert sol.get("Valid"), capsys.readouterr().out
    assert asm
    start = asm.index("label_PersistentLoopStart:")
    first_tdm = asm.index("tensor_load", start)
    assert "reuse vector epilogue" not in asm[start:first_tdm]
    # Examine the staging block, after all compute/tail code. A blanket
    # tensor wait here would destroy the intended PAP overlap.
    first_store = asm.index("store bias")
    summation_ends = list(re.finditer(r"^label_Summation_End\w*:", asm[:first_store], re.M))
    assert summation_ends, "missing summation/epilogue boundary"
    start = summation_ends[-1].end()
    end = asm.index("store bias", start)
    staging = asm[start:end]
    assert "LDS write barrier" in asm[start:]
    assert "s_wait_tensorcnt" not in asm[start:]
    handoff = "hand epilogue LDS back to persistent compute"
    if pap:
        # Separate storage lets PAP continue through both epilogue barriers.
        assert handoff not in asm
        assert staging.index("buffer_load") < staging.index("reuse vector epilogue LDS scratch")
        _assert_lds_sync_at_barrier(asm[start:], "reuse vector epilogue LDS scratch")
    else:
        # Shared storage is handed back only after the whole epilogue, beyond
        # the main-loop barrier pass. No wait is moved to the next tile's entry.
        assert "reuse vector epilogue LDS scratch" not in asm
        end_gw = re.search(r"^label_GW_End:", asm, re.M).start()
        close = asm.index("label_PersistentLoopClose:", end_gw)
        backedge = asm.index("label_PersistentLoopStart", close)
        _assert_lds_sync_at_barrier(asm[close:backedge], handoff)


def test_nonpersistent_vector_epilogue_keeps_its_existing_lds_layout(
        _gp_gfx1250, gfx1250_iim, assembler, capsys):
    common = dict(mi=_BF16_VECTOR_MI, DepthU=128, StreamK=0, StreamKForceDPOnly=0)
    baseline, reason = _derive(gfx1250_iim, assembler, capsys,
                              ProblemType=_BF16_VECTOR_EPILOGUE, **common)
    assert baseline.get("Valid"), reason
    sol, reason = _derive(gfx1250_iim, assembler, capsys,
                         ProblemType=dict(_BF16_VECTOR_EPILOGUE,
                                          UseBias=1, UseScaleAlphaVec=1), **common)
    assert sol.get("Valid"), reason
    assert not sol["_SeparateEpilogueLds"]
    assert sol["LdsOffsetBias"] == 0
    assert sol["LdsNumBytes"] == baseline["LdsNumBytes"]


# PAP and RAP on TDMFuse 1-3.


# MIWaveGroup [2,1] -> NumWaves 2: TDMFuse 2/3 put MXSA and MXSB on wave 1, MXSB on a set of its own.
_MI_W2 = [16, 16, 128, 1, 1, 2, 4, 2, 1]
_ROW_OF_FUSE = {1: "paired", 2: "A_MX", 3: "B_MX"}

# (TDMFuse, MatrixInstruction) of every PAP arrangement with MX scales on both tensors.
_PAP_MX_BOTH_CELLS = [(1, _MI_W4), (2, _MI_W4), (3, _MI_W4), (1, _MI_W2), (2, _MI_W2), (3, _MI_W2)]

# The PAP shape test_persistent_vector_scratch_is_separate_only_with_pap accepts without MX scales.
_NO_MX_COMMON = dict(ProblemType=_BF16_VECTOR_EPILOGUE, mi=_BF16_VECTOR_MI, DepthU=128,
                     SuppressNoLoadLoop=False)
# Emission takes a 64x64 tile on four waves: at the 256x256 tile the PAP and RAP kernels overflow the VGPRs.
_NO_MX_EMIT = dict(_NO_MX_COMMON, mi=[16, 16, 32, 1, 1, 2, 2, 2, 2])


class _PapSetRefreshWriter:
    """Records the pairs the PAP refresh and the PAP tail reset program, in order."""

    def __init__(self):
        self.calls = []
        self.labels = _StubLabels()
        self._next_tmp_sgpr = 300

    @contextmanager
    def allocTmpSgpr(self, size, alignment=1, tag=""):
        base = self._next_tmp_sgpr
        self._next_tmp_sgpr += size + alignment
        yield SimpleNamespace(idx=base, size=size)

    def papTdmRefreshPairs(self, kernel, tpa, tpb):
        return kwa_module.KernelWriterAssembly.papTdmRefreshPairs(self, kernel, tpa, tpb)

    def papTdmDescriptorSetOwners(self, kernel):
        return kwa_module.KernelWriterAssembly.papTdmDescriptorSetOwners(self, kernel)

    def tdmDescriptorSetOwner(self, kernel, tc):
        return kwa_module.KernelWriterAssembly.tdmDescriptorSetOwner(self, kernel, tc)

    def isTdmWaveSeparated(self, kernel):
        return kwa_module.KernelWriterAssembly.isTdmWaveSeparated(self, kernel)

    def _record(self, step, tpa, tpb):
        self.calls.append((step, tpa["tensorChar"], tpb["tensorChar"]))
        return Module(step)

    def papTdmRecomputeWaveIdx(self, kernel, wave_idx_sgpr):
        return Module("recompute WaveIdx")

    def initTDMDescriptorWaveSeparated(self, kernel, tpa, tpb, wave_idx_sgpr="WaveIdx"):
        return self._record("init", tpa, tpb)

    def tdmGlobalOffsetWaveSeparated(self, kernel, tpa, tpb, wave_idx_sgpr="WaveIdx"):
        return self._record("offset", tpa, tpb)

    def tdmApplyTileKOffsetWaveSeparated(self, kernel, tpa, tpb):
        return self._record("k offset", tpa, tpb)

    def tdmApplyTileTailOffsetWaveSeparated(self, kernel, tpa, tpb):
        return self._record("tail offset", tpa, tpb)

    def resetTDMDescriptorForTailWaveSeparated(self, kernel, tpa, tpb):
        return self._record("tail reset", tpa, tpb)

    def papTdmSelectTailLdsBank(self, kernel, dst_sgpr):
        return Module("select tail bank")

    def papTdmSetTailLdsBank(self, kernel, lds_addr_sgpr_name, tail_bank_sgpr):
        self.calls.append(("tail bank", lds_addr_sgpr_name))
        return Module("tail bank")


def _pap_set_kernel(fuse, num_waves, mx):
    blocks = {"MXBlockA": 32, "MXBlockB": 32} if mx else {"MXBlockA": 0, "MXBlockB": 0}
    return _ks(fuse=fuse, NumWaves=num_waves, ProblemType=blocks, LdsOffsetA_Blk=0x8000,
               TileProcessingStrategy="StreamK", WorkAssignment="StaticGrid")


def _pap_set_calls(kernel, entry, monkeypatch):
    """[(writer calls, rendered module)] of the A/B call and, with MX scales, the scale call."""
    monkeypatch.setattr(kwa_module.TensorDataMoverLoad, "find", lambda writer: _StubTdmComp())
    tpa = {"tensorChar": "A", "MX": {"tensorChar": "MXSA"}}
    tpb = {"tensorChar": "B", "MX": {"tensorChar": "MXSB"}}
    pairs = [(tpa, tpb)]
    if kernel["ProblemType"]["MXBlockA"]:
        pairs.append((tpa["MX"], tpb["MX"]))
    writer, out = _PapSetRefreshWriter(), []
    for pa, pb in pairs:
        writer.calls = []
        module = entry(writer, kernel, pa, pb)
        out.append((writer.calls, str(module)))
    return out


# (TDMFuse, NumWaves, MX scales on both tensors, descriptor set owners); TDMFuse=0 is the control.
_PAP_SET_CELLS = [
    (0, 4, True, ["A", "MXSA"]), (0, 4, False, ["A"]),
    (1, 4, True, ["A", "B"]), (2, 4, True, ["A", "B"]), (3, 4, True, ["A", "B"]),
    (1, 2, True, ["A", "B"]), (2, 2, True, ["A", "B", "MXSB"]), (3, 2, True, ["A", "B", "MXSB"]),
    (1, 4, False, ["A", "B"]), (2, 4, False, ["A", "B"]), (3, 4, False, ["A", "B"]),
]


def _expected_pap_calls(fuse, mx, steps, tail):
    """Per call: TDMFuse=0 programs its own pair; separate sets all go through the A/B call."""
    pairs = [("A", "B")] + ([("MXSA", "MXSB")] if mx else [])
    if fuse == 0:
        return [[(step,) + pair for step in steps] + tail(pair) for pair in pairs]
    return [[(step,) + pair for step in steps for pair in pairs] + tail(None)] + ([[]] if mx else [])


@pytest.mark.parametrize("fuse, num_waves, mx, owners", _PAP_SET_CELLS)
def test_pap_refresh_rebuilds_every_set_before_offsetting_any(monkeypatch, fuse, num_waves, mx, owners):
    """The scale pair of a separate-set row is refreshed on the A/B call, each bank restored once."""
    kernel = _pap_set_kernel(fuse, num_waves, mx)
    assert _PapSetRefreshWriter().papTdmDescriptorSetOwners(kernel) == owners
    out = _pap_set_calls(kernel, kwa_module.KernelWriterAssembly.papTdmUpdateDescriptor, monkeypatch)
    expected = _expected_pap_calls(fuse, mx, ("init", "offset", "k offset"), lambda pair: [])
    assert [calls for calls, _ in out] == expected
    restored = [re.findall(r"s_add_u32 s\[sgprtdm(\w+)Group0\+1\][^\n]*restore PAP LDS bank", text)
                for _, text in out]
    if fuse == 0:
        assert restored == [["A"]] + ([["MXSA"]] if mx else [])
    else:
        assert restored == [owners] + ([[]] if mx else [])


@pytest.mark.parametrize("fuse, num_waves, mx, owners", _PAP_SET_CELLS)
def test_pap_tail_reset_rebuilds_every_set_before_offsetting_any(monkeypatch, fuse, num_waves, mx, owners):
    """A tail offset on a set the scale call has not rebuilt yet is lost to that rebuild."""
    kernel = _pap_set_kernel(fuse, num_waves, mx)
    out = _pap_set_calls(kernel, kwa_module.KernelWriterAssembly.papResetTDMDescriptorForTailWaveSeparated,
                         monkeypatch)

    def tail(pair):
        tensors = pair if pair is not None else owners
        return [("tail bank", "tdm%sGroup0+1" % tc) for tc in tensors]

    expected = _expected_pap_calls(fuse, mx, ("init", "offset", "tail offset", "tail reset"), tail)
    assert [calls for calls, _ in out] == expected


@pytest.mark.parametrize("fuse", [1, 2, 3])
@pytest.mark.parametrize("in_pap_next_tile_prefetch", [False, True])
def test_rap_nulls_an_every_wave_a_set_on_every_wave(fuse, in_pap_next_tile_prefetch):
    """Without MX scales A rides every wave on a set of its own, so no wave parity selects it."""
    kernel = dict(_ks(fuse=fuse, ProblemType={"MXBlockA": 0, "MXBlockB": 0}), WavefrontSize=32)
    rendered = str(kwa_module.KernelWriterAssembly.rapNullTdmDescriptorForEvenWaves(
        _RapTdmNullWriter(in_pap_next_tile_prefetch), kernel, "tdmAGroup0+0"))
    assert "s[sgprArgType]" not in rendered, "an every-wave A set was nulled by wave parity"
    assert ("s[sgprWorkGroup2], s[sgprRAPResidentBatch]" in rendered) is in_pap_next_tile_prefetch
    assert "s[sgprtdmAGroup0+0], 0" in rendered


def test_rap_keeps_wave_parity_on_the_default_grouping():
    kernel = dict(_ks(fuse=0, ProblemType={"MXBlockA": 0, "MXBlockB": 0}), WavefrontSize=32)
    rendered = str(kwa_module.KernelWriterAssembly.rapNullTdmDescriptorForEvenWaves(
        _RapTdmNullWriter(False), kernel, "tdmAGroup0+0"))
    assert "s[sgprArgType]" in rendered


@pytest.mark.parametrize("fuse, mi", _PAP_MX_BOTH_CELLS)
def test_pap_accepts_scales_on_both_tensors_on_tdmfuse_1_3(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse, mi):
    sol, out = _derive(gfx1250_iim, assembler, capsys, PrefetchAcrossPersistent=1, TDMFuse=fuse, mi=mi)
    assert sol.get("Valid") is True, "TDMFuse=%d + PAP was rejected: %r" % (fuse, out)
    assert sol.get("PrefetchAcrossPersistent") == 1
    assert tdmGrouping(sol).name == _ROW_OF_FUSE[fuse]


@pytest.mark.parametrize("fuse", [1, 2, 3])
@pytest.mark.parametrize("pap, rap", [(1, 0), (0, 1), (1, 1)])
def test_pap_and_rap_accept_tdmfuse_1_3_without_scales(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse, pap, rap):
    # RAP needs NoTailLoop; PAP alone keeps the K tail so the tail reset is emitted.
    overrides = dict(_NO_MX_COMMON, PrefetchAcrossPersistent=pap, ReuseAcrossPersistent=rap,
                     AssertSummationElementMultiple=128 if rap else 1)
    control, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=0, **overrides)
    if not control.get("Valid"):
        pytest.skip("the TDMFuse=0 control is refused: %r" % out)
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse, **overrides)
    assert sol.get("Valid") is True, "TDMFuse=%d without MX scales was rejected: %r" % (fuse, out)
    assert (sol.get("PrefetchAcrossPersistent"), sol.get("ReuseAcrossPersistent")) == (pap, rap)
    assert tdmGrouping(sol).name == _ROW_OF_FUSE[fuse]


def _owner_hits(asm, phys, comments):
    """Physical registers of the `s_add_u32 tdm*Group0+n` lines carrying one of `comments`."""
    hits = []
    for line in asm.splitlines():
        # StinkyTofu may fold the copy before an add into the add's first source.
        m = re.search(r"s_add_u32 s\[(sgprtdm\w+Group0\+\d)\], \S+, \S+\s*//\s*(.*)", line)
        if m and m.group(2).strip() in comments:
            hits.append(phys(m.group(1)))
    return hits


_PAP_EMIT_CELLS = ([dict(TDMFuse=fuse, mi=mi) for fuse, mi in _PAP_MX_BOTH_CELLS]
                   + [dict(_NO_MX_EMIT, TDMFuse=fuse) for fuse in (1, 2, 3)]
                   + [dict(TDMFuse=0), dict(_NO_MX_EMIT, TDMFuse=0)])


@pytest.mark.parametrize("cell", _PAP_EMIT_CELLS,
                         ids=lambda c: "f%d_%s_%s" % (c["TDMFuse"], "bf16" if "ProblemType" in c else "mx",
                                                      "w2" if c.get("mi") == _MI_W2 else "w4"))
def test_pap_moves_each_descriptor_set_once(_gp_gfx1250, gfx1250_iim, assembler, capsys, cell):
    """The PAP bank restore, refresh and tail offset reach every live set, each the same number of times."""
    sol, asm = _emit_asm(gfx1250_iim, assembler, PrefetchAcrossPersistent=1, **cell)
    out = capsys.readouterr().out
    assert asm, "rejected or not emitted: %r" % out
    resolved, phys = _resolve_sets(asm)
    for comments, word, once in ((_NON_IDEMPOTENT_BANK_SITES["papTdmRestoreLdsBank"], 1, True),
                                 (_NON_IDEMPOTENT_BANK_SITES["papTdmUpdateDescriptor"], 1, True),
                                 (("Apply StreamK tail K-offset to TDM global addr",), 2, False)):
        allocated = {phys("sgprtdm%sGroup0+%d" % (tc, word)) for tc in ("A", "B", "MXSA", "MXSB")} - {None}
        hits = _owner_hits(asm, phys, comments)
        if not hits and not once:
            # No K tail in this shape, so no PAP tail reset to offset.
            continue
        if once and not sol.get("LdsOffsetA_Blk"):
            # LdsOffsetA_Blk == 0 folds the bank helpers out: no set has a bank to move to.
            assert not [line for line in asm.splitlines() if any(c in line for c in comments)], comments
            continue
        counts = {reg: hits.count(reg) for reg in hits}
        assert set(counts) == allocated, (comments, sorted(counts), sorted(allocated))
        assert len(set(counts.values())) == 1 and (not once or set(counts.values()) == {1}), (comments, counts)
    undefined = sorted(s for s in set(re.findall(r"\bsgprtdm\w*Incs\b", asm)) if resolved.get(s) is None)
    assert not undefined, "referenced with no .set: %s" % undefined


_RAP_NO_MX = dict(_NO_MX_EMIT, ReuseAcrossPersistent=1, AssertSummationElementMultiple=128)


def _labelsDefinedTwice(text):
    """Labels the assembly text defines more than once, sorted."""
    import re
    from collections import Counter
    counts = Counter(re.findall(r"^([A-Za-z_.$][\w.$]*):", text, re.M))
    return sorted(name for name, n in counts.items() if n > 1)


@contextmanager
def _stinkyTofuInputs():
    """Collects the text of every module handed to StinkyTofu inside the block.

    StinkyTofu's CFG builder aborts the process on a module that defines a label twice, so such a
    module is stopped here, before the pipeline sees it, and the test fails instead of the worker.
    """
    import rocisa
    original = rocisa.toStinkyTofuModule
    inputs = []

    def checked(body, *args, **kwargs):
        text = str(body)
        inputs.append(text)
        twice = _labelsDefinedTwice(text)
        if twice:
            raise AssertionError("labels defined twice: %s" % twice)
        return original(body, *args, **kwargs)

    rocisa.toStinkyTofuModule = checked
    try:
        yield inputs
    finally:
        rocisa.toStinkyTofuModule = original


def _assertReuseCopyOwnsItsShadowInit(text):
    """The RAP reuse copy re-emits the shadow-init block under its own names and branches only to them."""
    import re
    fill, rest = text.split("label_RAP_IterN:", 1)
    reuse = rest.split("label_RAP_StoreJoin:", 1)[0]
    assert re.search(r"^label_ShadowInitStart:", fill, re.M)
    assert set(re.findall(r"\blabel_ShadowInitStart\w*", reuse)) == {"label_ShadowInitStart_RAPIterN"}


def test_rap_with_shadow_init_emits_each_label_once(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    """TDMFuse=0 and RAP with shadow init at its default. The reuse copy re-emits the shadow-init block,
    whose labels were built from literals, so each was defined twice and StinkyTofu's CFG builder
    aborted the process; they now go through rapLabel."""
    with _stinkyTofuInputs() as inputs:
        sol, asm = _emit_asm(gfx1250_iim, assembler, TDMFuse=0, **_RAP_NO_MX)
    out = capsys.readouterr().out
    assert [_labelsDefinedTwice(text) for text in inputs] == [[]]
    assert asm, "rejected or not emitted: %r" % out
    assert not sol["ForceDisableShadowInit"]
    _assertReuseCopyOwnsItsShadowInit(inputs[0])
    assert "RAP: even wave carries A" in inputs[0]
    assert _labelsDefinedTwice(asm) == []


@pytest.mark.parametrize("fuse", [1, 2, 3])
def test_rap_without_scales_silences_a_on_every_wave(_gp_gfx1250, gfx1250_iim, assembler, capsys, fuse):
    """Without MX scales A has its own set on every wave, so keeping A resident nulls that set on
    every wave; the reuse copy keeps its own shadow-init labels here too."""
    with _stinkyTofuInputs() as inputs:
        _, asm = _emit_asm(gfx1250_iim, assembler, TDMFuse=fuse, **_RAP_NO_MX)
    out = capsys.readouterr().out
    assert [_labelsDefinedTwice(text) for text in inputs] == [[]]
    assert asm, "rejected or not emitted: %r" % out
    _assertReuseCopyOwnsItsShadowInit(inputs[0])
    assert "RAP: every wave carries A" in inputs[0]
    assert "RAP: even wave carries A" not in inputs[0]
    assert _labelsDefinedTwice(asm) == []


@pytest.mark.parametrize("pap", [0, 1], ids=["rap", "paprap"])
@pytest.mark.parametrize("fuse", [2, 3])
def test_rap_rejects_a_shared_scale_set(_gp_gfx1250, gfx1250_iim, assembler, capsys, fuse, pap):
    """RAP silences A by wave parity, which the shared-scale rows do not give it.

    TDMFuse 2/3 with both scales seat A on part of the waves, so Solution rejects
    ReuseAcrossPersistent there rather than letting the parity null reach the writer."""
    extra = dict(PrefetchAcrossPersistent=1) if pap else {}
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse, mi=_MI_W2,
                       ReuseAcrossPersistent=1, AssertSummationElementMultiple=256, **extra)
    assert sol.get("Valid") is not True, "TDMFuse=%d + RAP was accepted" % fuse
    assert "ReuseAcrossPersistent is not implemented with TDMFuse=%d" % fuse in out, out[-2000:]


@pytest.mark.parametrize("fuse", [1, 2, 3])
def test_rap_accepts_the_every_wave_pair(_gp_gfx1250, gfx1250_iim, assembler, capsys, fuse):
    """Without MX scales TDMFuse 1-3 give A and B a set each on every wave, which RAP can silence."""
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse, mi=_MI_W2,
                       ReuseAcrossPersistent=1, AssertSummationElementMultiple=256,
                       ProblemType={"MacDataTypeB": "F8", "MXBlockA": 0, "MXBlockB": 0})
    assert sol.get("Valid") is True, "TDMFuse=%d + RAP without scales was rejected: %r" % (fuse, out)
    assert sol.get("ReuseAcrossPersistent") == 1
