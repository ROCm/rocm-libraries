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
from Tensile.Components.StreamK import StreamKDynamic, StreamKHybrid, StreamKTwoTileDPFirst
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
import copy
import re
from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.Components.TDMFuse import TDM_GROUPS, tdmGrouping, tdmPapRejectReason, tdmScaleSharesDataSet

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

    def tdmApplyStreamKOffsetWaveSeparated(self, kernel, tpa, tpb):
        return self._module("tdmApplyStreamKOffsetWaveSeparated_%s_%s" % (tpa["tensorChar"], tpb["tensorChar"]))

    def releaseGlobalReadIncsSgprsAfterTdmWaveSep(self, kernel):
        return self._module("releaseGlobalReadIncsSgprsAfterTdmWaveSep")

    def isTdmWaveSeparated(self, kernel):
        return kwa_module.KernelWriterAssembly.isTdmWaveSeparated(self, kernel)


    def tdmFuseAMx(self, kernel):
        return kwa_module.KernelWriterAssembly.tdmFuseAMx(self, kernel)

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
    def papHasNextPersistentIteration(self, writer, kernel, skipLabel):
        return _module_with_comment("papHasNextPersistentIteration", "unit: has next persistent iteration")

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
    "HalfPLR": 0,
    "NoTailLoop": False,
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
    assert _pap_solution(StreamK=4, TDMInst=0)["Valid"] is True


def test_solution_validation_rejects_pap_streamk_dynamic_with_tdm(capsys):
    # The TDM + PAP twin gate is deliberately NOT relaxed for SK4: TDM+PAP
    # remains StreamK==3 only.
    assert _pap_solution(StreamK=4, TDMInst=3)["Valid"] is False
    assert "TDM + PrefetchAcrossPersistent requires StreamK == 3" in capsys.readouterr().out


def test_solution_validation_accepts_pap_streamk_hybrid():
    # PAP is allowed for StreamK==5 (StreamKHybrid) in addition to StreamK==3
    # and StreamK==4. The validation gate accepts StreamK in (3, 4, 5). A
    # single PAP-enabled SK5 kernel is correct for BOTH runtime sub-paths
    # (static SK3-like and dynamic SK4-like) via StreamKHybridMode dispatch.
    # TDM is disabled here (TDMInst=0): the TDM+PAP twin gate is intentionally
    # kept SK3-only, so SK5 PAP is supported for the non-TDM path.
    assert _pap_solution(StreamK=5, TDMInst=0)["Valid"] is True


def test_solution_validation_rejects_pap_streamk_hybrid_with_tdm(capsys):
    # The TDM + PAP twin gate is deliberately NOT relaxed for SK5: TDM+PAP
    # remains StreamK==3 only (hybrid TDM+PAP deferred).
    assert _pap_solution(StreamK=5, TDMInst=3)["Valid"] is False
    assert "TDM + PrefetchAcrossPersistent requires StreamK == 3" in capsys.readouterr().out

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
    writer, items = _prefetch_across_persistent(monkeypatch, StreamKForceDPOnly=1, HalfPLR=1)

    _assert_loop_counters_checkpointed_in_vgprs(writer, items)


def test_dp_only_pap_skips_loop_counter_checkpoint(monkeypatch):
    # DP-only StreamK keeps LoopCounter/OrigLoopCounter constant (idempotent
    # recompute, PAP never runs on the last tile), so prefetchAcrossPersistent
    # skips the 2-VGPR checkpoint/restore entirely
    # (KernelWriterAssembly: snapshotLoopCounter = HalfPLR or not StreamKForceDPOnly).
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


def test_streamk3_pap_has_next_persistent_iteration_uses_streamkiter_compare():
    # The PAP "is there a next persistent iteration?" predicate is now a
    # StreamK-component seam. The static StreamK variants (SK3 TwoTileDPFirst,
    # and the SK3/static path of SK5) keep the historical
    # StreamKIter >= StreamKIterEnd compare + skip branch, byte-for-byte, so
    # relaxing the seam for SK4 (StreamKDynamic) does not perturb SK3 codegen.
    from rocisa.code import Label

    skip_label = Label("SK_SkipNllPAP_unit", "")
    module = StreamKTwoTileDPFirst().papHasNextPersistentIteration(
        writer=None, kernel={}, skipLabel=skip_label
    )
    rendered = str(module)
    assert "s_cmp_ge_u32 s[sgprStreamKIter], s[sgprStreamKIterEnd]" in rendered
    assert "No next persistent iteration" in rendered
    assert "s_cbranch_scc1 label_SK_SkipNllPAP_unit" in rendered


class _PapFetchWriter:
    def __init__(self):
        self.labels = _StubLabels()
        self.sgprPool = RegisterPool(
            0, RegisterType.Sgpr, defaultPreventOverflow=False, printRP=False
        )


def _fake_fetch(self, writer, kernel, preventOverflow=True, uniqueLabels=False):
    sidx = writer.sgprPool.checkOut(1, "workItemIdx")
    return _module_with_comment("fakeFetch", "unit: queue pop"), sidx


def test_sk4_pap_has_next_primes_before_drain_check(monkeypatch):
    # SK4 PAP must stash SkNextWorkItem and set SkPrefetchPrimed before the
    # TotalItems drain compare so the back-edge never re-pops a termination token.
    monkeypatch.setattr(StreamKDynamic, "_fetchWorkItemAndBroadcast", _fake_fetch)
    from rocisa.code import Label

    skip_label = Label("SK_SkipNllPAP_sk4", "")
    module = StreamKDynamic().papHasNextPersistentIteration(
        _PapFetchWriter(), {"StreamK": 4}, skip_label
    )
    rendered = str(module)
    primed = rendered.find("s[sgprSkPrefetchPrimed]")
    drain = rendered.find("s[sgprTotalItems]")
    assert primed != -1 and drain != -1 and primed < drain
    assert "s[sgprSkNextWorkItem]" in rendered


def test_sk5_pap_has_next_dispatches_static_and_dynamic(monkeypatch):
    # SK5 PAP is a runtime hybrid: mode==0 keeps the SK3 StreamKIter compare;
    # mode!=0 reuses the SK4 pop-and-prime handoff.
    monkeypatch.setattr(StreamKHybrid, "_fetchWorkItemAndBroadcast", _fake_fetch)
    from rocisa.code import Label

    skip_label = Label("SK_SkipNllPAP_sk5", "")
    module = StreamKHybrid().papHasNextPersistentIteration(
        _PapFetchWriter(), {"StreamK": 5}, skip_label
    )
    rendered = str(module)
    assert "s[sgprStreamKHybridMode]" in rendered
    assert "s[sgprStreamKIter]" in rendered
    assert "s[sgprSkPrefetchPrimed]" in rendered
    assert "s[sgprSkNextWorkItem]" in rendered


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
        {"StreamK": 4},
        {"tensorChar": "A"},
        {"tensorChar": "B"},
    )
    rendered = str(module)
    assert "s[sgprSkNextWorkItem]" in rendered
    assert "unit: tile identity" in rendered


# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""PrefetchAcrossPersistent against the non-default TDM descriptor groupings.

PAP hands a persistent tile over by rebuilding the TDM descriptors and by
restoring the LDS bank the next-tile prefetch landed in. Every helper on that
path walks the tensors as the two pairs (A, B) and (MXSA, MXSB) and applies
exactly one offset per pair, which is only the ownership the default grouping
has: {A,B} on one descriptor set and {MXSA,MXSB} on another. A grouping that
seats a scale tensor on a data tensor's set makes tdmMXSAGroup0/tdmMXSBGroup0
RegSet aliases of tdmAGroup0/tdmBGroup0, and then:

  * `papTdmRestoreLdsBank` and `papTdmUpdateDescriptor` add the bank offset once
    under tdmAGroup0 and once under tdmMXSAGroup0. Aliased, that is one physical
    register twice while its sibling range is never shifted. Both are plain
    `s_add_u32` with no normalize-first step, so unlike `papTdmSetTailLdsBank`
    they are NOT idempotent. Measured on gfx1250 at TDMFuse=1: the emitted
    kernel carries `.set sgprtdmMXSAGroup0, sgprtdmAGroup0+0`, both adds land on
    s25, and B's s29 is never touched. It assembles cleanly, so it is silent.
  * `tdmApplyStreamKTailOffsetWaveSeparated` names tdmMXSAMXSBIncs, whose
    defineSgpr is guarded by `not tdmFuseAMx`. Measured at TDMFuse=2: the symbol
    is referenced with no `.set`, and the assembler refuses the kernel with
    "expected absolute expression".

The rejection is derived from group membership rather than from a TDMFuse value,
so a grouping row nobody has wired to an integer yet inherits it. What follows
pins the firing, the accept controls that stop it over-rejecting, the precedence
against TDMFuse's own guards, and the assembly-level invariant behind it.
"""


# Sibling unit tests mutate the process-global defaultSolution in place; snapshot
# it at import (collection precedes execution) so Solution construction here is
# not order-dependent.
_PRISTINE_TDMFUSE_SOLUTION = copy.deepcopy(dict(defaultSolution))


# MIWaveGroup [2,2] -> NumWaves 4, which is what TDMFuse=2 names explicitly and
# what TDMFuse=1 needs at least two waves for.
_MI_W4 = [16, 16, 128, 1, 1, 2, 4, 2, 2]


# The reject's own words, enough of them to tell it from its neighbours.
_PAP_GROUPING_MSG = "requires every TDM scale tensor to own its own descriptor set"


# TDMFuse's own drift guards, which must keep precedence over the one above.
_FUSE_DECLINED_MSG = "passed its solution-level guards but"


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


@pytest.mark.parametrize("fuse", [1, 2])
def test_reason_fires_for_every_sharing_grouping(fuse):
    reason = tdmPapRejectReason(_ks(fuse=fuse))
    assert reason is not None
    assert _PAP_GROUPING_MSG in reason
    # It must explain the aliasing, not merely name the knob: a reader who has
    # only this string has to be able to find the two register ranges.
    assert "tdmMXSAGroup0/tdmMXSBGroup0" in reason
    assert "tdmAGroup0/tdmBGroup0" in reason
    assert tdmGrouping(_ks(fuse=fuse)).name in reason


def test_default_grouping_is_not_rejected():
    assert tdmPapRejectReason(_ks(fuse=0)) is None


@pytest.mark.parametrize("decline", [{"NumWaves": 1}, {"TDMSplit": True},
                                     {"UseSubtileImpl": True}, {"TDMInst": 1}])
def test_declined_grouping_answers_with_the_fallback(decline):
    """A grouping TDMFuse asked for but the predicates declined shares nothing.

    The writer falls back to {A,B} + {MXSA,MXSB} in that case, so rejecting on
    the requested value rather than the resolved grouping would refuse a
    solution whose descriptors are in fact separate. TDMFuse's own guards refuse
    these for their own reasons; this only pins that the reason here is silent.
    """
    assert tdmPapRejectReason(_ks(fuse=1, **decline)) is None
    assert tdmPapRejectReason(_ks(fuse=2, **decline)) is None


@pytest.mark.parametrize("missing", [{"MXBlockA": 0, "MXBlockB": 32},
                                     {"MXBlockA": 32, "MXBlockB": 0},
                                     {"MXBlockA": 0, "MXBlockB": 0}])
def test_scale_less_types_share_nothing(missing):
    """No live MXSA/MXSB means no scale to seat on a data tensor's set."""
    assert tdmPapRejectReason(_ks(fuse=1, ProblemType=missing)) is None


@pytest.mark.parametrize("row, shares", [("MX_AB", False), ("AB", False), ("None", False),
                                         ("paired", True), ("A_MX", True), ("B_MX", True)])
def test_every_grouping_row_answers_without_a_new_branch(row, shares):
    """Rows no TDMFuse integer selects inherit the right answer as data.

    B_MX in particular is A_MX mirrored and gets refused for free, which is the
    property that keeps this reject from needing an edit per new row.
    """
    assert bool(tdmScaleSharesDataSet(_ks(), TDM_GROUPS[row])) is shares


# ---------------------------------------------------------------------------
# Solution level: real gfx1250 caps + assembler, assignDerivedParameters
# end-to-end. Mirrors test_halfplr_streamk_rejects.py's harness.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gfx1250_iim():
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Toolchain.Validators import validateToolchain

    cxx = validateToolchain("amdclang++")
    isa = gfxToIsa("gfx1250")
    iim = makeIsaInfoMap([isa], cxx)
    if not iim[isa].asmCaps["SupportedISA"]:
        pytest.skip("amdclang++ in this environment does not support gfx1250")
    return iim


@pytest.fixture(scope="module")
def assembler():
    from Tensile.Toolchain.Assembly import makeAssemblyToolchain
    from Tensile.Toolchain.Validators import validateToolchain, ToolchainDefaults

    cxx = validateToolchain("amdclang++")
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    return makeAssemblyToolchain(cxx, bundler, "default").assembler


@pytest.fixture(scope="module")
def _gp_gfx1250(gfx1250_iim):
    from Tensile.Common.GlobalParameters import globalParameters, assignGlobalParameters
    from Tensile.Common.ValidParameters import validParameters

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
    """A TN MXF8F4 StreamK=3 solution: the smallest shape PAP+TDM accepts.

    StreamKForceDPOnly=1 because PAP+TDM only emits there -- at
    StreamKForceDPOnly=0 the writer fails for every grouping including the
    default, which is a separate pre-existing limitation and not what these
    tests are about.
    """
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.SolutionStructs.Validators.MatrixInstruction import (
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
        "TDMFuse": 0, "TDMSplit": False, "TDMCross": 0, "InitCIterWmma": 0,
    }
    params.update(overrides)
    params.update(matrixInstructionToMIParameters(
        mi, isa, params["WavefrontSize"], problem_type, params["WorkGroup"], gfx1250_iim))
    return params


def _derive(gfx1250_iim, assembler, capsys, **overrides):
    from Tensile.SolutionStructs.Solution import Solution
    sol = Solution(_make_params(gfx1250_iim, **overrides), False, True, False,
                   assembler, gfx1250_iim)
    return sol, capsys.readouterr().out


def test_pap_on_the_default_grouping_is_accepted(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    """The accept control. Without it every reject below is vacuous."""
    sol, out = _derive(gfx1250_iim, assembler, capsys, PrefetchAcrossPersistent=1)
    assert sol.get("Valid") is True, "expected accept, rejected with: %r" % out
    assert sol.get("PrefetchAcrossPersistent") == 1, \
        "PAP was silently reset, so this control proves nothing"
    assert sol.get("LdsOffsetA_Blk") != 0, \
        "LdsOffsetA_Blk == 0 folds the LDS-bank helpers out entirely"


@pytest.mark.parametrize("fuse", [1, 2])
def test_pap_with_a_shared_scale_set_is_rejected(_gp_gfx1250, gfx1250_iim, assembler, capsys, fuse):
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       PrefetchAcrossPersistent=1, TDMFuse=fuse)
    assert sol.get("Valid") is False, "TDMFuse=%d + PAP was accepted" % fuse
    assert _PAP_GROUPING_MSG in out, "rejected for another reason: %r" % out
    # Diagnosis, not just validity: if an earlier guard starts shadowing this
    # one the message goes and only this assertion notices.
    assert _FUSE_DECLINED_MSG not in out


@pytest.mark.parametrize("fuse", [1, 2])
def test_the_same_grouping_without_pap_is_still_accepted(_gp_gfx1250, gfx1250_iim, assembler, capsys, fuse):
    """The rejection is about PAP, not about the grouping."""
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       PrefetchAcrossPersistent=0, TDMFuse=fuse)
    assert sol.get("Valid") is True, "expected accept, rejected with: %r" % out
    assert _PAP_GROUPING_MSG not in out


def test_tdmfuse_own_guards_keep_precedence(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    """A grouping that cannot be built reports its own reason, not this one.

    NumWaves=8 satisfies PAP's wave-separated gate but not TDMFuse=2's explicit
    four waves, so tdmFuseAMx declines, the writer falls back to the default
    grouping, and nothing is aliased for PAP to complain about.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys, PrefetchAcrossPersistent=1,
                       TDMFuse=2, mi=[16, 16, 128, 1, 1, 2, 4, 2, 4])
    assert sol.get("Valid") is False
    assert "TDMFuse=2 dispatches its shared descriptor three ways" in out
    assert _PAP_GROUPING_MSG not in out


@pytest.mark.parametrize("fuse", [0, 1, 2])
def test_no_accepted_pap_solution_aliases_a_scale_onto_a_data_set(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse):
    """The invariant every PAP TDM helper is written against.

    Stated over whatever survives validation rather than over a list of values,
    so it keeps its teeth if the reject is ever re-expressed. Against the
    unfixed tree TDMFuse=1 and 2 are accepted and this fails on both.
    """
    sol, _ = _derive(gfx1250_iim, assembler, capsys,
                     PrefetchAcrossPersistent=1, TDMFuse=fuse)
    if not sol.get("Valid"):
        pytest.skip("TDMFuse=%d + PAP is refused, nothing to check" % fuse)
    assert tdmScaleSharesDataSet(sol) == (), (
        "accepted a PAP solution whose %s grouping shares a descriptor set with "
        "a scale tensor" % tdmGrouping(sol).name)


# Both of these are plain `s_add_u32 dst, dst, x`. papTdmSetTailLdsBank is
# deliberately excluded: it normalizes to bank 0 before adding, so it IS
# idempotent and is applied more than once per register by design.
_NON_IDEMPOTENT_BANK_SITES = {
    "papTdmRestoreLdsBank": ("shift A/B descriptor to PAP bank",
                             "shift MX descriptor to PAP bank"),
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
    from Tensile.Common.Types import DebugConfig
    from Tensile.KernelWriterAssembly import KernelWriterAssembly
    from Tensile.SolutionStructs.Naming import getKernelFileBase
    from Tensile.SolutionStructs.Solution import Solution
    from Tensile.TensileCreateLibrary.Run import (generateKernelObjectsFromSolutions,
                                                  processKernelSource)
    from Tensile.Tests.rocisa_test_state import preserve_rocisa_kernel_state

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


@pytest.mark.parametrize("fuse", [0, 1, 2])
def test_pap_shifts_each_descriptor_lds_bank_exactly_once(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse):
    """Every non-idempotent bank shift lands on its own physical register.

    Against the unfixed tree TDMFuse=1 and 2 are accepted, tdmMXSAGroup0 is a
    RegSet alias of tdmAGroup0, and both sites shift one register twice while
    the sibling range is never shifted -- which is what this fails on.
    """
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

    # And the complement: no allocated descriptor set is left behind. Aliasing
    # does not only double one shift, it drops the sibling's shift entirely.
    if "papTdmRestoreLdsBank" in shifted:
        allocated = {phys(s) for s in _LDS_ADDR_SYMBOLS if phys(s) is not None}
        assert {reg for _, reg in shifted["papTdmRestoreLdsBank"]} == allocated, (
            "papTdmRestoreLdsBank shifted %s but the kernel allocates %s" %
            (sorted({reg for _, reg in shifted["papTdmRestoreLdsBank"]}), sorted(allocated)))


@pytest.mark.parametrize("fuse", [0, 1, 2])
def test_pap_never_names_an_unallocated_tdm_increment(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse):
    """Every tdm*Incs the kernel reads has a defineSgpr behind it.

    tdmMXSAMXSBIncs is allocated only when `not tdmFuseAMx`, while the PAP tail
    reset builds the name unconditionally. Against the unfixed tree TDMFuse=2 +
    PAP emits `s[sgprtdmMXSAMXSBIncs]` with no .set, and the assembler refuses
    it with "expected absolute expression".
    """
    sol, asm = _emit_asm(gfx1250_iim, assembler, PrefetchAcrossPersistent=1, TDMFuse=fuse)
    capsys.readouterr()
    if asm is None:
        pytest.skip("TDMFuse=%d + PAP is refused, nothing to emit" % fuse)
    resolved, _ = _resolve_sets(asm)
    referenced = set(re.findall(r"\bsgprtdm\w*Incs\b", asm))
    undefined = sorted(s for s in referenced if resolved.get(s) is None)
    assert not undefined, "referenced with no .set: %s" % undefined
