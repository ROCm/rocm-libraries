################################################################################
#
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

from .ExecutionPolicy import isPersistent, isStreamK

from typing import Dict

import copy

from .Activation import ActivationType
from . import Hardware
from . import Properties
from .Common import printExit, state, state_key_ordering, IsaInfo

# state() methods below shadow this name. Call the helper through the alias.
_dump_state = state
from .Common.Architectures import gfxToIsa
from .Common.DataType import DataType
from .SolutionStructs import Solution as OriginalSolution
from .SolutionStructs.Problem import getBiasDataTypeListDefault, getGateResidualDataTypeListDefault
from .Toolchain.Component import Assembler
from math import ceil

MIN_K_FOR_GSU = 32

def _state_from_document(obj):
    """Serialize a record holder. Assigned attributes overlay the C++ document."""
    record = getattr(obj, "_record", None)
    if record is None:
        return {key: _dump_state(getattr(obj, key)) for key in obj.StateKeys}
    rv = {}
    for key, value in record.items():
        rv[key] = _dump_state(getattr(obj, key, value))
    # Fields assigned after the record was built, when the document omitted them.
    for key in obj.StateKeys:
        if key not in rv and hasattr(obj, key):
            rv[key] = _dump_state(getattr(obj, key))
    return rv


def _fill_document(obj, document):
    obj._record = document
    for key, value in document.items():
        setattr(obj, key, value)
    return obj


def _restore_predicate_value(value):
    if isinstance(value, dict) and "type" in value:
        return _predicate_from_document(Properties.Predicate, value)
    if isinstance(value, list):
        return [_restore_predicate_value(item) for item in value]
    return value


def _predicate_from_document(cls, document):
    value = _restore_predicate_value(document["value"]) if "value" in document else None
    index = document["index"] if "index" in document else None
    predicate = cls(document["type"], index, value)
    predicate._record = document
    return predicate


def _pack_default(obj):
    # SolutionStructs.ProblemType stores its fields in a dict named state.
    # Data types and other values expose that dict through a state method.
    state_attr = getattr(obj, "state", None)
    if isinstance(state_attr, dict):
        return state_attr
    if callable(state_attr):
        return state_attr()
    raise TypeError("Cannot pack {} into a solution state".format(type(obj).__name__))


def _solution_record(d):
    """Build one solution's library record with tensilelite_write.solution_from_state."""
    try:
        from tensilelite import tensilelite_write
    except (ImportError, OSError) as exc:
        printExit(
            "tensilelite.tensilelite_write is required to build a solution record ({})".format(exc)
        )
    import msgpack
    raw = msgpack.packb(d, default=_pack_default, use_bin_type=True)
    packed = tensilelite_write.solution_from_state(raw)
    return msgpack.unpackb(packed, raw=False, strict_map_key=False)


@state_key_ordering
class FreeIndex:
    StateKeys = ['isA', 'i', 'c', 'd']

    def __init__(self, isA, i=None, c=None, d=None):
        self.isA = isA
        self.i = i # index into A or B (depending on isA)
        self.c = c
        self.d = d

@state_key_ordering
class BatchIndex:
    StateKeys = ['a', 'b', 'c', 'd']
    def __init__(self, a=None, b=None, c=None, d=None):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

@state_key_ordering
class BoundIndex:
    StateKeys = ['a', 'b', 'aMirror', 'bMirror']
    def __init__(self, a=None, b=None, aMirror=False, bMirror=False):
        self.a = a
        self.b = b
        self.aMirror = aMirror
        self.bMirror = bMirror


class ProblemType:
    StateKeys = ['operationIdentifier', 'transA', 'transB', 'computeInputTypeA', 'computeInputTypeB', 'aType', 'bType', 'cType', 'dType', 'eType', 'computeType',
                 'useBeta', 'useBias', 'biasSrcWhiteList', 'useE', 'useGateResidual', 'gateResidualDataTypeWhiteList', 'useScaleAB', 'useScaleCD', 'useScaleAlphaVec', 'biasDataTypeWhiteList',
                 'highPrecisionAccumulate', 'useInitialStridesAB', 'useInitialStridesCD', 'stridedBatched', 'groupedGemm',
                 'useGradient', 'activationType', 'activationArgLength', 'activationComputeDataType', 'activationNoGuard',
                 'sparse', 'f32XdlMathOp', 'supportDeviceUserArguments', 'outputAmaxD', 'swizzleTensorA', 'swizzleTensorB', 'metadataLayout',
                 'mxBlockA', 'mxBlockB', 'mxTypeA', 'mxTypeB', 'mxScaleFormat', 'fusedGemmA2A']
    @classmethod
    def FromOriginalState(cls, d):
        indices = [None]*d['TotalIndices']
        freeIndices  = []
        batchIndices = []
        boundIndices = []

        for i in d['IndicesSummation']:
            bi = BoundIndex(aMirror=('MirrorDimsA' in d and i in d['MirrorDimsA']),
                            bMirror=('MirrorDimsB' in d and i in d['MirrorDimsB']))
            indices[i] = bi
            boundIndices.append(bi)

        for i in range(0,d["NumIndicesC"]):
            if i in d['IndicesBatch']:
                bi = BatchIndex(c=i, d=i)
                indices[i] = bi
                batchIndices.append(bi)
            else:
                assert i in d['IndicesFree']
                if i in d['IndexAssignmentsA']:
                    fi = FreeIndex(isA=True, i=d["IndexAssignmentsA"].index(i), c=i, d=i)
                elif i in d['IndexAssignmentsB']:
                    fi = FreeIndex(isA=False, i=d["IndexAssignmentsB"].index(i), c=i, d=i)
                else:
                    raise RuntimeError("free index %u not in ia or ib"%i)
                indices[i] = fi
                freeIndices.append(fi)

        for ia, ic in enumerate(d['IndexAssignmentsA']):
            indices[ic].a = ia

        for ib, ic in enumerate(d['IndexAssignmentsB']):
            indices[ic].b = ib

        for idx in indices:
            assert idx is not None
            idxState = state(idx)
            for (key, value) in list(idxState.items()):
                assert value is not None

        rv = cls()
        rv.indices = indices
        rv.freeIndices = freeIndices
        rv.batchIndices = batchIndices
        rv.boundIndices = boundIndices
        rv.aDims = len(d['IndexAssignmentsA'])
        rv.bDims = len(d['IndexAssignmentsB'])
        rv.cDims = d['NumIndicesC']
        rv.dDims = rv.cDims

        rv.aConjugate = d['ComplexConjugateA']
        rv.bConjugate = d['ComplexConjugateB']

        srcType = DataType(d['DataType'])
        dstType = DataType(d['DestDataType']) if 'DestDataType' in d else srcType
        computeType = DataType(d['ComputeDataType']) if 'ComputeDataType' in d else dstType

        rv.transA = bool(d['TransposeA'])
        rv.transB = bool(d['TransposeB'])

        if 'MacDataTypeA' in d: #it will either be set as d['MacDataType'] or a specified input
            rv.computeInputTypeA = DataType(d['MacDataTypeA'])
        else:
            rv.computeInputTypeA = srcType
        if 'MacDataTypeB' in d:
            rv.computeInputTypeB = DataType(d['MacDataTypeB'])
        else:
            rv.computeInputTypeB = srcType

        if 'DataTypeA' in d: #it will either be set as d['DataType'] or a specified input
            rv.aType = DataType(d['DataTypeA'])
        else:
            rv.aType = rv.computeInputTypeA
        if 'DataTypeB' in d:
            rv.bType = DataType(d['DataTypeB'])
        else:
            rv.bType = rv.computeInputTypeB

        if rv.aType.isFloat8BFloat8() or rv.bType.isFloat8BFloat8():
            rv.aType = DataType("F8")
            rv.bType = DataType("B8")
        elif rv.aType.isFloat8BFloat8_fnuz() or rv.bType.isFloat8BFloat8_fnuz():
            rv.aType = DataType("F8N")
            rv.bType = DataType("B8N")
        elif rv.aType.isBFloat8Float8() or rv.bType.isBFloat8Float8():
            rv.aType = DataType("B8")
            rv.bType = DataType("F8")
        elif rv.aType.isBFloat8Float8_fnuz() or rv.bType.isBFloat8Float8_fnuz():
            rv.aType = DataType("B8N")
            rv.bType = DataType("F8N")

        if 'DataTypeE' in d:
            rv.eType = DataType(d['DataTypeE'])
        else:
            rv.eType = dstType

        if 'DataTypeAmaxD' in d:
            rv.amaxDType = DataType(d['DataTypeAmaxD'])
        else:
            rv.amaxDType = computeType

        rv.cType = dstType
        rv.dType = dstType
        # we already checked the src/dst/compute types are supported and well-assigned in SolutionStruct
        rv.alphaType = computeType
        rv.betaType  = computeType
        rv.computeType = computeType

        rv.highPrecisionAccumulate = False
        if 'HighPrecisionAccumulate' in d:
            rv.highPrecisionAccumulate = d['HighPrecisionAccumulate']

        rv.useInitialStridesAB = False
        if 'UseInitialStridesAB' in d:
            rv.useInitialStridesAB = d['UseInitialStridesAB']
        rv.useInitialStridesCD = False
        if 'UseInitialStridesCD' in d:
            rv.useInitialStridesCD = d['UseInitialStridesCD']

        rv.stridedBatched = True
        if 'StridedBatched' in d:
          rv.stridedBatched = d['StridedBatched']

        rv.groupedGemm = False
        if 'GroupedGemm' in d:
          rv.groupedGemm = d['GroupedGemm']

        rv.setConstStrideA = []
        if 'SetConstStrideA' in d:
            rv.setConstStrideA = d['SetConstStrideA']
        rv.setConstStrideB = []
        if 'SetConstStrideB' in d:
            rv.setConstStrideB = d['SetConstStrideB']

        rv.mirrorDimsA = d['MirrorDimsA'] if 'MirrorDimsA' in d else []
        rv.mirrorDimsB = d['MirrorDimsB'] if 'MirrorDimsB' in d else []

        rv.useBeta = True
        if 'UseBeta' in d:
            rv.useBeta = d['UseBeta']

        rv.useBias               = 0
        rv.biasDataTypeWhiteList = []
        rv.biasSrcWhiteList = []
        rv.setConstStrideBias = []

        if 'UseBias' in d:
            rv.useBias = d['UseBias']
            if 'BiasDataTypeList' in d:
                d["BiasDataTypeList"].sort()  # Sort to make sure names are unique
                rv.biasDataTypeWhiteList = d['BiasDataTypeList']
            else:
                rv.biasDataTypeWhiteList = getBiasDataTypeListDefault(d)
            if 'BiasSrc' in d:
                m = ["A", "B", "C", "D"]
                rv.biasSrcWhiteList = [m.index(d['BiasSrc'])]
            if 'SetConstStrideBias' in d:
                rv.setConstStrideBias = d['SetConstStrideBias']

        rv.useE = False
        if 'UseE' in d:
            rv.useE = d['UseE']

        rv.useGateResidual = False
        rv.gateResidualDataTypeWhiteList = []
        rv.setConstStrideGate = []
        if 'UseGateResidual' in d:
            rv.useGateResidual = bool(d['UseGateResidual'])
            if 'GateResidualDataTypeList' in d:
                d["GateResidualDataTypeList"].sort()  # Sort to make sure names are unique
                rv.gateResidualDataTypeWhiteList = d['GateResidualDataTypeList']
            else:
                rv.gateResidualDataTypeWhiteList = getGateResidualDataTypeListDefault(d)
            if 'SetConstStrideGate' in d:
                rv.setConstStrideGate = d['SetConstStrideGate']

        rv.useGradient = False
        if 'Gradient' in d:
            rv.useGradient = d["Gradient"]

        rv.outputAmaxD = False
        if 'OutputAmaxD' in d:
            rv.outputAmaxD = d['OutputAmaxD']

        rv.useScaleAB = ""
        if 'UseScaleAB' in d:
            rv.useScaleAB = d['UseScaleAB']
        rv.useScaleCD = False
        if 'UseScaleCD' in d:
            rv.useScaleCD = d['UseScaleCD']

        rv.useScaleAlphaVec = 0
        if 'UseScaleAlphaVec' in d:
            rv.useScaleAlphaVec = d['UseScaleAlphaVec']

        rv.batched = d['Batched']

        rv.activationType      = ActivationType('none')
        rv.activationArgLength = 0
        if 'ActivationType' in d:
            rv.activationType = ActivationType(d['ActivationType'])
            rv.activationArgLength = len(rv.activationType.getAdditionalArgStringList())
        rv.activationComputeDataType = DataType(d['ActivationComputeDataType'])
        rv.activationNoGuard = False
        if 'ActivationNoGuard' in d:
            rv.activationNoGuard = d["ActivationNoGuard"]
        rv.sparse = 0
        if 'Sparse' in d:
            rv.sparse = int(d['Sparse'])

        rv.f32XdlMathOp = DataType(d['F32XdlMathOp']) if 'F32XdlMathOp' in d else DataType(0)

        rv.supportDeviceUserArguments = False
        if 'SupportUserArgs' in d:
            rv.supportDeviceUserArguments = d['SupportUserArgs']

        rv.swizzleTensorA = d.get('SwizzleTensorA', False)
        rv.swizzleTensorB = d.get('SwizzleTensorB', False)

        rv.fusedGemmA2A = d.get('FusedGemmA2A', False)

        rv.mxBlockA = d.get('MXBlockA', 0)
        rv.mxBlockB = d.get('MXBlockB', 0)
        rv.mxTypeA = DataType(d['DataTypeMXSA']) if 'DataTypeMXSA' in d else DataType(0)
        rv.mxTypeB = DataType(d['DataTypeMXSB']) if 'DataTypeMXSB' in d else DataType(0)
        # mxScaleFormat is a Solution-level parameter and is populated by
        # Solution.FromOriginalState, which has access to the parent dict.

        rv.metadataLayout = 0
        if 'MetadataLayout' in d:
            rv.metadataLayout = d['MetadataLayout']
        return rv

    def __init__(self, freeIndices=None, batchIndices=None, boundIndices=None, aDims=None, bDims=None, cDims=None, dDims=None):
        self.freeIndices  = freeIndices
        self.batchIndices = batchIndices
        self.boundIndices = boundIndices
        self.aDims = aDims
        self.bDims = bDims
        self.cDims = cDims
        self.dDims = dDims

    @property
    def indexNames(self):
        aNames = ['_'] * self.aDims
        bNames = ['_'] * self.bDims
        cNames = ['_'] * self.cDims

        allIndexNames = 'ijklmnopqrstuvwxyz'
        index = 0

        dNames = list([allIndexNames[index+i] for i in range(self.cDims)])
        index += len(dNames)

        sumNames = list([allIndexNames[index+i] for i in range(len(self.boundIndices))])
        index += len(sumNames)

        for free in self.freeIndices:
            if free.isA:
                aNames[free.i ] = dNames[free.d]
            else:
                bNames[free.i ] = dNames[free.d]
            cNames[free.c] = dNames[free.d]

        for batch in self.batchIndices:
            name = dNames[batch.d]
            aNames[batch.a] = name
            bNames[batch.b] = name
            cNames[batch.c] = name

        for i, bound in enumerate(self.boundIndices):
            name = sumNames[i]
            aNames[bound.a] = name.upper() if bound.aMirror else name
            bNames[bound.b] = name.upper() if bound.bMirror else name

        aNames = ''.join(aNames)
        bNames = ''.join(bNames)
        cNames = ''.join(cNames)
        dNames = ''.join(dNames)
        sumNames = ''.join(sumNames)

        return (aNames, bNames, cNames, dNames, sumNames)

    @property
    def operationIdentifier(self):
        (aNames, bNames, cNames, dNames, sumNames) = self.indexNames

        aOp = 'C' if self.aConjugate else ''
        bOp = 'C' if self.bConjugate else ''

        return '_'.join(['Contraction', sumNames,
                         'A' + aNames + aOp,
                         'B' + bNames + bOp,
                         'C' + cNames,
                         'D' + dNames])

    def placeholderStr(self, includeBatch=False, includeOperation=False, includeType=False):
        ret = ""
        if includeOperation:
            ret = self.operationIdentifier
            if not self.useBeta:
                ret += "_Beta0"
            ret += "_StridedBatched{}".format(int(self.stridedBatched))
        if includeType:
            ret += "_Type_{}{}".format(DataType(self.aType).toChar(), DataType(self.cType).toChar())
            if self.highPrecisionAccumulate:
                ret += "_HPA"

        return ret

    def state(self):
        record = getattr(self, "_record", None)
        if record is not None:
            return copy.deepcopy(record)
        return {key: _dump_state(getattr(self, key)) for key in self.StateKeys}

    def predicates(self, includeBatch=False, includeOperation=False, includeType=False):
        predicates = []

        #if includeBatch and not self.batched:
        #    predicates.append(ProblemPredicate("BatchSizeEqual", index=0, value=1))

        if includeOperation:
            predicates.append(ProblemPredicate("OperationIdentifierEqual", value=self.operationIdentifier))
            if not self.useBeta:
                predicates.append(ProblemPredicate("BetaZero"))
            predicates.append(ProblemPredicate("BiasDataTypeWhiteList", value=self.biasDataTypeWhiteList))
            predicates.append(ProblemPredicate("GateResidualDataTypeWhiteList", value=self.gateResidualDataTypeWhiteList))
            predicates.append(ProblemPredicate("BiasSrcWhiteList", value=self.biasSrcWhiteList))
            predicates.append(ProblemPredicate("AmaxDCheck", value=self.outputAmaxD))
            if self.activationType in ['all', 'hipblaslt_all']:
                exportType = ActivationType.Export.GRADONLY if self.useGradient else ActivationType.Export.NORMAL
                supportedBy = ActivationType.SupportedBy.ALL if self.activationType == 'all' else ActivationType.SupportedBy.HIPBLASLT
                enumList = [actEnum.capitalize() for actEnum in ActivationType.getEnumStrList(self.activationComputeDataType, supportedBy, exportType=exportType)]
                predicates.append(ProblemPredicate("ActivationEnumWhiteList", value=enumList))
            # predicates.append(ProblemPredicate("UseScaleAlphaVec", value=self.useScaleAlphaVec))
            # predicates.append(ProblemPredicate("GroupedGemm", value=self.groupedGemm))

        if includeType:
            predicates.append(ProblemPredicate("TypesEqual", value=(self.aType, self.bType, self.cType, self.dType, self.computeInputTypeA, self.computeInputTypeB)))
            predicates.append(ProblemPredicate("HighPrecisionAccumulate", value=self.highPrecisionAccumulate))
            predicates.append(ProblemPredicate("Activation", value=self.activationType))
            predicates.append(ProblemPredicate("ActivationComputeType", value=self.activationComputeDataType))
            predicates.append(ProblemPredicate("ActivationNoGuard", value=self.activationNoGuard))
            predicates.append(ProblemPredicate("UseGradient", value=self.useGradient))
            predicates.append(ProblemPredicate("UseBias", value=self.useBias))
            predicates.append(ProblemPredicate("UseE", value=self.useE))
            predicates.append(ProblemPredicate("UseGateResidual", value=self.useGateResidual))
            predicates.append(ProblemPredicate("DataTypeE", value=self.eType))
            predicates.append(ProblemPredicate("StridedBatched", value=self.stridedBatched))
            predicates.append(ProblemPredicate("GroupedGemm", value=self.groupedGemm))
            predicates.append(ProblemPredicate("UseScaleAB", value=self.useScaleAB))
            predicates.append(ProblemPredicate("UseScaleCD", value=self.useScaleCD))
            predicates.append(ProblemPredicate("UseScaleAlphaVec", value=self.useScaleAlphaVec))
            predicates.append(ProblemPredicate("Sparse", value=self.sparse))
            predicates.append(ProblemPredicate("F32XdlMathOp", value=self.f32XdlMathOp))
            predicates.append(ProblemPredicate("SupportDeviceUserArguments", value=self.supportDeviceUserArguments))
            predicates.append(ProblemPredicate("SwizzleTensorA", value=self.swizzleTensorA))
            predicates.append(ProblemPredicate("SwizzleTensorB", value=self.swizzleTensorB))
            predicates.append(ProblemPredicate("FusedGemmA2A", value=self.fusedGemmA2A))
            if self.fusedGemmA2A and self.batched:
                predicates.append(ProblemPredicate("BatchSizeEqual", index=0, value=1))
            predicates.append(ProblemPredicate("MXBlockA", value=self.mxBlockA))
            if self.mxBlockA:
                predicates.append(ProblemPredicate("DataTypeMXSA", value=self.mxTypeA))
            predicates.append(ProblemPredicate("MXBlockB", value=self.mxBlockB))
            if self.mxBlockB:
                predicates.append(ProblemPredicate("DataTypeMXSB", value=self.mxTypeB))
        return predicates

# Dict-valued Assert* keys and the runtime predicate each one emits per index.
ASSERT_DIM_MAP_PREDICATES = {
    "AssertSizeEqual": "SizeEqual",
    "AssertSizeGreaterThan": "SizeGreaterThan",
    "AssertSizeLessThan": "SizeLessThan",
    "AssertStrideAEqual": "StrideAEqual",
    "AssertStrideBEqual": "StrideBEqual",
    "AssertStrideCEqual": "StrideCEqual",
    "AssertStrideDEqual": "StrideDEqual",
}

def extractDimPredicate(cls, value, predicateName):
    """
    Extract predicates for dict-valued Assert* maps (AssertSizeEqual, ...).
    Value is a dictionary of {index: size}. A value of -1 skips that index.
    """
    predicates = []
    for pos, val in value.items():
        if val != -1:
            predicates.append(cls(predicateName, index=pos, value=val))
    if len(predicates) == 1:
        return predicates[0]
    elif len(predicates) > 1:
        return cls.And(predicates)

class TaskPredicate(Properties.Predicate):
    def state(self):
        record = getattr(self, "_record", None)
        if record is not None:
            return copy.deepcopy(record)
        return super().state()

class ProblemPredicate(Properties.Predicate):
    @classmethod
    def FromOriginalKeyPair(cls, pair):
        (key, value) = pair
        # Arithmetic intensity assertions
        if key == "AssertAIGreaterThanEqual":
            return cls("AIGreaterThanEqual", value=value) if value > 0 else None
        if key == "AssertAILessThanEqual":
            return cls("AILessThanEqual", value=value) if value > 0 else None
        if key in ASSERT_DIM_MAP_PREDICATES:
            if not isinstance(value, dict):
                raise RuntimeError(
                    "{} must be a dict of {{index: value}}, got {!r}".format(key, value)
                )
            return extractDimPredicate(cls, value, ASSERT_DIM_MAP_PREDICATES[key])

        if key.endswith('Multiple'):
            if value == 1:
                return None

            if key == "AssertFree0ElementMultiple":
                tag = "Free0SizeMultiple"
                index = 0
            elif key == "AssertFree1ElementMultiple":
                tag = "Free1SizeMultiple"
                index = 0
            elif key == "AssertSummationElementMultiple":
                tag = "BoundSizeMultiple"
                index = -1
            else:
                raise RuntimeError("Unknown Multiple Value: {}".format(key))

            return cls(tag, index=index, value=value)

        if key.startswith('Assert'):
            raise RuntimeError("Unknown assertion key: {}".format(key))

    @classmethod
    def CompoundPredicates(cls, state, problemType):
        rv = []

        if "BatchSizeEqual" in state:
            rv += [cls('BatchSizeEqual', index=0, value=state["BatchSizeEqual"])]

        if "SynchronizerSizeCheck" in state:
            if state["SynchronizerSizeCheck"]:
                valuepredicates = []
                valuepredicates.append(state["MacroTile0"])
                valuepredicates.append(state["MacroTile1"])
                if state["EnableMatrixInstruction"]:
                    valuepredicates.append(state["MIWaveTile"][0]*state["MIWaveTile"][1])
                else:
                    valuepredicates.append(state["ThreadTile0"]*state["ThreadTile1"])
                if state["NumElementsPerBatchStore"] != 0:
                    valuepredicates.append(int((state["NumElementsPerThread"])/state["NumElementsPerBatchStore"]))
                else:
                    valuepredicates.append(state["NumElementsPerThread"])
                valuepredicates.append(ceil(state["NumThreads"] / state["WavefrontSize"]))
                valuepredicates.append(state["GlobalSplitU"])

                rv += [cls('SynchronizerSizeCheck', index=0, value=valuepredicates)]

        if state["InternalSupportParams"]["KernArgsVersion"] >= 1 and \
                 not isPersistent(state):
            valuepredicates = []
            valuepredicates.append(state["MacroTile0"])
            valuepredicates.append(state["MacroTile1"])
            valuepredicates.append(state["GlobalSplitU"])
            rv += [cls('WorkgroupNumberCheck', index=0, value=valuepredicates)]

        if not problemType.aType.isInt8x4():
            # calculate the minimum supported free dimension size
            TLUA = state['ProblemType']['TLUA']
            TLUB = state['ProblemType']['TLUB']
            minFree0 = state['GlobalReadVectorWidthA'] if TLUA else 1
            minFree1 = state['GlobalReadVectorWidthB'] if TLUB else 1
            rv += [cls('LeadingFree0SizesGreaterOrEqual', value=minFree0)]
            rv += [cls('LeadingFree1SizesGreaterOrEqual', value=minFree1)]

        if len(state["PackedC0IndicesX"]) > 1:
          rv += [cls("CDStridesEqual")]

        if "KernelLanguage" in state:
            rv += [cls("KernelLanguageCompatible", value=state["KernelLanguage"])]

        if ('GlobalSplitU' in state) and (state['GlobalSplitU'] > 1 or state['GlobalSplitU'] == -1):
            if ('_GlobalAccumulation' not in state) or (state['_GlobalAccumulation'] != 'MultipleBuffer'):
                rv += [cls("DeterministicMode", value = False)]

        if isStreamK(state) and ('StreamKAtomic' in state) and (state['StreamKAtomic'] == 1):
            # StreamKAtomic = 1 uses atomic for partial tiles
            rv += [cls("DeterministicMode", value = False)]

        # if bufferload is performed, we output some predication info for host side,
        # to prevent from some extremely large problems from launching and causing bufferload offset limit < 2^32
        # thoses cases will not satisfy the assertion thus won't use the kernel.
        # See Common.py for more details, we will need four values:
        # TODO - haven't been fully tested for FP16 and BF, need to verify the false-positive
        if 'BufferLoad' in state and state['BufferLoad'] == True:
            TLUA = state['ProblemType']['TLUA']
            TLUB = state['ProblemType']['TLUB']
            MayShiftA = TLUA and state['AssertFree0ElementMultiple'] < state['GlobalReadVectorWidthA']
            MayShiftB = TLUB and state['AssertFree1ElementMultiple'] < state['GlobalReadVectorWidthB']
            subrv={}
            subrv['ShiftPtrElemB'] = state['GlobalReadVectorWidthB'] if MayShiftB else 0
            subrv['ShiftPtrElemA'] = state['GlobalReadVectorWidthA'] if MayShiftA else 0
            subrv['DUorMT1'] = state['DepthU'] if TLUB else state['MacroTile1']
            subrv['DUorMT0'] = state['DepthU'] if TLUA else state['MacroTile0']
            # value is also a dict for better readibility, client side need to handel the serialization
            rv += [cls('BufferLoadOffsetLimitCheck', value=subrv)]

        # When doing globol write, may need to load matrix C if beta !=0
        if 'BufferLoad' in state and state['BufferLoad'] == True:
            rv += [cls('BufferLoadOffsetLimitCheck_Beta', value=state['MacroTile1'])]

        # similiar check is applied for bufferstore,
        # for bufferstore offset, test if the bot-right offset < 2^32,
        # it should be StrideA*MT1, so we need to output MT1 and use the StrideA of problem in host-side for predication
        if 'BufferStore' in state and state['BufferStore'] == True:
            rv += [cls('BufferStoreOffsetLimitCheck', value=state['MacroTile1'])]

        if '_GlobalAccumulation' in state and state['_GlobalAccumulation'] != None and not isPersistent(state):
            value = MIN_K_FOR_GSU
            rv += [cls('GlobalSplitUCheckMinK', value=[value, state["GlobalSplitU"]])]

        if ('WorkGroupMappingXCC' in state) and ('WorkGroupMappingXCCGroup' in state):
            rv += [cls("WorkgroupMappingXCCCheck", value=[state['WorkGroupMappingXCC'], state['WorkGroupMappingXCCGroup']])]

        if state['ProblemType']['SwizzleTensorA']:
            rv += [cls('SwizzleTensorA', value=state['ProblemType']['SwizzleTensorA'])]

        if state['ProblemType']['SwizzleTensorB']:
            rv += [cls('SwizzleTensorB', value=state['ProblemType']['SwizzleTensorB'])]

        if state['ProblemType']['FusedGemmA2A']:
            rv += [cls('FusedA2ATileDivisible', value=state['MacroTile0'])]

        if state.get('ReuseAcrossPersistent', 0):
            # A lives in VGPRs for the whole persistent loop, so the kernel is only
            # correct for problems where every tile a workgroup visits reads the
            # same A, and where K is a whole number of resident k-tiles.
            #   M == MacroTile0     - one M-tile only, and no masked store in M. A
            #                         half-filled tile would take the edge path,
            #                         which v0 does not budget registers for.
            #   N % MacroTile1 == 0 - no edge tile in N either.
            #
            # K is a range, not a point. The loop shell owns every resident k-tile
            # and leaves as soon as the counter runs out, so any K from the floor up
            # to the count the kernel holds works and the k-tiles above it simply go
            # unused. K must still land on a k-tile boundary, because a section's
            # registers are picked by a codegen-time k-tile number.
            #
            # The floor is one k-tile, and the reason is the loop-entry guard: it
            # skips the loop when the counter reaches zero, and the InitCIterWmma
            # clone that zeroes C lives inside the loop, so at zero k-tiles C would
            # never be initialised. One is enough even at PrefetchGlobalRead 2 --
            # the pre-loop prefetch skips its second stage when the counter is 1
            # (label_skipPGR2_1), so nothing is fetched past this K, and the single
            # section computes and then leaves through its own early exit. The
            # toPGR1 escape that carries this case in a non-RAP kernel is not needed
            # here: it exists to hand the work to the NGLL drain, which RAP does not
            # emit because its sections do the work themselves.
            #
            # SizeGreaterThan and SizeLessThan are strict, hence the -1 and +1.
            kIdx = state['ProblemType']['NumIndicesC']
            kTiles = state['_RAPNumResidentKTiles']
            floorTiles = 1
            rv += [cls('SizeEqual', index=0, value=state['MacroTile0'])]
            rv += [cls('SizeMultiple', index=1, value=state['MacroTile1'])]
            rv += [cls('SizeMultiple', index=kIdx, value=state['DepthU'])]
            rv += [cls('SizeGreaterThan', index=kIdx,
                       value=floorTiles * state['DepthU'] - 1)]
            rv += [cls('SizeLessThan', index=kIdx,
                       value=kTiles * state['DepthU'] + 1)]

        return rv

    def state(self):
        record = getattr(self, "_record", None)
        if record is not None:
            return copy.deepcopy(record)
        return super().state()

class CustomKernel:
    StateKeys = ['name',
                 'args',
                 'macrotile',
                 'threads',
                 'grid',
                 'workspaceType',
                 'workspaceSizePerElemC',
                 'workspaceSizePerElemBias',
                 'generated']

    def state(self):
        return _state_from_document(self)

    def __init__(self, **kwargs):
        for (key, value) in list(kwargs.items()):
            setattr(self, key, value)

class SizeMapping:
    StateKeys = ['waveNum',
                 'workGroup',
                 'macroTile',
                 'threadTile',
                 'matrixInstruction',
                 'grvwA',
                 'grvwB',
                 'gwvwC',
                 'gwvwD',
                 'depthU',
                 'staggerU',
                 'staggerUMapping',
                 'globalSplitUPGR',
                 'globalSplitU',
                 'staggerStrideShift',
                 'workGroupMapping',
                 'packBatchDims',
                 'magicDivAlg',
                 'tileProcessingStrategy',
                 'workAssignment',
                 'streamKAtomic',
                 'prefetchAcrossPersistent',
                 'sourceKernel',
                 'globalAccumulation',
                 'adaptiveGemmGSUA',
                 'workspaceSizePerElemC',
                 'workspaceSizePerElemBias',
                 'activationFused',
                 'workGroupMappingXCC',
                 'workGroupMappingXCCGroup',
                 'globalSplitUCoalesced',
                 'globalSplitUWorkGroupMappingRoundRobin',
                 'CUOccupancy',
                 'PrefetchGlobalRead',
                 'MathClocksUnrolledLoop',
                 'synchronizerSizePerWG',
                 'nonTemporalA',
                 'nonTemporalB',
                 'temporalHintA',
                 'temporalHintB',
                 'hasTemporalHint',
                 'adaptiveGemmNTAB',
                 'customMainLoopScheduling',
                 'useSubtileImpl',
                 'NonTemporalD',
                 'WaveSeparateGlobalReadA',
                 'WaveSeparateGlobalReadB',
                 'UnrollLoopSwapGlobalReadOrder',
                 'DirectToVgprA',
                 'DirectToVgprB',
                 'NumLoadsCoalescedA',
                 'NumLoadsCoalescedB',
                 'WaveGroup',
                 'VectorWidthA',
                 'VectorWidthB',
                 'LocalSplitU',
                 'DirectToLdsA',
                 'DirectToLdsB',
                 'ExpertSchedulingMode',
                 'clusterDim'
                 ]

    def state(self):
        return _state_from_document(self)

    def __init__(self, **kwargs):
        for (key, value) in list(kwargs.items()):
            setattr(self, key, value)

class InternalArgsSupport:
    StateKeys = ['version',
                 'persistentLoopArgsVersion',
                 'gsu',
                 'wgm',
                 'staggerU',
                 'perTileExtraIters',
                 'useUniversalArgs',
                 'useSFC'
                 ]

    def state(self):
        return _state_from_document(self)

    def __init__(self, **kwargs):
        for (key, value) in list(kwargs.items()):
            setattr(self, key, value)

class Solution:
    StateKeys = ['name',
                 'kernelName',
                 'problemType',
                 'hardwarePredicate',
                 'problemPredicate',
                 'taskPredicate',
                 'sizeMapping',
                 'customKernel',
                 'internalArgsSupport',
                 'debugKernel',
                 'libraryLogicIndex',
                 'index',
                 'ideals',
                 'linearModel']
    HiddenKeys = ['originalSolution']

    @classmethod
    def FromSolutionStruct(
        cls,
        solution,
        splitGSU: bool,
        printSolutionRejectionReason: bool,
        printIndexAssignmentInfo: bool,
        assembler: Assembler,
        isaInfoMap: Dict[str, IsaInfo]
    ):
        return cls.FromOriginalState(
                   solution._state,
                   splitGSU,
                   printSolutionRejectionReason,
                   printIndexAssignmentInfo,
                   assembler,
                   isaInfoMap,
                   solution.srcName,
                   raiseProblemTypeOnTypeMismatch=False,
               )

    @classmethod
    def FromOriginalState(
            cls,
            d,
            splitGSU: bool,
            printSolutionRejectionReason: bool,
            printIndexAssignmentInfo: bool,
            #mink
            assembler,
            isaInfoMap,
            srcName = "",
            deviceInfo=None,
            *,
            raiseProblemTypeOnTypeMismatch: bool = True,
        ):
        if 'ISA' not in d:
            if d['KernelLanguage'] == 'Assembly':
                d['ISA'] = gfxToIsa(deviceInfo[1])

        if 'CUCount' not in d:
            d['CUCount'] = None

        record = _solution_record(d)
        rv = cls()
        rv.name = record.get("name")
        rv.kernelName = record.get("kernelName")
        rv.index = record.get("index")
        rv.debugKernel = record.get("debugKernel", False)
        rv.libraryLogicIndex = record.get("libraryLogicIndex", -1)
        rv.ideals = record.get("ideals", {})
        rv.linearModel = record.get("linearModel", {})

        # Index geometry and data-type objects are not in the library record.
        # Client config reads them off this object. state() returns the record.
        rv.problemType = ProblemType.FromOriginalState(d["ProblemType"])
        problemTypeRecord = record["problemType"]
        rv.problemType.mxScaleFormat = problemTypeRecord.get("mxScaleFormat", 0)
        rv.problemType._record = problemTypeRecord

        rv.problemPredicate = _predicate_from_document(ProblemPredicate, record["problemPredicate"])
        rv.taskPredicate = _predicate_from_document(TaskPredicate, record["taskPredicate"])
        rv.hardwarePredicate = _predicate_from_document(
            Hardware.HardwarePredicate, record["hardwarePredicate"]
        )
        rv.sizeMapping = _fill_document(SizeMapping(), record["sizeMapping"])
        if "customKernel" in record:
            rv.customKernel = _fill_document(CustomKernel(), record["customKernel"])
        else:
            rv.customKernel = {}
        rv.internalArgsSupport = _fill_document(
            InternalArgsSupport(), record["internalArgsSupport"]
        )

        rv.originalSolution = OriginalSolution(
                                  d,
                                  splitGSU,
                                  printSolutionRejectionReason,
                                  printIndexAssignmentInfo,
                                  assembler,
                                  isaInfoMap,
                                  srcName,
                                  raiseProblemTypeOnTypeMismatch=raiseProblemTypeOnTypeMismatch,
                              )
        rv.srcName = srcName

        return rv

    def __init__(self, **kwargs):
        self.name = None
        self.problemType = None
        self.hardwarePredicate = Hardware.HardwarePredicate('TruePred')
        self.problemPredicate = ProblemPredicate('TruePred')
        self.taskPredicate = TaskPredicate('TruePred')
        self.sizeMapping = None
        self.customKernel = None
        self.debugKernel = False
        self.libraryLogicIndex = {}
        self.index = None
        self.ideals = {}
        self.srcName = ""

        for key, value in kwargs:
            if key not in Solution.StateKeys and key not in Solution.HiddenKeys:
                raise KeyError("{0} is not a property of Solution.".format(key))

            setattr(self, key, value)

    def getSolutionKeys(self):
        return self.originalSolution.keys()
