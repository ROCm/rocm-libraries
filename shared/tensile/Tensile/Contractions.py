################################################################################
#
# Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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

from .DataType import DataType
from . import Hardware
from . import Properties
from .SolutionStructs import Solution as OriginalSolution
from .Utils import state, state_key_ordering

from . import Common

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
    StateKeys = ['operationIdentifier', 'aType', 'bType', 'cType', 'dType',
                 'useBeta', 'highPrecisionAccumulate', 'useInitialStridesAB', 'useInitialStridesCD', 'stridedBatched', 'f32XdlMathOp', 'stochasticRounding']
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

        rv.aType = srcType
        rv.bType = srcType
        # for hybrid 8bit float types, we need to split the type into a_type and b_type
        if srcType.isFloat8BFloat8():
            rv.aType = DataType("F8")
            rv.bType = DataType("B8")
        elif srcType.isBFloat8Float8():
            rv.aType = DataType("B8")
            rv.bType = DataType("F8")

        # We don't expect dstType and computeType to be hybrid types for now
        rv.cType = dstType
        rv.dType = dstType
        # we already checked the src/dst/compute types are supported and well-assigned in SolutionStruct
        rv.alphaType = computeType
        rv.betaType = computeType

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

        rv.setConstStrideA = []
        if 'SetConstStrideA' in d:
            rv.setConstStrideA = d['SetConstStrideA']
        rv.setConstStrideB = []
        if 'SetConstStrideB' in d:
            rv.setConstStrideB = d['SetConstStrideB']
        rv.zeroPadA=[]
        if 'ZeroPadA' in d:
            rv.zeroPadA = d['ZeroPadA']
        rv.zeroPadB=[]
        if 'ZeroPadB' in d:
            rv.zeroPadB = d['ZeroPadB']

        rv.mirrorDimsA = d['MirrorDimsA'] if 'MirrorDimsA' in d else []
        rv.mirrorDimsB = d['MirrorDimsB'] if 'MirrorDimsB' in d else []

        rv.useBeta = True
        if 'UseBeta' in d:
            rv.useBeta = d['UseBeta']

        rv.batched = d['Batched']

        rv.f32XdlMathOp = DataType(d['F32XdlMathOp']) if 'F32XdlMathOp' in d else DataType(0)

        rv.stochasticRounding = False
        if 'StochasticRounding' in d:
            rv.stochasticRounding = d['StochasticRounding']

        return rv

    def __init__(self, freeIndices=None, batchIndices=None, boundIndices=None, aDims=None, bDims=None, cDims=None, dDims=None):
        self.convolution = None
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

    def predicates(self, includeBatch=False, includeOperation=False, includeType=False):
        predicates = []

        #if includeBatch and not self.batched:
        #    predicates.append(ProblemPredicate("BatchSizeEqual", index=0, value=1))

        if includeOperation:
            predicates.append(ProblemPredicate("OperationIdentifierEqual", value=self.operationIdentifier))
            if not self.useBeta:
                predicates.append(ProblemPredicate("BetaZero"))
            predicates.append(ProblemPredicate("StridedBatched", value=self.stridedBatched))

        if includeType:
            predicates.append(ProblemPredicate("TypesEqual", value=(self.aType, self.bType, self.cType, self.dType)))
            predicates.append(ProblemPredicate("HighPrecisionAccumulate", value=self.highPrecisionAccumulate))
            predicates.append(ProblemPredicate("F32XdlMathOp", value=self.f32XdlMathOp))
            predicates.append(ProblemPredicate("StochasticRounding", value=self.stochasticRounding))


        return predicates

def extractDimPredicate(cls, key, value, predicateName):
    """
    Extract the predicate for AssertStrideEqual*
    Value is a dictionary
    """
    predicates = []
    for pos,val in value.items():
        if val != -1:
            predicates.append(cls(predicateName, index=pos, value=val))
    if len(predicates) == 1:
        return predicates[0]
    elif len(predicates) > 1:
        return cls.And(predicates)

class ProblemPredicate(Properties.Predicate):
    @classmethod
    def FromOriginalKeyPair(cls, pair):
        (key, value) = pair
        if key == "AssertStrideAEqual":
            return extractDimPredicate(cls, key, value, "StrideAEqual")
        if key == "AssertStrideBEqual":
            return extractDimPredicate(cls, key, value, "StrideBEqual")
        if key == "AssertStrideCEqual":
            return extractDimPredicate(cls, key, value, "StrideCEqual")
        if key == "AssertStrideDEqual":
            return extractDimPredicate(cls, key, value, "StrideDEqual")

        if key == "AssertSizeEqual":
            return extractDimPredicate(cls, key, value, "SizeEqual")
        if key == "AssertSizeGreaterThan":
            return extractDimPredicate(cls, key, value, "SizeGreaterThan")
        if key == "AssertSizeLessThan":
            return extractDimPredicate(cls, key, value, "SizeLessThan")
        if key == "AssertSizeMultiple":
            return extractDimPredicate(cls, key, value, "SizeMultiple")

        # Arithmetic intensity assertions
        if key == "AssertAIGreaterThanEqual":
            return cls("AIGreaterThanEqual", value=value) if value > 0 else None
        if key == "AssertAILessThanEqual":
            return cls("AILessThanEqual", value=value) if value > 0 else None

        # Alpha and beta value assertions
        if key == "AssertAlphaValue":
            return cls("AlphaValue", value=str(value)) if value != False else None
        if key == "AssertBetaValue":
            return cls("BetaValue", value=str(value)) if value != False else None

        if key == "AssertCEqualsD":
            return cls("CEqualsD") if value != False else None

        # TODO - remove this when logic files have been updated
        if key == 'AssertMinApproxSize':
            return None

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

        if key == "Fp16AltImpl":
            return cls("Fp16AltImpl") if value != False else None

        if key == "Fp16AltImplRound":
            return cls("Fp16AltImplRound") if value != False else None

    @classmethod
    def CompoundPredicates(cls, state, problemType):
        rv = []

        if not problemType.aType.isInt8x4():
            # calculate the minimum supported free dimension size
            TLUA = state['ProblemType']['TLUA']
            TLUB = state['ProblemType']['TLUB']
            minFree0 = max(state['GlobalLoadVectorWidthA'], 1) if TLUA else 1
            minFree1 = max(state['GlobalLoadVectorWidthB'], 1) if TLUB else 1
            rv += [cls('LeadingFree0SizesGreaterOrEqual', value=minFree0)]
            rv += [cls('LeadingFree1SizesGreaterOrEqual', value=minFree1)]

        if len(state["PackedC0IndicesX"]) > 1:
          rv += [cls("CDStridesEqual")]

        if "KernelLanguage" in state:
            rv += [cls("KernelLanguageCompatible", value=state["KernelLanguage"])]

        if ('GlobalSplitU' in state) and (state['GlobalSplitU'] > 1):
            if ('_GlobalAccumulation' not in state) or (state['_GlobalAccumulation'] != 'MultipleBuffer'):
                rv += [cls("DeterministicMode", value = False)]

        if ('StreamK' in state) and (state['StreamK'] > 0) and ('StreamKAtomic' in state) and (state['StreamKAtomic'] == 1):
            # StreamKAtomic = 1 uses atomic for partial tiles
            rv += [cls("DeterministicMode", value = False)]

        # debugging: mark this set to allow the problem always runnable with PK
        if 'PersistentKernel' in state and state['PersistentKernel']:
            rv += [cls("PersistentKernelCheck")]

        if ("MatrixInstruction" in state and state["MatrixInstruction"]) or \
           ("EnableMatrixInstruction" in state and state["EnableMatrixInstruction"] is True):
            rv += [cls("ArithmeticUnitCompatible", value="MFMA")]
        else:
            rv += [cls("ArithmeticUnitCompatible", value="VALU")]

        # if bufferload is performed, we output some predication info for host side,
        # to prevent from some extremely large problems from launching and causing bufferload offset limit < 2^32
        # those cases will not satisfy the assertion thus won't use the kernel.
        # See Common.py for more details, we will need four values:
        # TODO - haven't been fully tested for FP16 and BF, need to verify the false-positive
        if 'BufferLoad' in state and state['BufferLoad'] == True:
            TLUA = state['ProblemType']['TLUA']
            TLUB = state['ProblemType']['TLUB']
            MayShiftA = TLUA and state['AssertFree0ElementMultiple'] < state['GlobalLoadVectorWidthA']
            MayShiftB = TLUB and state['AssertFree1ElementMultiple'] < state['GlobalLoadVectorWidthB']
            subrv={}
            subrv['ShiftPtrElemB'] = state['GlobalLoadVectorWidthB'] if MayShiftB else 0
            subrv['ShiftPtrElemA'] = state['GlobalLoadVectorWidthA'] if MayShiftA else 0
            subrv['DUorMT1'] = state['DepthU'] if TLUB else state['MacroTile1']
            subrv['DUorMT0'] = state['DepthU'] if TLUA else state['MacroTile0']
            # value is also a dict for better readability, client side need to handel the serialization
            rv += [cls('BufferLoadOffsetLimitCheck', value=subrv)]

        # similar check is applied for bufferstore,
        # for bufferstore offset, test if the bot-right offset < 2^32,
        # it should be StrideA*MT1, so we need to output MT1 and use the StrideA of problem in host-side for predication
        if 'BufferStore' in state and state['BufferStore'] == True:
            val = state['MacroTile1']
            if (1 in state["AssertSizeLessThan"].keys()):
                # use smaller value if array size assert is enabled
                val = min(val, state["AssertSizeLessThan"][1] - 1)
            rv += [cls('BufferStoreOffsetLimitCheck', value=val)]

        if '_GlobalAccumulation' in state and state['_GlobalAccumulation'] != None and not state["StreamK"]:
            value = state['MinKForGSU'] * state['GlobalSplitU']
            rv += [cls('GlobalSplitUCheckMinK', value=value)]

        return rv

    @classmethod
    def FromOriginalState(cls, d, problemType, morePreds=[]):
        problemTypePreds = problemType.predicates(True, True, True)
        compoundPreds = cls.CompoundPredicates(d, problemType)
        extraPreds = problemTypePreds + compoundPreds + morePreds

        predicates = [p for p in map(cls.FromOriginalKeyPair, d.items()) if p is not None] + extraPreds
        return cls.And(predicates)

class TaskPredicate(Properties.Predicate):
    @classmethod
    def FromOriginalKeyPair(cls, pair):
        (key, value) = pair
        if key == "_WorkspaceSizePerElemC" and value > 0:
            return cls("WorkspaceCheck")
        return None

    @classmethod
    def FromOriginalState(cls, d, problemType, morePreds=[]):
        predicates = [p for p in map(cls.FromOriginalKeyPair, d.items()) if p is not None]
        return cls.And(predicates)

class SizeMapping:
    StateKeys = ['workGroup',
                 'macroTile',
                 'threadTile',
                 'depthU',
                 'staggerU',
                 'globalSplitU',
                 'staggerStrideShift',
                 'workGroupMapping',
                 'packSummationDims',
                 'packBatchDims',
                 'magicDivAlg',
                 'streamK',
                 'streamKAtomic',
                 'persistentKernel',
                 'persistentKernelAlongBatch',
                 'sourceKernel',
                 'globalAccumulation',
                 'workspaceSizePerElemC',
                 'preloadKernargs'
                 ]

    @classmethod
    def FromOriginalState(cls, d):
        globalAccum = 0
        if d['_GlobalAccumulation'] == 'SingleBuffer':
            globalAccum = 1
        if d['_GlobalAccumulation'] == 'MultipleBuffer':
            globalAccum = 2
        if d['_GlobalAccumulation'] == 'PartialsBuffer':
            globalAccum = 3

        assert d["PreloadKernelArguments"] in (0, 1), d["PreloadKernelArguments"]
        return cls(workGroup             = d['WorkGroup'],
                   macroTile             = cls.ReadOriginalMacroTile(d),
                   threadTile            = d['ThreadTile'],
                   workGroupMapping      = d['WorkGroupMapping'],
                   staggerU              = d['StaggerU'] if 'StaggerU' in d else 0,
                   depthU                = d['DepthU'],
                   globalSplitU          = d['GlobalSplitU'],
                   staggerStrideShift    = d['_staggerStrideShift'] if '_staggerStrideShift' in d else 0,
                   packSummationDims     = d['PackSummationDims'] if 'PackSummationDims' in d else 0,
                   packBatchDims         = d['PackBatchDims'] if 'PackBatchDims' in d else 0,
                   streamK               = d['StreamK'] if 'StreamK' in d else 0,
                   streamKAtomic         = d['StreamKAtomic'] if 'StreamKAtomic' in d else 0,
                   persistentKernel      = d['PersistentKernel'] if 'PersistentKernel' in d else 0,
                   persistentKernelAlongBatch   = d['PersistentKernelAlongBatch'] if 'PersistentKernelAlongBatch' in d else False,
                   magicDivAlg           = d.get('MagicDivAlg', 1),
                   sourceKernel          = d['KernelLanguage'] == 'Source',
                   globalAccumulation    = globalAccum,
                   workspaceSizePerElemC = d['_WorkspaceSizePerElemC'],
                   preloadKernargs       = d["PreloadKernelArguments"]
                   )

    @classmethod
    def ReadOriginalMacroTile(cls, d):
        rv = [1,1,1]
        rv[0] = d['MacroTile0']
        rv[1] = d['MacroTile1']
        return rv

    def __init__(self, **kwargs):
        for (key, value) in list(kwargs.items()):
            setattr(self, key, value)

class Solution:
    StateKeys = ['name',
                'problemType',
                'hardwarePredicate',
                'problemPredicate',
                'taskPredicate',
                'sizeMapping',
                'debugKernel',
                'libraryLogicIndex',
                'index',
                'ideals',
                'linearModel']
    HiddenKeys = ['originalSolution']

    @classmethod
    def FromSolutionStruct(cls, solution):
        return cls.FromOriginalState(solution._state)

    @classmethod
    def FromOriginalState(cls, d, deviceInfo=None):
        rv = cls()


        if 'SolutionNameMin' in d:
            rv.name = d['SolutionNameMin']

        rv.problemType = ProblemType.FromOriginalState(d['ProblemType'])

        rv.problemPredicate = ProblemPredicate.FromOriginalState(d, rv.problemType)
        rv.taskPredicate = TaskPredicate.FromOriginalState(d, rv.problemType)

        if 'DebugKernel' in d:
            rv.debugKernel = d['DebugKernel']

        if 'SolutionIndex' in d:
            rv.index = d['SolutionIndex']

        info = cls.ReadOriginalInfo(d)
        rv.libraryLogicIndex = int(info.get("SolutionIndex", -1))

        rv.sizeMapping = SizeMapping.FromOriginalState(d)
        if 'Ideals' in d:
            rv.ideals = d['Ideals']
        else:
            rv.ideals = {}

        if 'LinearModel' in d:
            rv.linearModel = d['LinearModel']
        else:
            rv.linearModel = {}

        if 'ISA' not in d:
            if d['KernelLanguage'] == 'Assembly':
                d['ISA'] = Common.gfxArch(deviceInfo[1])
            else:
                d['ISA'] = [0,0,0]

        rv.originalSolution = OriginalSolution(d)
        # hacky, can just construct Convolution yet again?
        rv.problemType.convolution = rv.originalSolution["ProblemType"].convolution

        return rv

    @classmethod
    def ReadOriginalInfo(cls, d):
        return dict([(key, str(value)) for (key, value) in list(d.items()) if key != 'ProblemType'])

    def __init__(self, **kwargs):
        self.name = None
        self.problemType = None
        self.hardwarePredicate = Hardware.HardwarePredicate('TruePred')
        self.problemPredicate = ProblemPredicate('TruePred')
        self.taskPredicate = TaskPredicate('TruePred')
        self.sizeMapping = None
        self.debugKernel = False
        self.libraryLogicIndex = {}
        self.index = None
        self.ideals = {}

        for key, value in kwargs:
            if key not in Solution.StateKeys and key not in Solution.HiddenKeys:
                raise KeyError("{0} is not a property of Solution.".format(key))

            setattr(self, key, value)
