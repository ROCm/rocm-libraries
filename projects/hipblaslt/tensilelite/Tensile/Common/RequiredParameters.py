################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the
# Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

from functools import lru_cache
from .ValidParameters import validParameters


@lru_cache
def getRequiredParametersFull() -> set:
    return frozenset(validParameters.keys())


@lru_cache
def getRequiredParametersMin() -> set:
    return frozenset({
        '1LDSBuffer',
        'ActivationFuncCall',
        'AssertFree0ElementMultiple',
        'AssertFree1ElementMultiple',
        'AssertSummationElementMultiple',
        'ClusterLocalRead',
        'ConvertAfterDS',
        'DirectToVgprA',
        'DirectToVgprB',
        'DirectToLdsA',
        'DirectToLdsB',
        'ExpandPointerSwap',
        'ForceDisableShadowInit',
        'GlobalReadPerMfma',
        'GlobalReadVectorWidthA',
        'GlobalReadVectorWidthB',
        'GlobalSplitUAlgorithm',
        'GroupLoadStore',
        'ISA',
        'InnerUnroll',
        'Kernel',
        'LdsBlockSizePerPadA',
        'LdsBlockSizePerPadB',
        'LdsBlockSizePerPadMetadata',
        'LdsPadA',
        'LdsPadB',
        'LdsPadMetadata',
        'LDSTrInst',
        'LocalReadVectorWidth',
        'LocalWritePerMfma',
        'MIArchVgpr',
        'MaxOccupancy',
        'NonTemporal',
        'NonTemporalA',
        'NonTemporalB',
        'NonTemporalC',
        'NonTemporalD',
        'NonTemporalMetadata',
        'NumElementsPerBatchStore',
        'NumLoadsCoalescedA',
        'NumLoadsCoalescedB',
        'OptNoLoadLoop',
        'PrefetchGlobalRead',
        'PrefetchLocalRead',
        'PreloadKernArgs',
        'ScheduleIterAlg',
        'SourceSwap',
        'SpaceFillingAlgo',
        'StorePriorityOpt',
        'StoreRemapVectorWidth',
        'StoreSyncOpt',
        'StoreVectorWidth',
        'StreamK',
        'StreamKXCCMapping',
        'StreamKFixupTreeReduction',
        'TransposeLDS',
        'UnrollLoopSwapGlobalReadOrder',
        'Use64bShadowLimit',
        'UseInstOffsetForGRO',
        'UseSgprForGRO',
        'VectorStore',
        'VectorWidthA',
        'VectorWidthB',
        'WaveSeparateGlobalReadA',
        'WaveSeparateGlobalReadB',
        'WavefrontSize',
        'WorkGroup',
    })
