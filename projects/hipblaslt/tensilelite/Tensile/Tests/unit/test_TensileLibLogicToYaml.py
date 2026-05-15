################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

import pytest
from unittest.mock import mock_open, patch
import sys
import os
import tempfile
import subprocess
import filecmp

from Tensile import TensileLibLogicToYaml

# Test data
VALID_LIBLOGIC_FILE_CONTENT = """
- {MinimumRequiredVersion: 5.0.0}
- gfx950
- gfx950
- [Device 75a0]
- Activation: true
  ActivationComputeDataType: 0
  ActivationNoGuard: false
  ActivationType: none
  AllowNoFreeDims: false
  AssignedDerivedParameters: true
  Batched: true
  BetaOnlyUseBias: false
  BiasDataTypeList: [0, 7]
  BiasSrc: D
  ComplexConjugateA: false
  ComplexConjugateB: false
  ComputeDataType: 0
  DataType: 7
  DataTypeA: 7
  DataTypeAmaxD: 0
  DataTypeB: 7
  DataTypeE: 7
  DestDataType: 7
  F32XdlMathOp: 0
  Gradient: false
  GroupedGemm: false
  HighPrecisionAccumulate: true
  Index0: 0
  Index01A: 0
  Index01B: 1
  Index1: 1
  IndexAssignmentsA: [3, 0, 2]
  IndexAssignmentsB: [3, 1, 2]
  IndexAssignmentsLD: [4, 5, 6, 7]
  IndexAssignmentsMetadata: [3, 0, 2]
  IndexUnroll: 3
  IndexUnrollA: 0
  IndexUnrollB: 0
  IndexUnrollM: 0
  IndicesBatch: [2]
  IndicesFree: [0, 1]
  IndicesSummation: [3]
  MirrorDimsA: []
  MirrorDimsB: []
  MirrorDimsMetadata: []
  NumIndicesBatch: 1
  NumIndicesC: 3
  NumIndicesFree: 2
  NumIndicesLD: 4
  NumIndicesSummation: 1
  OperationType: GEMM
  OutputAmaxD: false
  SetConstStrideA: []
  SetConstStrideB: []
  SetConstStrideBias: []
  SilentHighPrecisionAccumulate: false
  Sparse: 0
  StochasticRounding: false
  StridedBatched: true
  SupportUserArgs: true
  SwizzleTensorA: false
  SwizzleTensorB: false
  TLUA: false
  TLUB: false
  Tensor0: 0
  Tensor1: 1
  TileA: 0
  TileAwareSelection: false
  TileB: 1
  TotalIndices: 4
  TransposeA: true
  TransposeB: false
  UseBeta: true
  UseBias: 1
  UseE: false
  UseInitialStridesAB: false
  UseInitialStridesCD: false
  UseScaleAB: ''
  UseScaleAlphaVec: 1
  UseScaleCD: false
- - 1LDSBuffer: 0
    ActivationAlt: false
    ActivationFuncCall: false
    ActivationFused: true
    AdaptiveGemm: 0
    AssertAIGreaterThanEqual: -1
    AssertAILessThanEqual: -1
    AssertFree0ElementMultiple: 1
    AssertFree1ElementMultiple: 1
    AssertSummationElementMultiple: 1
    AssignedDerivedParameters: true
    AssignedProblemIndependentDerivedParameters: true
    BaseName: Cijk_Alik_Bljk_BBS_BH_Bias_SAV_UserArgs_MT96x96xQmnABcT7HWcD6o5PKlLr1Vf4E4hlypHXHNWsDqsqxdQ=
    BufferLoad: true
    BufferStore: true
    CUCount: null
    CUOccupancy: -1
    ClusterLocalRead: 0
    CodeObjectVersion: '4'
    ConvertAfterDS: false
    CustomKernelName: ''
    DebugStreamK: 0
    DepthU: 128
    DirectToLds: false
    DirectToLdsA: false
    DirectToLdsB: false
    DirectToVgprA: false
    DirectToVgprB: false
    DirectToVgprSparseMetadata: false
    EdgeType: ShiftPtr
    EnableF32XdlMathOp: false
    EnableMatrixInstruction: true
    ExpandPointerSwap: 0
    ExpertSchedulingMode: 0
    ForceDisableShadowInit: false
    ForceUnrollSubIter: false
    GlobalReadPerMfma: 1
    GlobalReadVectorWidthA: 8
    GlobalReadVectorWidthB: 8
    GlobalSplitU: 0
    GlobalSplitUAlgorithm: MultipleBuffer
    GlobalSplitUCoalesced: false
    GlobalSplitUWorkGroupMappingRoundRobin: false
    GlobalWriteVectorWidth: 1
    GroupLoadStore: false
    GuaranteeNoPartialA: true
    GuaranteeNoPartialB: true
    GuaranteeNoPartialMetadata: true
    ISA: [9, 5, 0]
    InnerUnroll: 1
    InterleaveAlpha: 0
    InternalSupportParams: {KernArgsVersion: 2, SupportCustomStaggerU: true, SupportCustomWGM: true,
      SupportUserGSU: false, UseSFC: false, UseUniversalArgs: true}
    Kernel: true
    KernelLanguage: Assembly
    KernelNameMin: Cijk_Alik_Bljk_BBS_BH_Bias_SAV_UserArgs_MT96x96x128_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT3_3_MO40_NTn1_NTA2_NTB7_NTC0_NTD4_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO1_SRVW0_SSO4_SVW1_SK3_SKFTR0_SKXCCM8_TLDS2_ULSGRO0_USL1_UIOFGRO0_USFGRO1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1
    LDSTrInst: false
    LSCA: 128
    LSCB: 128
    LSPA: 16
    LSPB: 16
    LVCA: 16
    LVCB: 16
    LVPA: 2
    LVPB: 2
    LdsBlockSizePerPadA: 256
    LdsBlockSizePerPadB: 256
    LdsBlockSizePerPadMetadata: 0
    LdsBytesNoAmax: 120832
    LdsInitCVgprs: false
    LdsNumBytes: 120832
    LdsNumElementsAlignedA: 27648
    LdsNumElementsAlignedB: 27648
    LdsNumElementsAlignedMetadata: 0
    LdsOffsetA: 0
    LdsOffsetA_Blk: 65536
    LdsOffsetB: 27648
    LdsOffsetB_Blk: 93184
    LdsOffsetBias: 0
    LdsOffsetBiasGSU: 0
    LdsOffsetBiasNonGSU: 0
    LdsOffsetMetadata: 27648
    LdsOffsetMetadata_Blk: 93184
    LdsPadA: 16
    LdsPadB: 16
    LdsPadMetadata: 0
    LocalReadVectorWidth: 8
    LocalSplitU: 1
    LocalSplitUReuseLDS: 1
    LocalWritePerMfma: -1
    LocalWriteUseSgprA: false
    LocalWriteUseSgprB: false
    LoopIters: 4
    LoopUnroll: 128
    MFMA_BF16_1K: false
    MIArchVgpr: 0
    MIBlock: [16, 16, 32, 1, 1, 1]
    MIInputPerThread: 8
    MIInputPerThreadA: 8
    MIInputPerThreadB: 8
    MIInputPerThreadMetadata: 8
    MIOutputVectorWidth: 4
    MIRegPerOut: 1
    MIWaveGroup: [2, 2]
    MIWaveTile: [3, 3]
    MIWaveTileA: 3
    MIWaveTileB: 3
    MIWaveTileMetadata: 0
    MacroTile0: 96
    MacroTile1: 96
    MacroTileA: 96
    MacroTileB: 96
    MagicDivAlg: 2
    MathClocksUnrolledLoop: 0
    MatrixInstB: 1
    MatrixInstBM: 1
    MatrixInstBN: 1
    MatrixInstK: 32
    MatrixInstM: 16
    MatrixInstN: 16
    MatrixInstruction: [16, 16, 32, 1]
    MaxLDS: 163840
    MaxOccupancy: 40
    MbskPrefetchMethod: 0
    MfmaInitCVgprs: false
    NoLdsWriteCode: false
    NoReject: false
    NoTailLoop: false
    NonDTLTailLoopA: true
    NonDTLTailLoopB: true
    NonTemporal: -1
    NonTemporalA: 2
    NonTemporalB: 7
    NonTemporalC: 0
    NonTemporalD: 4
    NonTemporalE: 0
    NonTemporalMetadata: 0
    NonTemporalWS: 0
    NumElementsPerBatchStore: 0
    NumElementsPerThread: 36
    NumGlobalWriteVectorsPerThread: 36
    NumLoadsA: 6
    NumLoadsB: 6
    NumLoadsCoalescedA: 1
    NumLoadsCoalescedB: 1
    NumLoadsPerpendicularA: 6
    NumLoadsPerpendicularB: 6
    NumThreads: 256
    NumTotalPackedLoadsA: -1
    NumTotalPackedLoadsB: -1
    NumWaveSplitK: 1
    OptNoLoadLoop: 1
    PackedC0IdxChars: [I]
    PackedC0IndicesX: [0]
    PackedC1IdxChars: [J]
    PackedC1IndicesX: [1]
    PrefetchGlobalRead: 2
    PrefetchLocalRead: 1
    PreloadKernArgs: true
    SFCWGM:
    - [1, 1]
    - [1, 1]
    ScheduleGlobalRead: 1
    ScheduleIterAlg: 3
    ScheduleLocalWrite: 1
    SolutionIndex: 0
    SolutionNameMin: Cijk_Alik_Bljk_BBS_BH_Bias_SAV_UserArgs_MT96x96x128_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GSUC0_GSUWGMRR0_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT3_3_MO40_NTn1_NTA2_NTB7_NTC0_NTD4_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SU8_SUM0_SUS256_SPO1_SRVW0_SSO4_SVW1_SK3_SKFTR0_SKXCCM8_TLDS2_ULSGRO0_USL1_UIOFGRO0_USFGRO1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1_WGM1_WGMXCC2_WGMXCCGn1
    SourceSwap: 1
    SpaceFillingAlgo: []
    StaggerU: 8
    StaggerUMapping: 0
    StaggerUStride: 256
    StorePriorityOpt: 1
    StoreRemapVectorWidth: 0
    StoreSwapAddr: false
    StoreSyncOpt: 4
    StoreVectorWidth: 1
    StreamK: 3
    StreamKAtomic: 0
    StreamKFixupTreeReduction: 0
    StreamKXCCMapping: 8
    SubGroup0: 8
    SubGroup1: 32
    SubGroupA: 8
    SubGroupB: 32
    SuppressNoLoadLoop: false
    SwapGlobalReadOrder: false
    ThreadTile: [1, 1]
    ThreadTile0: 12
    ThreadTile1: 3
    ThreadTileA: 12
    ThreadTileB: 3
    TransposeLDS: 2
    TransposeLDSMetadata: 1
    ULSGRODoubleG2L: 0
    UnrollLoopSwapGlobalReadOrder: 0
    UnrollMajorLDSA: 1
    UnrollMajorLDSB: 1
    UnrollMajorLDSMetadata: true
    Use64bShadowLimit: 1
    UseCustomMainLoopSchedule: 0
    UseDirect32XEmulation: false
    UseDot2F32XEmulation: false
    UseDotInstruction: false
    UseF32XEmulation: false
    UseGeneralizedNLCOneA: false
    UseGeneralizedNLCOneB: false
    UseGeneralizedNLCOneMetadata: false
    UseInstOffsetForGRO: 0
    UsePLRPack: false
    UseSgprForGRO: 1
    Valid: true
    VectorStore: -1
    VectorWidthA: 1
    VectorWidthB: 1
    WaveSeparateGlobalReadA: 0
    WaveSeparateGlobalReadB: 0
    WaveSeparateGlobalReadMetadata: 0
    WaveSplitK: false
    WavefrontSize: 64
    WorkGroup: [32, 8, 1]
    WorkGroupMapping: 1
    WorkGroupMappingXCC: 2
    WorkGroupMappingXCCGroup: -1
    WorkGroupReduction: false
    WorkspaceCheck: [4, 0, 0]
    _DepthU: 128
    _DepthUA: 128
    _DepthUB: 128
    _DepthUMetadata: 128
    _GlobalAccumulation: PartialsBuffer
    _UseSgprForGRO: 1
    _VectorStore: 1
    _WorkspaceSizePerElemBias: 0
    _WorkspaceSizePerElemC: 4
    _staggerStrideShift: 0
    enableGLTrA: false
    enableGLTrB: false
    enableLDSTrA: false
    enableLDSTrB: false
    numSubTiles: 1
    reorderGRInstForDTVA: false
    reorderGRInstForDTVB: false
    tailLoopOptA: true
    tailLoopOptB: true
- [2, 3, 0, 1]
- - - [768, 3072, 1, 3840, 768, 768, 3840, 3840]
    - [0, 0.0]
- null
- null
- DeviceEfficiency
- GridBased
"""

VALID_CONFIG_FILE_CONTENT = """GlobalParameters:
  MinimumRequiredVersion: 5.0.0
  SleepPercent: 0
  KernelTime: true
  NumElementsToValidate: 0
  DataInitTypeBeta: 1
  DataInitTypeAlpha: 1
  DataInitTypeA: 12
  DataInitTypeB: 13
  DataInitTypeC: 12
  DataInitTypeD: 12
  PreciseKernelTime: 0
  Device: 0
  SkipSlowSolutionRatio: 0
  KeepBuildTmp: false
BenchmarkProblems:
- - OperationType: GEMM
    Activation: true
    Batched: true
    ComputeDataType: 0
    DataType: 7
    DataTypeA: 7
    DataTypeB: 7
    DataTypeE: 7
    DestDataType: 7
    HighPrecisionAccumulate: true
    IndexAssignmentsA: [3, 0, 2]
    IndexAssignmentsB: [3, 1, 2]
    IndexAssignmentsLD: [4, 5, 6, 7]
    NumIndicesC: 3
    TransposeA: true
    TransposeB: false
    UseBias: 1
    UseScaleAlphaVec: 1
  - InitialSolutionParameters:
    BenchmarkCommonParameters:
    - KernelLanguage: [Assembly]
    ForkParameters:
    - ActivationFuncCall: [false]
    - ClusterLocalRead: [0]
    - DepthU: [128]
    - ExpandPointerSwap: [0]
    - GlobalReadVectorWidthA: [8]
    - GlobalReadVectorWidthB: [8]
    - GlobalSplitU: [0]
    - LdsBlockSizePerPadA: [256]
    - LdsBlockSizePerPadB: [256]
    - LdsPadA: [16]
    - LdsPadB: [16]
    - LocalReadVectorWidth: [8]
    - MaxLDS: [163840]
    - MbskPrefetchMethod: [0]
    - NonTemporalA: [2]
    - NonTemporalB: [7]
    - NonTemporalD: [4]
    - PrefetchGlobalRead: [2]
    - SourceSwap: [1]
    - StaggerU: [8]
    - StorePriorityOpt: [1]
    - StoreSyncOpt: [4]
    - StoreVectorWidth: [1]
    - StreamK: [3]
    - StreamKXCCMapping: [8]
    - ThreadTile: [[1, 1]]
    - TransposeLDS: [2]
    - TransposeLDSMetadata: [1]
    - UseCustomMainLoopSchedule: [0]
    - UseSgprForGRO: [1]
    - VectorWidthA: [1]
    - VectorWidthB: [1]
    - WavefrontSize: [64]
    - WorkGroupMapping: [1]
    - WorkGroupMappingXCC: [2]
    - Groups:
      - - MatrixInstruction: [16, 16, 32, 1, 1, 3, 3, 2, 2]
          WorkGroup: [32, 8, 1]
          MIArchVgpr: 0
    BenchmarkJoinParameters:
    BenchmarkFinalParameters:
    - ProblemSizes:
      - Exact: [768, 3072, 1, 3840, 768, 768, 3840, 3840]
    - BiasTypeArgs: [0, 7]
LibraryLogic:
  ScheduleName: "gfx950"
  DeviceNames: ["Device 75a0"]
  ArchitectureName: "gfx950"
"""


@pytest.fixture
def mockLibLogicFile():
    with patch(
        "builtins.open", mock_open(read_data=VALID_LIBLOGIC_FILE_CONTENT)
    ) as mockFile:
        yield mockFile


def findAvailableArchs():
    availableArchs = []
    rocmpath = "/opt/rocm"
    if "ROCM_PATH" in os.environ:
        rocmpath = os.environ.get("ROCM_PATH")
    if "TENSILE_ROCM_PATH" in os.environ:
        rocmpath = os.environ.get("TENSILE_ROCM_PATH")
    rocmAgentEnum = os.path.join(rocmpath, "bin/rocm_agent_enumerator")
    if not os.path.exists(rocmAgentEnum):
        return availableArchs
    try:
        output = subprocess.check_output([rocmAgentEnum, "-t", "GPU"])
    except subprocess.CalledProcessError:
        return availableArchs
    lines = output.decode().splitlines()
    for line in lines:
        line = line.strip()
        if (not line in availableArchs) and (not "gfx000" in line):
            availableArchs.append(line)
    return availableArchs


@pytest.mark.skipif(
    "gfx950" not in findAvailableArchs(), reason="Requires gfx950 architecture"
)
def test_TensileLibLogicToYaml():
    solutionIndex = 0

    with tempfile.NamedTemporaryFile("w+", delete=False) as f:
        f.write(VALID_LIBLOGIC_FILE_CONTENT)
        f.flush()
        libLogicFileName = f.name

    with tempfile.TemporaryDirectory() as WORKSPACE:
        configYaml = os.path.join(WORKSPACE, "config.yaml")
        TensileLibLogicToYaml.TensileLibLogicToYaml(
            libLogicFileName, solutionIndex, configYaml, False
        )

        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            f.write(VALID_CONFIG_FILE_CONTENT)
            f.flush()
            configFileName = f.name

        assert filecmp.cmp(configYaml, configFileName, shallow=False)
