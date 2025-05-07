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
# SPDX-License-Identifier: MIT
################################################################################
import yaml
from pprint import pformat

from Tensile.Common.Architectures import SUPPORTED_ISA
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Common.Types import IsaVersion
from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.Toolchain.Validators import validateToolchain
from Tensile.SolutionStructs.Validators.MatrixInstruction import matrixInstructionToMIParameters, validateMIParameters
from Tensile.SolutionStructs.Validators.WorkGroup import validateWorkGroup

cxxCompiler = validateToolchain("amdclang++")
isaInfoMap = makeIsaInfoMap(SUPPORTED_ISA, cxxCompiler)

def test_convert_9_item_custom_kernel_config():
    input_conf = yaml.load(
"""
ProblemType:
    OperationType: GEMM
    DataTypeA: f8n
    DataTypeB: h
    UseScaleAB: "Scalar"
    DataType: h
    DestDataType: s
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: False
    TransposeB: False
    UseBias: 1
    Activation: True
    UseScaleAlphaVec: 1
    UseBeta: True
    Batched: True
    GroupedGemm:   True
    SupportUserArgs: True
MatrixInstruction: [32, 32, 8, 1, 1, 31, 16, 4, 2]
WavefrontSize: 48
WorkGroup: [16, 16, 1]
""", yaml.SafeLoader)

    # NOTE: This is not a valid MatrixInstruction, but since we are testing the conversion
    # functionality, it is useful to have unique values for each item in the list.
    mi = input_conf["MatrixInstruction"]
    wavefrontSize = input_conf["WavefrontSize"]
    workgroup = input_conf["WorkGroup"]
    ptype = input_conf["ProblemType"]
    isa = IsaVersion(9, 0, 10)

    outputConf = matrixInstructionToMIParameters(
        mi,
        isa,
        wavefrontSize,
        ptype,
        workgroup,
        isaInfoMap,
    )

    input = {
        "MatrixInstruction": input_conf["MatrixInstruction"],
    }

    print("inputConf: ", pformat(input))
    print("outputConf: ", pformat(outputConf))

    assert outputConf["MatrixInstruction"] == [32, 32, 8, 1]
    assert outputConf["EnableMatrixInstruction"] == True
    assert outputConf["MIBlock"] == [32, 32, 8, 1, 1, 1]
    assert outputConf["MIWaveGroup"] == [4, 2]
    assert outputConf["MIWaveTile"] == [31, 16]
    assert outputConf["MatrixInstBM"] == 1
    assert outputConf["MatrixInstBN"] == 1
    assert outputConf["MIInputPerThread"] == 5
    assert outputConf["MIInputPerThreadA"] == 5
    assert outputConf["MIInputPerThreadB"] == 5
    assert outputConf["MIInputPerThreadMetadata"] == 5
    assert outputConf["ThreadTile"] == [1, 1]
    assert outputConf["Sparse"] == 0
    assert outputConf["WorkGroup"] == [128, 3, 1]
    assert outputConf["WavefrontSize"] == 48
    assert outputConf["ISA"] == isa
    assert outputConf["EnableF32XdlMathOp"] == False
    assert outputConf["MFMA_BF16_1K"] == False

    solution = defaultSolution
    solution.update(input_conf)
    solution.update(outputConf)

    assert validateMIParameters(solution, isaInfoMap, True)
    assert validateWorkGroup(solution)

    mi4 = solution["MatrixInstruction"]
    bm = solution["MatrixInstBM"]
    tt0 = solution["MIWaveTile"][0]
    tt1 = solution["MIWaveTile"][1]
    wg0 = solution["MIWaveGroup"][0]
    wg1 = solution["MIWaveGroup"][1]
    format9 = [mi4[0], mi4[1], mi4[2], mi4[3], bm, tt0, tt1, wg0, wg1]

    assert format9 == mi


def testConvert9ItemCustomKernelConfig():

    inputConf = yaml.load(
"""
custom.config:
   ProblemType:
      OperationType: GEMM
      DataTypeA: f8n
      DataTypeB: h
      UseScaleAB: "Scalar"
      DataType: h
      DestDataType: s
      ComputeDataType: s
      HighPrecisionAccumulate: True
      TransposeA: False
      TransposeB: False
      UseBias: 1
      Activation: True
      UseScaleAlphaVec: 1
      UseBeta: True
      Batched: True
      GroupedGemm:   True
      SupportUserArgs: True
   MatrixInstruction: [32, 32, 8, 1, 5, 6, 7, 8, 9]
   1LDSBuffer: 1
   ScheduleIterAlg: 3
   DepthU: 32
   StaggerU: 0
   WorkGroupMapping: 8
   WaveSeparateGlobalReadA: 1
   WaveSeparateGlobalReadB: 1
   GlobalReadVectorWidthA: 4
   GlobalReadVectorWidthB: 2
   AssertFree0ElementMultiple: 4
   AssertSummationElementMultiple: 1
   NoReject: 1
   InternalSupportParams:
      KernArgsVersion: 0
      SupportUserGSU: False
      SupportCustomWGM: False
      SupportCustomStaggerU: False
      UseUniversalArgs: False
""", yaml.SafeLoader)

    inputConf = inputConf["custom.config"]

    isa = IsaVersion(9, 4, 2)
    wavefrontSize = 48
    workGroup = [4, 5, 6]

    outputConf = matrixInstructionToMIParameters(
        inputConf["MatrixInstruction"],
        isa,
        wavefrontSize,
        inputConf["ProblemType"],
        workGroup,
        isaInfoMap,
    )

    input = {
        "MatrixInstruction": inputConf["MatrixInstruction"],
    }

    print("inputConf: ", pformat(input))
    print("outputConf: ", pformat(outputConf))

    assert outputConf["MatrixInstruction"] == [32, 32, 8, 1]
    assert outputConf["EnableMatrixInstruction"] == True
    assert outputConf["MIBlock"] == [32, 32, 8, 1, 1, 1]
    assert outputConf["MIWaveGroup"] == [40, 1]
    assert outputConf["MIWaveTile"] == [6, 7]
    assert outputConf["MatrixInstBM"] == 1
    assert outputConf["MatrixInstBN"] == 1
    assert outputConf["MIInputPerThread"] == 5
    assert outputConf["MIInputPerThreadA"] == 5
    assert outputConf["MIInputPerThreadB"] == 5
    assert outputConf["MIInputPerThreadMetadata"] == 5
    assert outputConf["ThreadTile"] == [1, 1]
    assert outputConf["Sparse"] == 0
    assert outputConf["WorkGroup"] == [1280, 2, 6]  # Why do we change the workgroup here?
    assert outputConf["WavefrontSize"] == 48
    assert outputConf["ISA"] == isa
    assert outputConf["EnableF32XdlMathOp"] == False
    assert outputConf["MFMA_BF16_1K"] == False

    solution = defaultSolution
    solution.update(inputConf)
    solution.update(outputConf)

    assert validateMIParameters(solution, isaInfoMap, True) == True
