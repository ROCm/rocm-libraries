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

import pprint
from typing import Dict, Optional

from Tensile.Common import IsaVersion, IsaInfo, print2, elineno
from Tensile.Common.Architectures import SUPPORTED_ISA
from Tensile.Common.DataType import DataType
from Tensile.Common.ValidParameters import makeValidMatrixInstructions, makeValidMFMA, makeValidSMFMA, makeValidWMMA

from ..Utilities import reject

MI_KEY: str = "MatrixInstruction"
MI_ENABLED_KEY: str = "EnableMatrixInstruction"

def matrixInstructionToMIParameters(
      mi: list,
      isa: IsaVersion,
      wavefrontSize: int,
      problemType: dict,
      workGroup: Optional[list],
      isaInfoMap: Dict[IsaVersion, IsaInfo]
    ):
    """
    Converts a 9-item matrix instruction into the associated 4-item representation and
    populates supporting MI parameters.

    Args:
        mi: The matrix instruction to convert. Must have length 9.
        isa: The ISA tuple.
        wavefrontSize: The wavefront size. Typically "WavefrontSize" in a solution.
        problemType: The problem type dictionary. Typically "ProblemType" in a solution.
    """
    print2(f">> Converting MatrixInstruction {mi} to MI parameters")

    if len(mi) != 9:
      raise ValueError(f"MatrixInstruction must be 9 items long to convert into MI"
                       f" Parameters, found {mi} with length {len(mi)}")

    result = {}
    result["ISA"] = isa

    # Enable F32 XDL math operation only when the input type is f32.
    enableF32xdl = (
      "F32XdlMathOp" in problemType
      and not problemType["F32XdlMathOp"].isSingle()
      and problemType["DataType"].isSingle()
    )
    result["EnableF32XdlMathOp"] = enableF32xdl

    mi4  = [mi[0], mi[1], mi[2], mi[3]]
    result["MatrixInstruction"] = mi4
    result["EnableMatrixInstruction"] = True
    result["MatrixInstM"] = mi[0]
    result["MatrixInstN"] = mi[1]
    result["MatrixInstK"] = mi[2]
    result["MatrixInstB"] = mi[3]

    waves = mi[7]* mi[8]
    wg0 = mi[4] * mi[0] * mi[7]

    result["WavefrontSize"] = wavefrontSize
    if workGroup:
      # NOTE: Typically, the WorkGroup is set to the default [16, 16, 1] for a
      # length-9 matrix instruction. However, some custom kernel solutions used
      # during benchmarking don't have WorkGroup set at all.
      result["WorkGroup"] = [wg0, waves*wavefrontSize//wg0, workGroup[2]]
    result["ThreadTile"] = [1, 1]  # dummy

    isSparse = problemType.get("Sparse", 0)
    miDataType = DataType(
        problemType["DataType"]
        if not enableF32xdl
        else problemType["F32XdlMathOp"]
    )

    validMFMA = makeValidMFMA()
    result["MFMA_BF16_1K"] = (
        not isSparse
        and isaInfoMap[isa].asmCaps["HasMFMA"]
        and not (miDataType.toChar() in validMFMA and mi4 in validMFMA[miDataType.toChar()])
        and miDataType.isBFloat16()
        and mi4 in validMFMA["B1k"]
    )

    # set MIBlock
    MIBlockBM = wg0 // mi[0]
    MIBlockBM = min(MIBlockBM, mi[3])
    MIBlockBN = mi[3] // MIBlockBM
    result["MatrixInstBM"] = MIBlockBM
    result["MatrixInstBN"] = MIBlockBN
    result["MIBlock"]    = [mi[0], mi[1], mi[2], mi[3], MIBlockBM, MIBlockBN]

    # set MIWaveGroup
    miwg0 = min((wg0 // mi[0]) // MIBlockBM, waves)
    result['MIWaveGroup'] = [miwg0, waves // miwg0]

    # set MIWaveTile
    result['MIWaveTile'] = [mi[5], mi[6]]

    # set MIInputPerThread
    hasMFMA = isaInfoMap[isa].asmCaps["HasMFMA"]
    hasWMMA = isaInfoMap[isa].asmCaps["HasWMMA"]

    result['MIInputPerThread'] = mi[0] * mi[2] * mi[3] // wavefrontSize
    if (not hasMFMA) and hasWMMA and (isa[0] == 10 or isa[0] == 11):
      result['MIInputPerThread'] = mi[2]

    sparseA = False if not isSparse else False if isSparse == 2 else True
    sparseB = False if not isSparse else True if isSparse == 2 else False
    result['MIInputPerThreadA'] = result['MIInputPerThread'] if not sparseA else result['MIInputPerThread'] // 2
    result['MIInputPerThreadB'] = result['MIInputPerThread'] if not sparseB else result['MIInputPerThread'] // 2
    result['MIInputPerThreadMetadata'] = result['MIInputPerThread'] if not isSparse else result['MIInputPerThread'] // 8

    print2(f">> MI Parameters: {pprint.pformat(result)}")
    return result


def validateMIParameters(
    solution: dict, isaInfoMap: Dict[IsaVersion, IsaInfo], printSolutionRejectionReason: bool = True
):
    """
    Validates matrix instruction (MI) related parameters in the given solution.

    This function performs the following checks:
    - Ensures that "MatrixInstruction" is of length 4.
    - Ensures that the solution contains the required keys for matrix instruction support.
    - Ensures that the matrix instruction is not empty when it is enabled.
    - Validates that the matrix instruction is in the list of valid matrix instructions.
    - Validates the work group dimensions.
    - Checks if the matrix instruction is supported by the assembler capabilities (MFMA or WMMA).
    - Validates the input per thread for sparse and non-sparse configurations.
    - Validates the matrix instruction block, wave group, and wave tile dimensions.
    - If the matrix instruction has 4 elements, it ensures that matrix instructions are enabled.
    - If the matrix instruction is empty, it ensures that matrix instructions are disabled.

    Args:
        solution: The solution to validate.
        isaInfoMap: The map of ISA versions to assembler and architecture capabilities.
        filepath: The path to the file containing the solution.

    Raises:
        AssertionError: If any of the validation checks fail.
    """
    validMatrixInstructions = makeValidMatrixInstructions()
    validMFMA = makeValidMFMA()
    validSMFMA = makeValidSMFMA()
    validWMMA = makeValidWMMA()

    assert MI_KEY in solution, elineno() + ": missing MatrixInstruction"
    assert MI_ENABLED_KEY in solution, elineno() + ": missing EnableMatrixInstruction"
    assert not (solution[MI_KEY] == [] and solution[MI_ENABLED_KEY] == True), (
        elineno() + ": MI empty but enabled"
    )

    isa = IsaVersion(*solution["ISA"])
    assert isa in SUPPORTED_ISA, elineno() + ": Unsupported ISA: " + str(isa)
    # TODO: Temporary until all 940/941 ISAs are removed
    if (9, 4, 0) <= isa <= (9, 4, 1):
        isa = (9, 4, 2)

    ptype = solution["ProblemType"]
    isSparse = ptype.get("Sparse", 0)
    miDataType = DataType(
        ptype["DataType"]
        if not solution.get("EnableF32XdlMathOp", False)
        else ptype["F32XdlMathOp"]
    )

    mi4 = solution[MI_KEY]
    miEnabled = solution[MI_ENABLED_KEY]
    assert len(mi4) == 4 or len(mi4) == 0, elineno() + ": MI length not 4 or 0"
    if len(mi4) == 0:
        assert miEnabled == False, elineno()
        return True

    assert solution["MatrixInstM"] == mi4[0]
    assert solution["MatrixInstN"] == mi4[1]
    assert solution["MatrixInstK"] == mi4[2]
    assert solution["MatrixInstB"] == mi4[3]

    assert mi4 in validMatrixInstructions, f"{elineno()} : invalid MI4: {str(mi4)} for type {miDataType.toChar()}"

    mi9 = [mi4[0], mi4[1], mi4[2], mi4[3]]
    assert "MatrixInstBM" in solution, elineno() + ": missing MatrixInstBM"
    mi9.append(solution["MatrixInstBM"])
    assert "MIWaveTile" in solution, elineno() + ": missing MIWaveTile"
    mi9.extend(solution["MIWaveTile"])
    assert "MIWaveGroup" in solution, elineno() + ": missing MIWaveGroup"
    mi9.extend(solution["MIWaveGroup"])

    assert len(mi4) == 4 and len(mi9) == 9, elineno() + " MI4: " + str(mi4) + " MI9: " + str(mi9)

    if not miEnabled:
        return False


    wfsize = solution["WavefrontSize"]
    waves = solution["MIWaveGroup"][0] * solution["MIWaveGroup"][1]
    wg0 = mi9[4] * mi9[0] * mi9[7]  # Work group 0

    hasMFMA = isaInfoMap[isa].asmCaps["HasMFMA"]
    hasWMMA = isaInfoMap[isa].asmCaps["HasWMMA"]

    miBlock = solution["MIBlock"]
    miWaveGroup = solution["MIWaveGroup"]
    miWaveTile = solution["MIWaveTile"]

    if not isSparse:
        if hasMFMA:
            if not (miDataType.toChar() in validMFMA and mi4 in validMFMA[miDataType.toChar()]):
                if miDataType.isBFloat16() and mi4 in validMFMA["B1k"]:  # but is valid bf16 MFMA
                    assert solution["MFMA_BF16_1K"], elineno()
                else:
                    return not reject(
                        solution,
                        printSolutionRejectionReason,
                        f"Invalid MFMA configuration: {solution}",
                    )
        elif hasWMMA and (not mi4 in validWMMA):
            return not reject(
                solution, printSolutionRejectionReason, f"Invalid WMMA configuration: {solution}"
            )
    else:
        if not (miDataType.toChar() in validSMFMA and mi4 in validSMFMA[miDataType.toChar()]):
            return not reject(
                solution, printSolutionRejectionReason, f"Invalid SMFMA configuration: {solution}"
            )

    # Check MIBlock
    assert miBlock[0] == mi4[0], elineno()
    assert miBlock[1] == mi4[1], elineno()
    assert miBlock[2] == mi4[2], elineno()
    assert miBlock[3] == mi4[3], elineno()
    assert miBlock[4] == min(wg0 // mi4[0], mi4[3]), elineno()
    assert miBlock[5] == mi4[3] // miBlock[4], elineno()

    # Check MIWaveGroup
    assert miWaveGroup[0] == min((wg0 // mi4[0]) // miBlock[4], waves), elineno()
    assert miWaveGroup[1] == waves // miWaveGroup[0], elineno()

    # Check MIWaveTile
    assert miWaveTile[0] == mi9[5], elineno()
    assert miWaveTile[1] == mi9[6], elineno()

    # Check MIInputPerThread
    miInputPerThread = solution["MIInputPerThread"]

    if (not hasMFMA) and hasWMMA:
        if isa[0] == 10 or isa[0] == 11:
            assert miInputPerThread == mi4[2], elineno()

    # If Navi architecture, the input per thread is different
    if IsaVersion(10, 0, 0) <= isa <= IsaVersion(11, 5, 1):
        assert miInputPerThread == mi4[2], elineno()
    else:
        assert miInputPerThread == mi4[0] * mi4[2] * mi4[3] // wfsize, f"{elineno()} MIInputPerThread: {miInputPerThread} != {mi4[0]} * {mi4[2]} * {mi4[3]} / {wfsize} = {mi4[0] * mi4[2] * mi4[3] // wfsize}"

    if "MIInputPerThreadA" in solution:
        miInputPerThreadA = solution["MIInputPerThreadA"]
        sparseA = False if not isSparse else False if isSparse == 2 else True
        assert miInputPerThreadA == miInputPerThread if not sparseA else miInputPerThread // 2, elineno()

    if "MIInputPerThreadB" in solution:
        miInputPerThreadB = solution["MIInputPerThreadB"]
        sparseB = False if not isSparse else True if isSparse == 2 else False
        assert miInputPerThreadB == miInputPerThread if not sparseB else miInputPerThread // 2, elineno()

    if "MIInputPerThreadMetadata" in solution:
        miInutPerThreadMeta = solution["MIInputPerThreadMetadata"]
        assert miInutPerThreadMeta == miInputPerThread if not isSparse else miInputPerThread // 8, elineno()
    return True
