import math
from pathlib import Path
from inspect import currentframe, getframeinfo

MI_KEY: str = "MatrixInstruction"
MI_ENABLED_KEY: str = "EnableMatrixInstruction"


validMFMA = {}
validMFMA["H"] = [[32, 32, 4, 2], [32, 32, 8, 1], [16, 16, 4, 4], [16, 16, 16, 1], [4, 4, 4, 16]]
               + [[32, 32, 16, 1], [16, 16, 32, 1]]
validMFMA["S"] = [[32, 32, 1, 2], [32, 32, 2, 1], [16, 16, 1, 4], [16, 16, 4, 1], [4, 4, 1, 16]]
validMFMA["B"] = [[32, 32, 2, 2], [32, 32, 4, 1], [16, 16, 2, 4], [16, 16, 8, 1], [4, 4, 2, 16]]
               + [[32, 32, 16, 1], [16, 16, 32, 1]]
validMFMA["4xi8"] = [
    [32, 32, 4, 2],
    [32, 32, 8, 1],
    [16, 16, 4, 4],
    [16, 16, 16, 1],
    [4, 4, 4, 16],
    [32, 32, 16, 1],
    [16, 16, 32, 1],
]
validMFMA["D"] = [[16, 16, 4, 1], [4, 4, 4, 4]]
validMFMA["B1k"] = [[32, 32, 4, 2], [32, 32, 8, 1], [16, 16, 4, 4], [16, 16, 16, 1], [4, 4, 4, 16]]
validMFMA["C"] = validMFMA["S"]
validMFMA["Z"] = validMFMA["D"]
validMFMA["I8"] = [
    [32, 32, 4, 2],
    [32, 32, 8, 1],
    [16, 16, 4, 4],
    [16, 16, 16, 1],
    [4, 4, 4, 16],
] + [[32, 32, 16, 1], [16, 16, 32, 1]]
validMFMA["X"] = [[32, 32, 4, 1], [16, 16, 8, 1]]
validMFMA["F8"] = [[32, 32, 16, 1], [16, 16, 32, 1]] + [[32, 32, 64, 1], [16, 16, 128, 1]]
validMFMA["B8"] = validMFMA["F8"]
validMFMA["F8B8"] = validMFMA["F8"]
validMFMA["B8F8"] = validMFMA["F8"]
validMFMA["F8N"] = [[32, 32, 16, 1], [16, 16, 32, 1]]
validMFMA["B8N"] = validMFMA["F8N"]
validMFMA["F8B8N"] = validMFMA["F8N"]
validMFMA["B8F8N"] = validMFMA["F8N"]
validWMMA = [
    [16, 16, 16, 1],
]
validTT = 32
validMFMA["_format9"] = []

for MFMA in [
    validMFMA["H"],
    validMFMA["S"],
    validMFMA["B"],
    validMFMA["D"],
    validMFMA["X"],
    validMFMA["F8N"],
    validWMMA,
]:
    for MI in MFMA:
        for bm in range(int(math.log(MI[3], 2)) + 1):
            for tt0 in range(1, validTT + 1):
                for tt1 in range(1, validTT + 1):
                    for wave_m in range(3):
                        for wave_n in range(3):
                            validMFMA["_format9"].append(
                                [MI[0], MI[1], MI[2], MI[3], 2**bm, tt0, tt1, 2**wave_m, 2**wave_n]
                            )
validMatrixInstructions = (
    [[], [-1]]
    + validMFMA["H"]
    + validMFMA["S"]
    + validMFMA["B"]
    + validMFMA["D"]
    + validMFMA["B1k"]
    + validMFMA["X"]
)
validMatrixInstructions = validMatrixInstructions + validMFMA["_format9"]

validSMFMA = {}
validSMFMA["H"] = [[32, 32, 16, 1], [16, 16, 32, 1]] + [[32, 32, 32, 1], [16, 16, 64, 1]]
validSMFMA["B"] = [[32, 32, 16, 1], [16, 16, 32, 1]] + [[32, 32, 32, 1], [16, 16, 64, 1]]
validSMFMA["4xi8"] = [[32, 32, 32, 1], [16, 16, 64, 1]] + [[32, 32, 64, 1], [16, 16, 128, 1]]
validSMFMA["I8"] = validSMFMA["4xi8"]
validSMFMA["F8"] = [[32, 32, 32, 1], [16, 16, 64, 1]] + [[32, 32, 64, 1], [16, 16, 128, 1]]
validSMFMA["B8"] = validSMFMA["F8"]
validSMFMA["F8B8"] = validSMFMA["F8"]
validSMFMA["B8F8"] = validSMFMA["F8"]
validSMFMA["F8N"] = [[32, 32, 32, 1], [16, 16, 64, 1]]
validSMFMA["B8N"] = validSMFMA["F8N"]
validSMFMA["F8B8N"] = validSMFMA["F8N"]
validSMFMA["B8F8N"] = validSMFMA["F8N"]
validSMFMA["_format9"] = []
for SMFMA in [validSMFMA["H"], validSMFMA["B"], validSMFMA["4xi8"], validSMFMA["F8N"]]:
    for MI in SMFMA:
        for bm in range(int(math.log(MI[3], 2)) + 1):
            for tt0 in range(1, validTT + 1):
                for tt1 in range(1, validTT + 1):
                    for wave_m in range(3):
                        for wave_n in range(3):
                            validSMFMA["_format9"].append(
                                [MI[0], MI[1], MI[2], MI[3], 2**bm, tt0, tt1, 2**wave_m, 2**wave_n]
                            )
validSparseMatrixInstructions = validSMFMA["H"] + validSMFMA["B"] + validSMFMA["4xi8"]
validMatrixInstructions = (
    validMatrixInstructions + validSparseMatrixInstructions + validSMFMA["_format9"]
)


def elineno():
    """
    Return the file name and line number of the caller.
    """
    frame = getframeinfo(currentframe().f_back)
    return f"{Path(frame.filename).name}:{frame.lineno}"


def validateMatrixInstruction(solution: dict, filepath: Path, params: dict):
    """
    Validates the matrix instruction configured in the given solution.

    The function performs the following checks:
    - Ensures that the solution contains the required keys for matrix instruction support.
    - Ensures that the matrix instruction is not empty when it is enabled.
    - Validates that the matrix instruction is in the list of valid matrix instructions.
    - If the matrix instruction has 9 elements, it performs detailed validation checks:
        - Validates the work group dimensions.
        - Checks if the matrix instruction is supported by the assembler capabilities (MFMA or WMMA).
        - Validates the input per thread for sparse and non-sparse configurations.
        - Validates the matrix instruction block, wave group, and wave tile dimensions.
    - If the matrix instruction has 4 elements, it ensures that matrix instructions are enabled.
    - If the matrix instruction is empty, it ensures that matrix instructions are disabled.

    Args:
        solution: The solution to validate.
        filepath: The path to the file containing the solution.
        params: The global parameters for the solution.

    Raises:
        AssertionError: If any of the validation checks fail.
    """
    try:
        _validateMatrixInstruction(solution, params)
        return True
    except AssertionError as e:
        print(f"Validation failed: {filepath} (index {solution['SolutionIndex']})")
        print(f"Error: file: {e}")
        return False


def _validateMatrixInstruction(solution: dict, params: dict):
    """
    Function to validate the matrix instruction for the provided solution.
    See exported function for more details.
    """
    assert MI_KEY in solution, elineno()
    assert MI_ENABLED_KEY in solution, elineno()
    assert not (solution[MI_KEY] == [] and solution[MI_ENABLED_KEY] == True), elineno()

    isa = tuple(solution["ISA"])
    miFull = solution[MI_KEY]
    miEnabled = solution[MI_ENABLED_KEY]

    assert miFull in validMatrixInstructions, elineno()

    if len(solution[MI_KEY]) == 9:
        wfsize = solution["WavefrontSize"]
        mi = [miFull[0], miFull[1], miFull[2], miFull[3]]
        waves = miFull[7] * miFull[8]
        miwg0 = miFull[4] * miFull[0] * miFull[7]  # Matrix instruction work group 0
        miwg1 = waves * wfsize // miwg0

        isSparse = solution["ProblemType"]["Sparse"]
        miDataType = (
            solution["ProblemType"]["DataType"]
            if (not solution["EnableF32XdlMathOp"])
            else solution["ProblemType"]["F32XdlMathOp"]
        )
        miBlock = solution["MIBlock"]
        miWaveGroup = solution["MIWaveGroup"]
        miWaveTile = solution["MIWaveTile"]
        miInputPerThread = solution["MIInputPerThread"]
        miInputPerThreadA = solution["MIInputPerThreadA"]
        miInputPerThreadB = solution["MIInputPerThreadB"]
        miInutPerThreadMeta = solution["MIInputPerThreadMetadata"]

        # Check work group
        assert solution["WorkGroup"] == [miwg0, miwg1], elineno()

        # Check datatype
        if not isSparse:
            if params["AsmCaps"][isa]["HasMFMA"]:
                if not (miDataType.toChar() in validMFMA and mi in validMFMA[miDataType.toChar()]):
                    assert miDataType.isBFloat16() and mi in validMFMA["B1k"], elineno()
            elif params["AsmCaps"][isa]["HasWMMA"]:
                assert mi in validWMMA, elineno()
        else:
            assert miDataType.toChar() in validSMFMA and mi in validSMFMA[miDataType.toChar()], elineno()

        if (not params["AsmCaps"][isa]["HasMFMA"]) and params["AsmCaps"][isa]["HasWMMA"]:
            if isa[0] == 10 or isa[0] == 11:
                assert miInputPerThread == mi[2], elineno()

        assert solution["MFMA_BF16_1K"] == False, elineno()

        # Check MIBlock
        assert miBlock[0] == mi[0], elineno()
        assert miBlock[1] == mi[1], elineno()
        assert miBlock[2] == mi[2], elineno()
        assert miBlock[3] == mi[3], elineno()
        assert miBlock[4] == min(miwg0 // mi[0], mi[3]), elineno()
        assert miBlock[5] == mi[3] // miBlock[4], elineno()

        # Check MIWaveGroup
        assert miWaveGroup[0] == min((miwg0 // mi[0]) // miBlock[4], waves), elineno()
        assert miWaveGroup[1] == waves // miWaveGroup[0], elineno()

        # Check MIWaveTile
        assert miWaveTile[0] == mi[5], elineno()
        assert miWaveTile[1] == mi[6], elineno()

        # Check MIInputPerThread
        assert miInputPerThread == mi[0] * mi[2] * mi[3] // wfsize, elineno()

        # TODO: sparsity in hipBLASLt appears to be unused or always zero
        sparseA = not isSparse if isSparse != 2 else False
        sparseB = isSparse == 2 if isSparse else False
        assert miInputPerThreadA == miInputPerThread if not sparseA else miInputPerThread // 2, elineno()
        assert miInputPerThreadB == miInputPerThread if not sparseB else miInputPerThread // 2, elineno()
        assert miInutPerThreadMeta == miInputPerThread if not isSparse else miInputPerThread // 8, elineno()

        assert miEnabled == True, elineno()
    elif miFull != [] and len(miFull) == 4:
        assert miEnabled == True, elineno()
    else:
        assert miEnabled == False, elineno()
