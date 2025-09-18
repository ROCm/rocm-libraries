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


from pprint import pformat
from pathlib import Path
from typing import Dict, Tuple

from Tensile.Common import print1, IsaVersion, IsaInfo
from Tensile.SolutionStructs.Validators.MatrixInstruction import matrixInstructionToMIParameters

from Tensile.CustomKernels import isCustomKernelConfig, getCustomKernelConfig
from Tensile import CUSTOM_KERNEL_PATH


def handleCustomKernel(sol: dict, isaInfoMap: Dict[IsaVersion, IsaInfo]) -> Tuple[dict, bool]:
    """
    Process custom kernel configuration for a given solution.

    Args:
        sol: Dictionary containing the solution.
        isaInfoMap: Dictionary mapping IsaVersion to IsaInfo.

    Returns:
        A tuple containing the updated solution dictionary and a boolean indicating
        whether the solution uses a custom kernel.
    """
    if not isCustomKernelConfig(sol):
        return sol, False

    name = sol["CustomKernelName"]
    dir = CUSTOM_KERNEL_PATH
    config = getCustomKernelConfig(name, {}, dir)
    sol.update(config)

    mi = sol["MatrixInstruction"]
    print1(f">>   Found custom kernel: {name} with MI {mi}")

    if not (len(mi) == 4 or len(mi) == 0):
        print1(
            f">>     Error: Custom kernels in logic files should have 'MatrixInstruction' of length 4 or 0, not length {len(mi)}"
        )
        mi = sol["MatrixInstruction"]
        isa = next(iter(isaInfoMap.keys()))
        wavefrontSize = sol["WavefrontSize"]
        ptype = sol["ProblemType"]
        workgroup = sol.get("WorkGroup", None)
        miParams = matrixInstructionToMIParameters(
            mi, isa, wavefrontSize, ptype, workgroup, isaInfoMap
        )
        print1(
            f">>     Hint: Replace 'MatrixInstruction' in {name}.s with following diff:\n"
            f"{prepareCustomKernelConfig(miParams, mi)}"
        )
    return sol, True


def hasCustomKernel(file: Path) -> bool:
    """
    Check if the given logic file contains at least one custom kernel.

    Args:
        file: Path to a logic file.

    Returns:
        True if the logic file contains at least one custom kernel.
    """
    with open(file, "r") as f:
        for line in f:
            l = line.strip()
            if l.startswith("CustomKernelName:") and not l.endswith("''"):
                return True
    return False


def prepareCustomKernelConfig(miParams: dict, prevMi: list) -> str:
    """
    Formats fields on custom kernel configuration to be used by developers to update
    prototype custom kernel configuration for production serialized logic files.

    Args:
        miParams: Dictionary of matrix instruction parameters.

    Returns:
        A string containing formatted fields for the custom kernel configuration.
    """
    pformatStr = pformat(miParams)
    # Remove the dictionary braces and split into lines
    lines = pformatStr.strip("{}").split("\n")
    formattedLines = []
    for line in lines:
        line = line.strip()
        # Remove the 'ISA' line as it is not needed in the output
        if line.startswith("'ISA'"):
            continue
        line = line.replace("'", "")
        line = "+   " + line
        formattedLines.append(line[:-1] if line.endswith(",") else line)
    result = "\n".join(formattedLines)

    result = f"-   MatrixInstruction: {prevMi}\n" + result
    return result
