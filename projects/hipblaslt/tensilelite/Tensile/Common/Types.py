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

from dataclasses import dataclass
from typing import NamedTuple

@dataclass
class IsaInfo:
    asmCaps: dict
    archCaps: dict
    regCaps: dict
    asmBugs: dict


class SemanticVersion(NamedTuple):
    major: int
    minor: int
    patch: int

IsaVersion = SemanticVersion

class DebugConfig(NamedTuple):
    """
    Members:
        debugKernel: assembly only, kernel gets buffer for debug "printing";
                     kernel writes data to memory, gets coppied to host and printed.
        forceGenerateKernel: Even if error occurs in kernel generation (i.e. due to resource overflow),
                             generate the kernel source anyway. Tensile will also attempt to run
                             the kernel. Useful to examine and debug overflow errors.
        printSolutionRejectionReason: Print why a solution is marked as invalid.
        printIndexAssignmentInfo: Print the tensor index assignment info.

    """
    enableAsserts: bool=False
    enableDebugA: bool=False
    enableDebugB: bool=False
    enableDebugC: bool=False
    expectedValueC: float=16.0
    forceCExpectedValue: bool=False
    debugKernel: bool=False
    forceGenerateKernel: bool=False
    printSolutionRejectionReason: bool=False
    splitGSU: bool=False
    printIndexAssignmentInfo: bool=False


def makeDebugConfig(config: dict) -> DebugConfig:

    enableAsserts = False
    enableDebugA = False
    enableDebugB = False
    enableDebugC = False
    expectedValueC = 16.0
    forceCExpectedValue = False
    debugKernel = False
    forceGenerateKernel = False
    printSolutionRejectionReason = False
    splitGSU = False
    printIndexAssignmentInfo = False

    if "EnableAsserts" in config:
        enableAsserts = config["EnableAsserts"]
    if "EnableDebugA" in config:
        enableDebugA = config["EnableDebugA"]
    if "EnableDebugB" in config:
        enableDebugB = config["EnableDebugB"]
    if "EnableDebugC" in config:
        enableDebugC = config["EnableDebugC"]
    if "ExpectedValueC" in config:
        expectedValueC = config["ExpectedValueC"]
    if "ForceCExpectedValue" in config:
        forceCExpectedValue = config["ForceCExpectedValue"]
    if "DebugKernel" in config:
        debugKernel = config["DebugKernel"]
    if "ForceGenerateKernel" in config:
        forceGenerateKernel = config["ForceGenerateKernel"]
    if "PrintSolutionRejectionReason" in config:
        printSolutionRejectionReason = config["PrintSolutionRejectionReason"]
    if "SplitGSU" in config:
        splitGSU = config["SplitGSU"]
    if "PrintIndexAssignmentInfo" in config:
        printIndexAssignmentInfo = config["PrintIndexAssignmentInfo"]

    return DebugConfig(
               enableAsserts,
               enableDebugA,
               enableDebugB,
               enableDebugC,
               expectedValueC,
               forceCExpectedValue,
               debugKernel,
               forceGenerateKernel,
               printSolutionRejectionReason,
               splitGSU,
               printIndexAssignmentInfo,
            )
