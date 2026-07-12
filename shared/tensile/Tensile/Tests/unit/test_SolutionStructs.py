################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

from Tensile.Common import globalParameters
from Tensile.DataType import DataType
from Tensile.SolutionStructs import Solution, disablePreloadKernelArguments


def test_disable_preload_kernel_arguments_clears_delay():
    state = {"PreloadKernelArguments": 1, "DelayRemainingArguments": True}

    disablePreloadKernelArguments(state)

    assert state["PreloadKernelArguments"] == 0
    assert state["DelayRemainingArguments"] is False


def test_gfx12_wmma_mi_input_per_thread_uses_generic_formula():
    isa = (12, 0, 1)
    oldAsmCaps = globalParameters.get("AsmCaps")
    globalParameters["AsmCaps"] = {isa: {"HasMFMA": False, "HasWMMA": True}}
    state = {
        "ISA": isa,
        "MatrixInstruction": [16, 16, 16, 1, 1, 4, 4, 2, 2],
        "ThreadTile": [0, 0],
        "WorkGroup": [0, 0, 1],
        "WavefrontSize": 32,
        "ProblemType": {"DataType": DataType("H")},
        "EnableF32XdlMathOp": False,
    }

    try:
        Solution.parameterWrapper(state)
    finally:
        if oldAsmCaps is None:
            del globalParameters["AsmCaps"]
        else:
            globalParameters["AsmCaps"] = oldAsmCaps

    assert state["MIInputPerThread"] == state["MatrixInstruction"][0] * state["MatrixInstruction"][2] * state["MatrixInstruction"][3] // state["WavefrontSize"]


def test_gfx11_wmma_mi_input_per_thread_keeps_full_k():
    isa = (11, 0, 0)
    oldAsmCaps = globalParameters.get("AsmCaps")
    globalParameters["AsmCaps"] = {isa: {"HasMFMA": False, "HasWMMA": True}}
    state = {
        "ISA": isa,
        "MatrixInstruction": [16, 16, 16, 1, 1, 4, 4, 2, 2],
        "ThreadTile": [0, 0],
        "WorkGroup": [0, 0, 1],
        "WavefrontSize": 32,
        "ProblemType": {"DataType": DataType("H")},
        "EnableF32XdlMathOp": False,
    }

    try:
        Solution.parameterWrapper(state)
    finally:
        if oldAsmCaps is None:
            del globalParameters["AsmCaps"]
        else:
            globalParameters["AsmCaps"] = oldAsmCaps

    assert state["MIInputPerThread"] == 16
