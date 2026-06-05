################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

import sys
import math

from Tensile.Common.DataType import DataType
from rocisa.enum import DataTypeEnum

def getMiInputType(kernel: dict):
  """Select the effective MI operand type for MFMA latency lookup.

  Handles three cases based on kernel flags:
    1. EnableF32XdlMathOp + UseF32XEmulation → BFloat16 (BF16 emulation)
    2. EnableF32XdlMathOp only               → F32XdlMathOp (native XF32)
    3. Neither                               → MacDataTypeA (plain)

  UseF32XEmulation implies EnableF32XdlMathOp (guaranteed by Solution.__init__).

  Raises:
      KeyError: If EnableF32XdlMathOp or UseF32XEmulation is missing from the kernel dict.
  """
  if kernel["EnableF32XdlMathOp"]:
    if kernel["UseF32XEmulation"]:
      return DataType(DataTypeEnum.BFloat16)
    return kernel["ProblemType"]["F32XdlMathOp"]
  return kernel["ProblemType"]["DataType"]

def reject(state: dict, printSolutionRejectionReason: bool = True, *args) -> bool:
  """
  Reject a solution based on its internal state.

  Args:
      state: The state of the solution.
      printSolutionRejectionReason: If True, print the rejection reason.
      *args: Additional arguments to print if rejection occurs.

  Returns:
      True if the solution is rejected, False otherwise.
  """
  if state and "NoReject" in state and state["NoReject"]:
    return False
  if printSolutionRejectionReason:
    sys.stdout.write("\nreject: ")
    for a in args:
      print(a)
    #traceback.print_stack(None, 2)
    solutionIndex = state["SolutionIndex"] if (state != None and "SolutionIndex" in state) else -1
    if solutionIndex != -1:
      # If we have valid solutionIndex, this means we are during TensileCreateLibrary stage
      # In this stage, all solutions in the logic should be valid
      # So if any rejection happens, print the warning for further check
      # This will be done only when --global-parameters=PrintSolutionRejectionReason=True
      solutionNameMin = state["SolutionNameMin"] if ("SolutionNameMin" in state) else None
      # if we don't have SolutionNameMin, we simply use the problemTypeName
      solutionNameMin = str(state["ProblemType"]) if (solutionNameMin == None) else solutionNameMin
      raise Exception("!! Warning: Any rejection of a LibraryLogic is not expected, please check. \
        SolutionIndex: %d (or SolutionName/ProblemType: %s)"%(solutionIndex, solutionNameMin))
  if state != None:
    state["Valid"] = False
    return True

# print a labled variable
def pvar(state, field):
  return field + "=" + str(state[field])

def roundupRatio(dividend, divisor):
  return int(math.ceil(float(dividend) / float(divisor)))

def getRealDataTypeA(dataType):
    if dataType.value == DataTypeEnum.Float8BFloat8.value:
        return DataType(DataTypeEnum.Float8)
    elif dataType.value == DataTypeEnum.BFloat8Float8.value:
        return DataType(DataTypeEnum.BFloat8)
    elif dataType.value == DataTypeEnum.Float8BFloat8_fnuz.value:
        return DataType(DataTypeEnum.Float8_fnuz)
    elif dataType.value == DataTypeEnum.BFloat8Float8_fnuz.value:
        return DataType(DataTypeEnum.BFloat8_fnuz)
    else:
        return dataType

def getRealDataTypeB(dataType):
    if dataType.value == DataTypeEnum.Float8BFloat8.value:
        return DataType(DataTypeEnum.BFloat8)
    elif dataType.value == DataTypeEnum.BFloat8Float8.value:
        return DataType(DataTypeEnum.Float8)
    elif dataType.value == DataTypeEnum.Float8BFloat8_fnuz.value:
        return DataType(DataTypeEnum.BFloat8_fnuz)
    elif dataType.value == DataTypeEnum.BFloat8Float8_fnuz.value:
        return DataType(DataTypeEnum.Float8_fnuz)
    else:
        return dataType
