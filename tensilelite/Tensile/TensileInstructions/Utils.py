################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
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

from rocisa.enum import InstType
from rocisa.container import vgpr, sgpr, accvgpr, mgpr, Holder
from rocisa.instruction import Instruction, SWaitCnt

from .Code import Module
from .Containers import HolderContainer, RegisterContainer, RegName
from .DataType import DataType
from .Formatting import printAssert, printExit

from functools import lru_cache
from math import log
from typing import Tuple
import random
import string

########################################
# mfma
########################################

def dataTypeNameAbbrevToInstType(abbrev: str, sourceSwap: bool = False) -> InstType:
    if abbrev == 'f64':
        return InstType.INST_F64
    elif abbrev == 'f32':
        return InstType.INST_F32
    elif abbrev == 'f16':
        return InstType.INST_F16
    elif abbrev == 'i32':
        return InstType.INST_I32
    elif abbrev == 'i8':
        return InstType.INST_I8
    elif abbrev == 'bf16':
        return InstType.INST_BF16
    elif abbrev == 'xf32':
        return InstType.INST_XF32
    elif abbrev == 'fp8_fp8':
        return InstType.INST_F8
    elif abbrev == 'bf8_bf8':
        return InstType.INST_BF8
    elif (abbrev == 'fp8_bf8' and sourceSwap == False) or \
        (abbrev == 'bf8_fp8' and sourceSwap == True):
        return InstType.INST_F8_BF8
    elif (abbrev == 'bf8_fp8' and sourceSwap == False) or \
        (abbrev == 'fp8_bf8' and sourceSwap == True):
        return InstType.INST_BF8_F8
    else:
        assert("Unsupported data type.")
    return InstType.INST_NOTYPE

def dataTypeToMfmaInstTypePair(dataType: DataType, sourceSwap: bool) -> Tuple[InstType, InstType]:
    miInTypeStr  = dataType.toNameAbbrev()
    miInInstType = dataTypeNameAbbrevToInstType(miInTypeStr, sourceSwap) # v_mfma_[...xK]<InType>
    miOutInstType = dataTypeNameAbbrevToInstType(dataType.MIOutputTypeNameAbbrev()) # v_mfma_<OutType>..
    return miInInstType, miOutInstType

########################################
# Math
########################################

def log2(x):
    return int(log(x, 2) + 0.5)

def ceilDivide(numerator, denominator):
    # import pdb
    # pdb.set_trace()
    try:
        if numerator < 0 or denominator < 0:
            raise ValueError
    except ValueError:
        print("ERROR: Can't have a negative register value")
        return 0
    try:
        div = int((numerator+denominator-1) // denominator)
    except ZeroDivisionError:
        print("ERROR: Divide by 0")
        return 0
    return div

def roundUpToNearestMultiple(numerator, denominator):
    return ceilDivide(numerator,denominator)*int(denominator)

########################################
# Others
########################################

def replaceHolder(module, dst):
    if isinstance(module, Module):
        for item in module.items():
            replaceHolder(item, dst)
    elif isinstance(module, Instruction):
        for param in module.getParams():
            if isinstance(param, HolderContainer):
                param.setRegNum(dst)
                param = param.getCopiedRC()
    elif isinstance(module, SWaitCnt):
        assert(isinstance(dst, int))
        if isinstance(module.vmcnt, HolderContainer):
            module.vmcnt = module.vmcnt.holderIdx + dst
        if isinstance(module.lgkmcnt, HolderContainer):
            module.lgkmcnt = module.lgkmcnt.holderIdx + dst
        if isinstance(module.vscnt, HolderContainer):
            module.vscnt = module.vscnt.holderIdx + dst

    return module

