################################################################################
#
# Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

from .TensileInstructions import Module, SAddI32, SEndpgm
from .TensileInstructions.Instructions import *

from dataclasses import dataclass, field

#######################################
# Public functions
#######################################
@dataclass
class TensilePassOptions:
    removeDupActFunc:    bool = field(init=False)
    calculateMathClocks: bool = field(init=False)

def TensilePass(module, options: TensilePassOptions):
    if options.removeDupActFunc:
        _removeDuplicatedActivationFunctions(module)

def TensilePassGetCycles(module, options: TensilePassOptions, numWaves) -> int:
    if options.calculateMathClocks:
        return _calculateMathClocksInUnrolledLoop(module, numWaves)
    return -1

def getActivationFunctionModuleName(gwvw, sgpr, tmpVgpr, tmpSgpr):
    return "ActFunc_VW%d_Sgpr%d_Tmp%s_%s"%(gwvw, sgpr, tmpVgpr, tmpSgpr)

def getActivationBranchModuleName():
    return "InsertActFunctionCallAddrCalc"

#######################################
# Internal functions
#######################################
def _findActFunc(module) -> dict:
    modFunc = {}
    for item in module.items():
        if isinstance(item, Module):
            if "ActFunc_VW" in item.name:
                if item.name in modFunc:
                    modFunc[item.name].append(item)
                else:
                    modFunc[item.name] = [item]
            else:
                tmp = _findActFunc(item)
                for key, t in tmp.items():
                    if key in modFunc:
                        modFunc[key].extend(t)
                    else:
                        modFunc[key] = t
    return modFunc

def _replaceActBranchLabel(module, labels):
    for item in module.items():
        if isinstance(item, Module):
            if "InsertActFunctionCallAddrCalc" in item.name:
                labelFirst = labels[0]
                numUnderScores = labelFirst.count('_')
                partFirst  = labelFirst.rpartition("_")
                lastPostfix = partFirst[-1]
                labelLeft  = labels[1:]
                replaceLabel = False
                for inst in item.items():
                    if isinstance(inst, SAddI32) and inst.comment == "target branch offset":
                        if inst.srcs[0] in labelLeft:
                            replaceLabel = True
                            break
                if replaceLabel:
                    for idx in range(0, len(item.items())):
                        inst = item.getItem(idx)
                        if isinstance(inst, SAddI32) and inst.comment == "target branch offset":
                            # The label is generated in the format of XXXX_1, XXXX_2
                            # and string.rpartition returns ('XXXX', '_', '1').
                            # We only need the first string.
                            numUS = inst.srcs[0].count('_')
                            if numUnderScores == numUS:
                                part = inst.srcs[0].rpartition("_")
                                inst.setSrc(0, part[0] + "_" + lastPostfix)
                            elif numUnderScores == numUS - 1:
                                part = inst.srcs[0].rpartition("_")
                                inst.setSrc(0, part[0])
                            else:
                                assert 0, "Incorrect Activation Label"
                            
            else:
                _replaceActBranchLabel(item, labels)

def _removeDuplicatedActivationFunctions(module):
    modFunc = _findActFunc(module)
    moduleLast = Module("AddToLast")
    for _, mlist in modFunc.items():
        if len(mlist) > 1:
            labels = []
            for ml in mlist:
                labelName = ml.items()[0].items()[0].getLabelName()
                labels.append(labelName)
                ml.parent.removeItem(ml)
            # Avoid using deepcopy
            moduleLast.add(mlist[0])

            _replaceActBranchLabel(module, labels)
    if moduleLast.items():
        module.add(moduleLast)
        module.add(SEndpgm())

def _popInst(mod, moduleInst):
    for item in mod.items():
        if isinstance(item, Module):
            _popInst(item, moduleInst)
        elif isinstance(item, Instruction):
            moduleInst.add(item)

def _countCycles(item, numWaves):
    moduleInst = Module("Instructions to be issued")
    _popInst(item, moduleInst)

    cycles   = 0
    hwMFMA   = -99
    for item in moduleInst.items():
        if isinstance(item, Module):
            assert 0, "Module should be instructions here."
        elif isinstance(item, MFMAInstruction):
            if cycles - hwMFMA >= 3:
                cycles += 1
            else:
                cycles = hwMFMA + 4
            hwMFMA = cycles
        elif isinstance(item, BranchInstruction):
            cycles += 1
            # end of loop
            if "label_LoopBeginL" == item.labelName:
                break
        elif isinstance(item, Instruction):
            cycles += 1
        item.comment = "This is " + str(cycles) + "-cycle" # for debug
    return cycles

def _calculateMathClocksInUnrolledLoop(module, numWaves):
    # Kernel: openLoop -> loopBody -> noLoadLoop
    cycles     = -1
    isOpenLoop = False
    for item in module.items():
        # Find loopBody
        if item.name == "loopBody":
            cycles = _countCycles(item, numWaves)
            return cycles
    return -1