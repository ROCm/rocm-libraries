################################################################################
#
# Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

from .Base import Item
from .Code import KernelBody, Label, Macro, Module, RegSet, TextBlock
from .Containers import RegisterContainer
from .Instructions import BranchInstruction, CommonInstruction, Instruction, \
                          CompositeInstruction, MacroInstruction, \
                          ReadWriteInstruction, SEndpgm, SMovB32, \
                          _SWaitCnt, _SWaitCntVscnt, SSleep, SBarrier, SNop
from .Formatting import slash50

from dataclasses import dataclass, field

@dataclass
class TensileInstructionsPassOptions:
    removeDupAssign: bool = True

    def doOpt(self) -> bool:
        return self.removeDupAssign

def TensileInstructionsPass(kernelBody: KernelBody, \
    options: TensileInstructionsPassOptions):
    module = kernelBody.body
    assignDict = getAssignmentDict(module)
    compositeToInstruction(module)
    if options.doOpt():
        maxVgpr = kernelBody.totalVgprs
        maxSgpr = kernelBody.totalSgprs
        graph = buildGraph(module, maxVgpr, maxSgpr, assignDict)
        if options.removeDupAssign:
            removeDuplicateAssignment(graph)

#######################################
# Setup
#######################################

def compositeToInstruction(module):
    itemList = []
    for item in module.items():
        if isinstance(item, CompositeInstruction):
            items = item.getInstructions()
            itemList.extend(items)
            continue  # Skip appending composite instruction back to list
        elif isinstance(item, Module) or isinstance(item, Macro):
            compositeToInstruction(item)
        itemList.append(item)
    module.setItems(itemList)

def getAssignmentDict(module):
    assignmentDict = dict()
    _getAssignmentDictIter(module, assignmentDict)
    return assignmentDict

def buildGraph(module, vgprMax, sgprMax, assignmentDict):
    graph = dict()
    graph["v"] = [[] for _ in range(vgprMax)]
    graph["s"] = [[] for _ in range(sgprMax)]
    graph["m"] = [[] for _ in range(1)]
    _recordGraph(module, graph, assignmentDict)
    return graph

def removeDuplicateAssignment(graph):
    _removeDuplicateAssignmentGPR(graph, "s")

# No opt item container class
class NoOptItem:
    def __init__(self, item) -> None:
        self._item = item

    def getItem(self) -> Item:
        return self._item

################################################################################
################################################################################
###
###   Internal Functions
###
################################################################################
################################################################################

def _getAssignmentDictIter(module, assignmentDict):
    for item in module.items():
        if isinstance(item, Module):
            _getAssignmentDictIter(item, assignmentDict)
        elif isinstance(item, RegSet):
            num = 0
            if item.ref:
                num = assignmentDict[item.ref] + item.offset
            else:
                num = item.value
            assignmentDict[item.name] = num

def _addRegToGraph(item, assignmentDict, params: list, graph, noOpt):
    for p in params:
        if isinstance(p, RegisterContainer):
            _setName2RegNum(p, assignmentDict)
            if p.regType == "acc":
                continue
            for i in range(p.regIdx, p.regIdx + p.regNum):
                if graph[p.regType][i] and graph[p.regType][i][-1] == item:
                    continue
                # print("[%s] Index %d %d" %(p.regType, i, len(graph[p.regType])))
                if noOpt:
                    graph[p.regType][i].append(NoOptItem(item))
                else:
                    graph[p.regType][i].append(item)

def _recordGraph(module, graph, assignmentDict):
    for item in module.items():
        if isinstance(item, Module):
            _recordGraph(item, graph, assignmentDict)
        elif isinstance(item, (CommonInstruction, ReadWriteInstruction, MacroInstruction)):
            _addRegToGraph(item, assignmentDict, item.getParams(), graph, module.isNoOpt())
        elif isinstance(item, (BranchInstruction, Label, _SWaitCnt, _SWaitCntVscnt, \
                            SEndpgm, SBarrier, SNop, SSleep)):
            for i in range(len(graph["v"])):
                graph["v"][i].append(item)
            for i in range(len(graph["s"])):
                graph["s"][i].append(item)

# Currently only removes s_mov_b32, does not support 2 sgpr at lvalue
def _removeDuplicateAssignmentGPR(graph, regType):
    for idx, sList in enumerate(graph[regType]):
        assignValue = None
        newList = []
        for item in sList:
            isRemoved = False
            if isinstance(item, (NoOptItem, BranchInstruction, Label, MacroInstruction, \
                                _SWaitCnt, _SWaitCntVscnt)):
               assignValue = None
            # FIXME: Need refactor.
            elif isinstance(item, SMovB32):
                gpr      = item.dst
                gprValue = item.srcs
                if str(item.dst) == "exec_lo":
                    continue
                if gpr.regIdx == idx and gprValue == assignValue:
                    if item.comment:
                        comment = item.comment + " (dup assign opt.)"
                        newItem = TextBlock(slash50(comment))
                        module  = item.parent
                        module.replaceItem(item, newItem) # type: ignore
                    else:
                        module = item.parent
                        module.removeItem(item) # type: ignore
                    isRemoved = True
                assignValue = gprValue
            elif isinstance(item, Instruction):
                params = item.getParams()
                if len(params) > 1:
                    gpr = params[0]
                    if isinstance(gpr, RegisterContainer) and (gpr.regType == regType):
                        for i in range(gpr.regIdx, gpr.regIdx + gpr.regNum):
                            if i == idx:
                                assignValue = None
                                break
            if not isRemoved:
                newList.append(item)

        if len(newList) != len(sList):
            graph["s"][idx] = newList

################################################################################
################################################################################
###
###   Helper Functions
###
################################################################################
################################################################################

# Find ".set AAAAA 0" and convert "s[AAAAA]" into "s0"
def _setName2RegNum(gpr, assignmentDict):
    assert(isinstance(gpr, RegisterContainer))
    if gpr.regIdx == None and gpr.regName:
        name = gpr.getRegNameWithType()
        num = assignmentDict[name] + gpr.regName.getTotalOffsets()
        gpr.regIdx = num
    RegNumList = []
    for i in range(0, gpr.regNum):
        RegNumList.append(i + gpr.regIdx)
    return RegNumList

def _graphDebugSaveToTxt(graph, kernelName):
    f = open('%s.txt' % kernelName, 'w')
    f.write("VGPR\n")
    i = 0
    for d in graph["v"]:
        f.write("[%d]\n" % i)
        for dd in d:
            ss = str(dd)
            f.write(ss)
        f.write("\n")
        i += 1
    i = 0
    f.write("SGPR\n")
    for d in graph["s"]:
        f.write("[%d]\n" % i)
        for dd in d:
            ss = str(dd)
            f.write(ss)
        f.write("\n")
        i += 1
    f.close()
