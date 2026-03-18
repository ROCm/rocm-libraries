################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

import yaml
import os
import sys
import shutil
import argparse
from copy import deepcopy

from Tensile import __version__
from Tensile.SolutionStructs.Naming import getSolutionNameMin
from Tensile.SolutionStructs.Naming import getKernelNameMin
from Tensile.SolutionStructs.Problem import ProblemType, problemTypeToEnum
from Tensile.Common import ParallelMap2

verbosity = 1

def ensurePath(path):
  if not os.path.exists(path):
    os.makedirs(path)
  return path

def allFiles(startDir):
    current = os.listdir(startDir)
    files = []
    for filename in [_current for _current in current if os.path.splitext(_current)[-1].lower() == '.yaml']:
        fullPath = os.path.join(startDir,filename)
        if os.path.isdir(fullPath):
            files = files + allFiles(fullPath)
        else:
            files.append(fullPath)
    return files

def fixSizeInconsistencies(sizes, fileType):
    origNumSizes = len(sizes)
    sizesDict = dict()
    for size, index in sizes:
        size = size[:-4] if len(size) >= 8 else size
        sizesDict[(value for value in size)] = [size, index]
    newSizes = list()
    for value in sizesDict.values():
        newSizes.append(value)
    numSize = len(newSizes)
    if numSize - origNumSizes > 0:
        verbose(numSize - origNumSizes, "duplicate size(s) removed from", fileType, "logic file")
    return newSizes, len(newSizes)

def addKernel(solutionPool, solDict, solution):
    if solution["SolutionNameMin"] in solDict:
        index = solDict[solution["SolutionNameMin"]]["SolutionIndex"]
        debug("...Reuse previously existed solution", end="")
    else:
        index = len(solutionPool)
        _solution = deepcopy(solution) # if we don't we will see some subtle errors
        _solution["SolutionIndex"] = index
        solutionPool.append(_solution)
        solDict[solution["SolutionNameMin"]] = _solution
        debug("...A new solution has been added", end="")
    debug("({}) {}".format(index, solutionPool[index]["SolutionNameMin"] if "SolutionNameMin" in solutionPool[index] else "(SolutionName N/A)"))
    return solutionPool, solDict, index

# update dependant parameters if StaggerU == 0
def sanitizeSolutions(solList):
    for sol in solList:
        if sol.get("StaggerU") == 0:
            sol["StaggerUMapping"] = 0
            sol["StaggerUStride"] = 0
            sol["_staggerStrideShift"] = 0

from Tensile.Common.GlobalParameters import defaultSolution, defaultInternalSupportParams
from Tensile.Common import assignParameterWithDefault

def reNameSolutions(data):
    solList = data[5]
    for sol in solList:
        # Assign solution state from config, filling missing from the defaultSolution
        for key in defaultSolution:
            assignParameterWithDefault(sol, key, sol, defaultSolution)
        sol["ProblemType"] = data[4]
        sol["SolutionNameMin"] = getSolutionNameMin(sol,splitGSU=False)
        sol["KernelNameMin"] = getKernelNameMin(sol,splitGSU=False)
        del sol["ProblemType"]

def removeUnusedSolutions(oriData, prefix=""):
    origNumSolutions = len(oriData[5])

    kernelsInUse = [ index for _, [index, _] in oriData[7] ]
    for i, solution in enumerate(oriData[5]):
        solutionIndex = solution["SolutionIndex"]
        oriData[5][i]["__InUse__"] = True if solutionIndex in kernelsInUse else False

    # debug prints
    for o in [o for o in oriData[5] if o["__InUse__"]==False]:
        debug("{}Solution ({}) {} is unused".format(
            prefix,
            o["SolutionIndex"],
            o["SolutionNameMin"] if "SolutionNameMin" in o else "(SolutionName N/A)"))

    # filter out dangling kernels
    oriData[5] = [ {k: v for k, v in o.items() if k != "__InUse__"}
                    for o in oriData[5] if o["__InUse__"]==True ]

    # reindex solutions
    idMap = {} # new = idMap[old]
    for i, solution in enumerate(oriData[5]):
        idMap[solution["SolutionIndex"]] = i
        oriData[5][i]["SolutionIndex"] = i
    for i, [size, [oldSolIndex, eff]] in enumerate(oriData[7]):
        oriData[7][i] = [size, [idMap[oldSolIndex], eff]]

    numInvalidRemoved = origNumSolutions - len(oriData[5])
    return oriData, numInvalidRemoved

def removeDuplicatedSolutions(oriData, prefix=""):
    origNumSolutions = len(oriData[5])

    solutionsName = {}
    solutions = []
    kernelsName = {}

    for solution in oriData[5]:
        if solution["SolutionNameMin"] not in solutionsName:
            solutionsName[solution["SolutionNameMin"]] = len(solutionsName)
            solutions.append(solution)
        if solution["KernelNameMin"] not in kernelsName:
            kernelsName[solution["KernelNameMin"]] = len(kernelsName)

    for i, solution in enumerate(solutions):
        solution["SolutionIndex"] = i

    for data in oriData[7]:
        index = data[1][0]
        data[1][0] = solutionsName[oriData[5][index]["SolutionNameMin"]]

    oriData[5] = solutions
    numRemoved = origNumSolutions - len(solutions)
    return oriData, numRemoved, len(solutions), len(kernelsName)

from .CustomYamlLoader import load_yaml_stream

def loadData(filename):
    data = load_yaml_stream(filename, yaml.CSafeLoader)
    return [filename, data]

def compareDestFolderToYaml(originalDir, incFile, incData):
    checkFolders = ["Equality", "GridBased"]
    # Parsing destination folder and yaml attribute
    destFolder = originalDir.rstrip('/').split('/')[-1]
    incAttribute = incData[11] # the last item in yaml file
    if not incAttribute:
        sys.exit(f"[Error] Empty YAML attribute. Need to set Equality or GridBased in {incFile}.")
    # Check Equality and GradBased folders only
    if destFolder in checkFolders and destFolder != incAttribute:
        restuls = f"\t{incFile} must be {destFolder} tuning"
        sys.exit(f"[Error] Destination folder(={destFolder}) failed to match YAML attribute(={incAttribute}): \n{restuls}")

def compareProblemType(oriData, incData):
    # ProblemType defined in originalFiles
    oriData[4] = ProblemType(oriData[4],False)
    problemTypeToEnum(oriData[4])
    oriData[4] = oriData[4].state
    oriProblemType = oriData[4]

    incData[4] = ProblemType(incData[4],False)
    problemTypeToEnum(incData[4])
    incData[4] = incData[4].state
    incProblemType = incData[4]

    results = ""
    if oriProblemType != incProblemType:
        for item in oriProblemType:
            if oriProblemType[item] != incProblemType[item]:
                results += f"\t{item}: {oriProblemType[item]} != {incProblemType[item]}\n"
    if (results):
        sys.exit(f"[Error] ProblemType in library logic doesn't match: \n{results}")

def msg(*args, **kwargs):
    for i in args: print(i, end=" ")
    print(**kwargs)

def verbose(*args, **kwargs):
    if verbosity < 1: return
    msg(*args, **kwargs)

def debug(*args, **kwargs):
    if verbosity < 2: return
    msg(*args, **kwargs)

def findSolutionWithIndex(solutionData, solIndex):
    # Check solution at the index corresponding to solIndex first
    if solIndex < len(solutionData) and solutionData[solIndex]["SolutionIndex"] == solIndex:
        return solutionData[solIndex]
    else:
        debug("Searching for index...")
        solution = [s for s in solutionData if s["SolutionIndex"]==solIndex]
        assert(len(solution) == 1)
        return solution[0]

# returns merged logic data as list
def mergeLogic(oriData, incData, forceMerge, noEff=False):
    origNumSizes = len(oriData[7])
    origNumSolutions = len(oriData[5])

    incData[7] = incData[7] or []
    incNumSizes = len(incData[7])
    incNumSolutions = len(incData[5])

    verbose(origNumSizes, "sizes and", origNumSolutions, "solutions in base logic file")
    verbose(incNumSizes, "sizes and", incNumSolutions, "solutions in incremental logic file")

    # trim 8-tuple gemm size format to 4-tuple [m, n, b, k]
    # TODO future gemm size could include dictionary format so need robust preprocessing
    [oriData[7], origNumSizes] = fixSizeInconsistencies(oriData[7], "base")
    [incData[7], incNumSizes] = fixSizeInconsistencies(incData[7], "incremental")

    oriData, numOrigRemoved = removeUnusedSolutions(oriData, "Base logic file: ")
    incData, numIncRemoved = removeUnusedSolutions(incData, "Inc logic file: ")

    solutionPool = deepcopy(oriData[5])
    solDict = {sol["SolutionNameMin"]: sol for sol in oriData[5]}
    solutionMap = deepcopy(oriData[7])

    origDict = {tuple(origSize): [i, origEff] for i, [origSize, [origIndex, origEff]] in enumerate(oriData[7])}
    for incSize, [incIndex, incEff] in incData[7]:
        incSolution = findSolutionWithIndex(incData[5], incIndex)

        storeEff = incEff if noEff == False else 0.0
        try:
            j, origEff = origDict[tuple(incSize)]
            if incEff > origEff or forceMerge:
                if incEff > origEff:
                    verbose("[O]", incSize, "already exists and has improved in performance.", end="")
                elif forceMerge:
                    verbose("[!]", incSize, "already exists but does not improve in performance.", end="")
                verbose("Efficiency:", origEff, "->", incEff, "(force_merge=True)" if forceMerge else "")
                solutionPool, solDict, index = addKernel(solutionPool, solDict, incSolution)
                solutionMap[j][1] = [index, storeEff]
            else:
                verbose("[X]", incSize, "already exists but does not improve in performance.", end="")
                verbose("Efficiency:", origEff, "->", incEff)
        except KeyError:
                verbose("[-]", incSize, "has been added to solution table, Efficiency: N/A ->", incEff)
                solutionPool, solDict, index = addKernel(solutionPool, solDict, incSolution)
                solutionMap.append([incSize,[index, storeEff]])

    verbose(numOrigRemoved, "unused solutions removed from base logic file")
    verbose(numIncRemoved, "unused solutions removed from incremental logic file")

    mergedData = deepcopy(oriData)
    mergedData[5] = solutionPool
    mergedData[7] = solutionMap
    mergedData, numReplaced = removeUnusedSolutions(mergedData, "Merged data: ")

    numSizesAdded = len(solutionMap) - len(oriData[7])
    numSolutionsAdded = len(solutionPool) - len(oriData[5])
    numSolutionsRemoved = numReplaced + numOrigRemoved # incremental file not counted

    return [mergedData, numSizesAdded, numSolutionsAdded, numSolutionsRemoved]

def avoidRegressions(originalDir, incrementalDir, outputPath, forceMerge, noEff=False):
    originalFiles = allFiles(originalDir)
    incrementalFiles = allFiles(incrementalDir)
    ensurePath(outputPath)

    incrementalFilesTemp = []
    originalFileNames = [os.path.split(o)[-1] for o in originalFiles]
    for file in incrementalFiles:
        if os.path.split(file)[-1] in originalFileNames:
            incrementalFilesTemp.append(file)
        else:
            outputFile = os.path.join(outputPath, os.path.split(file)[-1])
            shutil.copyfile(file, outputFile)
            msg("Copied", file, "to", outputFile)

    incrementalFiles = incrementalFilesTemp

    logicsFiles = {}
    for incFile in incrementalFiles:
        basename = os.path.split(incFile)[-1]
        origFile = os.path.join(originalDir, basename)
        logicsFiles[origFile] = origFile
        logicsFiles[incFile] = incFile

    iters = zip(logicsFiles.keys())
    logicsList = ParallelMap2(loadData, iters, "Loading Logics...", return_as="list")
    logicsDict = {}
    for i, _ in enumerate(logicsList):
        logicsDict[logicsList[i][0]] = logicsList[i][1]

    for incFile in incrementalFiles:
        basename = os.path.split(incFile)[-1]
        origFile = os.path.join(originalDir, basename)

        msg("Base logic file:", origFile, "| Incremental:", incFile, "| Merge policy: %s"%("Forced" if forceMerge else "Winner"))
        oriData = logicsDict[origFile]
        incData = logicsDict[incFile]

        # Terminate when the destination folder doesn't match Incremental logic yaml
        # For example, merge Gridbased yaml to Equality folder or Equality yaml to GridBased folder
        compareDestFolderToYaml(originalDir, incFile, incData)

        # Terminate when ProblemType of originalFiles and incrementalFiles mismatch
        compareProblemType(oriData, incData)

        sanitizeSolutions(oriData[5])
        sanitizeSolutions(incData[5])

        reNameSolutions(oriData)
        reNameSolutions(incData)

        # So far "SolutionIndex" in logic yamls has zero impact on actual 1-1 size mapping (but the order of the Solution does)
        # since mergeLogic() takes that value very seriously so we reindex them here so it doesn't choke on duplicated SolutionIndex
        oriData, numRemoved, numSolutions, numKernels = removeDuplicatedSolutions(oriData)
        msg("Base logic file:", numRemoved, "duplicated solution(s) removed,",\
            "sizes: %d, solutions: %d, kernels: %d"%(len(oriData[7]),numSolutions,numKernels))
        incData, numRemoved, numSolutions, numKernels = removeDuplicatedSolutions(incData)
        msg("Inc logic file:", numRemoved, "duplicated solution(s) removed,",\
            "sizes: %d, solutions: %d, kernels: %d"%(len(incData[7]),numSolutions,numKernels))

        mergedData, *stats = mergeLogic(oriData, incData, forceMerge, noEff)
        mergedData[0] = {"MinimumRequiredVersion": "%s"%__version__}
        msg(stats[0], "size(s) and", stats[1], "solution(s) added,", stats[2], "solution(s) removed.", \
            len(mergedData[7]), "sizes and", len(mergedData[5]), "solutions")
        with open(os.path.join(outputPath, basename), "w") as outFile:
            yaml.safe_dump(mergedData,outFile,default_flow_style=None)
        msg("File written to", os.path.join(outputPath, basename))
        msg("------------------------------")

def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("original_dir", help="The library logic directory without tuned sizes")
    argParser.add_argument("incremental_dir", help="The incremental logic directory")
    argParser.add_argument("output_dir", help="The output logic directory")
    argParser.add_argument("-v", "--verbosity", help="0: summary, 1: verbose, 2: debug", default=1, type=int)
    argParser.add_argument("--force_merge", help="Merge previously known sizes unconditionally. Default behavior if not arcturus", default="none")
    argParser.add_argument("--no_eff", help="force set eff as 0.0.", action="store_true")

    args = argParser.parse_args(sys.argv[1:])
    originalDir = args.original_dir
    incrementalDir = args.incremental_dir
    outputPath = args.output_dir
    global verbosity
    verbosity = args.verbosity
    forceMerge = args.force_merge.lower()
    no_eff = args.no_eff

    if forceMerge in ["none"]: forceMerge=True
    elif forceMerge in ["true", "1"]: forceMerge=True
    elif forceMerge in ["false", "0"]: forceMerge=False

    avoidRegressions(originalDir, incrementalDir, outputPath, forceMerge, no_eff)
