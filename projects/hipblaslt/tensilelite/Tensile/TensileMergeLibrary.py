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
from typing import Any

from Tensile import __version__
from Tensile import LibraryIO
from Tensile.SolutionStructs.Naming import getSolutionNameMin
from Tensile.SolutionStructs.Naming import getKernelNameMin
from Tensile.SolutionStructs.Problem import ProblemType, problemTypeToEnum
from Tensile.Common import ParallelMap2
from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.Common import assignParameterWithDefault
from .CustomYamlLoader import load_yaml_stream

verbosity = 1


def normalizeDictLibraryLayout(data: dict[str, Any]) -> bool:
    """Normalize dict-format logic: drop ``Library``, align ``LibraryType``.

    Canonical dict YAML stores the tuning mode only at top-level ``LibraryType``
    (``Equality``, ``GridBased``, ``Range``, ``FreeSize``, or ``Prediction``) with
    exact logic in ``ExactLogic``. If an in-memory ``Library`` block is still
    present (for example after list-to-dict conversion), this copies a recognized
    ``Library["distance"]`` to ``LibraryType`` when needed, then removes ``Library``.
    Only ``Equality``, ``GridBased``, and ``Range`` are promoted from ``Library``
    or from the current top-level ``LibraryType``.

    Args:
        data: Dict-format library logic (mutated in place).

    Returns:
        bool: True when *data* was modified and should be persisted, else False.

    Raises:
        None.
    """
    distanceModes = frozenset({"Equality", "GridBased", "Range"})
    old_lt = data.get("LibraryType")
    hadLibrary = "Library" in data
    lib = data.get("Library")
    distance = lib.get("distance") if isinstance(lib, dict) else None
    if distance in distanceModes:
        data["LibraryType"] = distance
    data.pop("Library", None)
    return bool(hadLibrary or data.get("LibraryType") != old_lt)


def ensurePath(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def allFiles(startDir):
    current = os.listdir(startDir)
    files = []
    for filename in [_current for _current in current if os.path.splitext(_current)[-1].lower() == '.yaml']:
        fullPath = os.path.join(startDir, filename)
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
        sizesDict[tuple(value for value in size)] = [size, index]
    newSizes = list()
    for value in sizesDict.values():
        newSizes.append(value)
    numSize = len(newSizes)
    if origNumSizes - numSize > 0:
        verbose(origNumSizes - numSize, "duplicate size(s) removed from", fileType, "logic file")
    return newSizes, len(newSizes)


def addKernel(solutionPool, solDict, solution):
    if solution["SolutionNameMin"] in solDict:
        index = solDict[solution["SolutionNameMin"]]["SolutionIndex"]
        debug("...Reuse previously existed solution", end="")
    else:
        index = len(solutionPool)
        _solution = deepcopy(solution)
        _solution["SolutionIndex"] = index
        solutionPool.append(_solution)
        solDict[solution["SolutionNameMin"]] = _solution
        debug("...A new solution has been added", end="")
    debug("({}) {}".format(
        index,
        solutionPool[index]["SolutionNameMin"] if "SolutionNameMin" in solutionPool[index] else "(SolutionName N/A)",
    ))
    return solutionPool, solDict, index


def sanitizeSolutions(data: dict[str, Any]) -> None:
    for sol in data["Solutions"]:
        if sol.get("StaggerU") == 0:
            sol["StaggerUMapping"] = 0
            sol["StaggerUStride"] = 0
            sol["_staggerStrideShift"] = 0


def reNameSolutions(data: dict[str, Any]) -> None:
    problemType = data["ProblemType"]
    defaultSol = data.get("DefaultSolution") if isinstance(data.get("DefaultSolution"), dict) else None
    for sol in data["Solutions"]:
        for key in defaultSolution:
            assignParameterWithDefault(sol, key, sol, defaultSolution)
        sol["ProblemType"] = problemType
        if defaultSol and "GlobalSplitU" not in sol:
            sol["GlobalSplitU"] = defaultSol["GlobalSplitU"]
        sol["SolutionNameMin"] = getSolutionNameMin(sol, splitGSU=False)
        sol["KernelNameMin"] = getKernelNameMin(sol, splitGSU=False)
        del sol["ProblemType"]
        if defaultSol:
            defaultGsu = defaultSol.get("GlobalSplitU")
            if sol.get("GlobalSplitU") == defaultGsu and not sol.get("CustomKernelName"):
                del sol["GlobalSplitU"]


def removeUnusedSolutions(data: dict[str, Any], prefix=""):
    solutions = data["Solutions"]
    exactLogic = data["ExactLogic"]
    origNumSolutions = len(solutions)

    kernelsInUse = [index for _, [index, _] in exactLogic]
    for i, solution in enumerate(solutions):
        solutionIndex = solution["SolutionIndex"]
        solutions[i]["__InUse__"] = solutionIndex in kernelsInUse

    for o in [o for o in solutions if o["__InUse__"] is False]:
        debug("{}Solution ({}) {} is unused".format(
            prefix,
            o["SolutionIndex"],
            o["SolutionNameMin"] if "SolutionNameMin" in o else "(SolutionName N/A)",
        ))

    solutions = [{k: v for k, v in o.items() if k != "__InUse__"}
                 for o in solutions if o["__InUse__"] is True]

    idMap = {}
    for i, solution in enumerate(solutions):
        idMap[solution["SolutionIndex"]] = i
        solutions[i]["SolutionIndex"] = i
    for i, [size, [oldSolIndex, eff]] in enumerate(exactLogic):
        exactLogic[i] = [size, [idMap[oldSolIndex], eff]]

    data["Solutions"] = solutions
    data["ExactLogic"] = exactLogic
    return data, origNumSolutions - len(solutions)


def removeDuplicatedSolutions(data: dict[str, Any]):
    solutions = data["Solutions"]
    exactLogic = data["ExactLogic"]
    origNumSolutions = len(solutions)

    solutionsName = {}
    newSolutions = []
    kernelsName = {}

    for solution in solutions:
        if solution["SolutionNameMin"] not in solutionsName:
            solutionsName[solution["SolutionNameMin"]] = len(solutionsName)
            newSolutions.append(solution)
        if solution["KernelNameMin"] not in kernelsName:
            kernelsName[solution["KernelNameMin"]] = len(kernelsName)

    for i, solution in enumerate(newSolutions):
        solution["SolutionIndex"] = i

    for entry in exactLogic:
        index = entry[1][0]
        entry[1][0] = solutionsName[solutions[index]["SolutionNameMin"]]

    data["Solutions"] = newSolutions
    numRemoved = origNumSolutions - len(newSolutions)
    return data, numRemoved, len(newSolutions), len(kernelsName)


def convertToDict(data: list | dict, filename: str) -> dict:
    """Convert list-format library logic data to dict format.

    Args:
        data: Loaded logic as a legacy list or already-converted dict.
        filename: Source path passed through to ``parseLibraryLogicList`` for
            error messages.

    Returns:
        dict: Dict-format logic; unchanged when *data* is already a dict.

    Raises:
        None: Errors from ``parseLibraryLogicList`` propagate via
            ``printExit``.
    """
    if isinstance(data, list):
        rv = LibraryIO.parseLibraryLogicList(data, filename)
        for kernel in rv["Solutions"]:
            for k in list(kernel.keys()):
                v = kernel[k]
                if k == 'ProblemType':
                    del kernel['ProblemType']
                if k in defaultSolution.keys() and v == defaultSolution[k]:
                    del kernel[k]
        # Sort each solution's keys (naming keys first, then Capital/_/lowercase)
        # so the dict layout does not depend on the source file's key order.
        rv["Solutions"] = [
            LibraryIO.reorderSolutionDictForDictMerge(dict(kernel))
            for kernel in rv["Solutions"]
        ]
        return rv
    return data


def loadData(filename: str) -> list[Any]:
    """Load logic YAML and normalize to dict format.

    Args:
        filename: Path to YAML logic file.

    Returns:
        list[Any]: ``[filename, data, normalized]`` where *data* is the loaded
        (and possibly converted) logic, and *normalized* is True when a legacy
        list file was converted to dict format, or when an existing dict file
        was rewritten to the canonical layout (no ``Library`` block,
        ``LibraryType`` set to the tuning mode).

    Raises:
        AssertionError: When the YAML stream/document structure is invalid.
        RuntimeError: When the root element is not a sequence or mapping.
        SystemExit: When ``parseLibraryLogicList`` rejects the file via
            ``printExit``.
    """
    data = load_yaml_stream(filename, yaml.CSafeLoader)
    normalized = False
    wasList = isinstance(data, list)
    data = convertToDict(data, filename)
    layoutUpdated = normalizeDictLibraryLayout(data)
    normalized = wasList or layoutUpdated
    return [filename, data, normalized]


def compareDestFolderToYaml(originalDir, incFile, incData: dict[str, Any]):
    checkFolders = ["Equality", "GridBased"]
    destFolder = originalDir.rstrip('/').split('/')[-1]
    incAttribute = incData.get("LibraryType")
    if not incAttribute:
        sys.exit(
            f"[Error] Empty YAML attribute. Need top-level LibraryType "
            f"Equality or GridBased in {incFile}."
        )
    if destFolder in checkFolders and destFolder != incAttribute:
        restuls = f"\t{incFile} must be {destFolder} tuning"
        sys.exit(
            f"[Error] Destination folder(={destFolder}) failed to match YAML "
            f"LibraryType (={incAttribute}): \n{restuls}"
        )


def compareProblemType(oriData: dict[str, Any], incData: dict[str, Any]):
    oriPT = ProblemType(oriData["ProblemType"], False)
    problemTypeToEnum(oriPT)
    oriData["ProblemType"] = oriPT.state
    oriProblemType = oriPT.state

    incPT = ProblemType(incData["ProblemType"], False)
    problemTypeToEnum(incPT)
    incData["ProblemType"] = incPT.state
    incProblemType = incPT.state

    results = ""
    if oriProblemType != incProblemType:
        for item in oriProblemType:
            if oriProblemType[item] != incProblemType[item]:
                results += f"\t{item}: {oriProblemType[item]} != {incProblemType[item]}\n"
    if results:
        sys.exit(f"[Error] ProblemType in library logic doesn't match: \n{results}")


def msg(*args, **kwargs):
    for i in args:
        print(i, end=" ")
    print(**kwargs)


def verbose(*args, **kwargs):
    if verbosity < 1:
        return
    msg(*args, **kwargs)


def debug(*args, **kwargs):
    if verbosity < 2:
        return
    msg(*args, **kwargs)


def syncDefaultParams(origData, origDefaultValues, incDefaultValues):
    if origDefaultValues == incDefaultValues:
        return

    paramsToUpdate = []
    for p in set(origDefaultValues) | set(incDefaultValues):
        if origDefaultValues.get(p) != incDefaultValues.get(p):
            paramsToUpdate.append(p)

    for soln in origData["Solutions"]:
        for p in paramsToUpdate:
            if p in origDefaultValues and p not in soln:
                soln[p] = origDefaultValues[p]
            elif p in soln and p in incDefaultValues and soln[p] == incDefaultValues[p]:
                del soln[p]


def removeDefaultInitParams(data: dict[str, Any]) -> None:
    """Drop solution keys that match ``DefaultSolution`` and strip ``CUCount`` from defaults.

    For each entry in ``data["Solutions"]``, removes any parameter whose value
    equals the corresponding value in ``data["DefaultSolution"]``. Also removes
    ``CUCount`` from ``DefaultSolution`` when present (it belongs to
    architecture metadata, not per-solution defaults).

    Args:
        data: Dict-format library logic (mutated in place). Must contain
            ``"DefaultSolution"`` and ``"Solutions"``.

    Returns:
        None.

    Raises:
        None.
    """
    defaultSol = data["DefaultSolution"]
    for soln in data["Solutions"]:
        solnParams = list(soln.keys())
        for param in solnParams:
            if param in defaultSol and soln[param] == defaultSol[param]:
                del soln[param]
    if "CUCount" in defaultSol:
        defaultSol.pop("CUCount")


def findSolutionWithIndex(solutionData, solIndex):
    if solIndex < len(solutionData) and solutionData[solIndex]["SolutionIndex"] == solIndex:
        return solutionData[solIndex]
    debug("Searching for index...")
    solution = [s for s in solutionData if s["SolutionIndex"] == solIndex]
    assert len(solution) == 1
    return solution[0]


def mergeLogic(oriData: dict[str, Any], incData: dict[str, Any], forceMerge, noEff=False):
    oriSolutions = oriData["Solutions"]
    oriExactLogic = oriData["ExactLogic"]
    incSolutions = incData["Solutions"]
    incExactLogic = incData["ExactLogic"] or []

    origNumSizes = len(oriExactLogic)
    origNumSolutions = len(oriSolutions)
    incData["ExactLogic"] = incExactLogic
    incNumSizes = len(incExactLogic)
    incNumSolutions = len(incSolutions)

    verbose(origNumSizes, "sizes and", origNumSolutions, "solutions in base logic file")
    verbose(incNumSizes, "sizes and", incNumSolutions, "solutions in incremental logic file")

    oriExactLogic, origNumSizes = fixSizeInconsistencies(oriExactLogic, "base")
    incExactLogic, incNumSizes = fixSizeInconsistencies(incExactLogic, "incremental")
    oriData["ExactLogic"] = oriExactLogic
    incData["ExactLogic"] = incExactLogic

    _, numOrigRemoved = removeUnusedSolutions(oriData, "Base logic file: ")
    _, numIncRemoved = removeUnusedSolutions(incData, "Inc logic file: ")

    oriSolutions = oriData["Solutions"]
    oriExactLogic = oriData["ExactLogic"]
    incSolutions = incData["Solutions"]
    incExactLogic = incData["ExactLogic"]

    solutionPool = deepcopy(oriSolutions)
    solDict = {sol["SolutionNameMin"]: sol for sol in oriSolutions}
    solutionMap = deepcopy(oriExactLogic)

    origDict = {tuple(origSize): [i, origEff] for i, [origSize, [origIndex, origEff]] in enumerate(oriExactLogic)}
    for incSize, [incIndex, incEff] in incExactLogic:
        incSolution = findSolutionWithIndex(incSolutions, incIndex)
        storeEff = incEff if noEff is False else 0.0
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
            solutionMap.append([incSize, [index, storeEff]])

    verbose(numOrigRemoved, "unused solutions removed from base logic file")
    verbose(numIncRemoved, "unused solutions removed from incremental logic file")

    mergedData = deepcopy(oriData)
    mergedData["Solutions"] = solutionPool
    mergedData["ExactLogic"] = solutionMap
    mergedData, numReplaced = removeUnusedSolutions(mergedData, "Merged data: ")

    numSizesAdded = len(solutionMap) - len(oriExactLogic)
    numSolutionsAdded = len(solutionPool) - len(oriSolutions)
    numSolutionsRemoved = numReplaced + numOrigRemoved

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
    for filename, data, normalized in logicsList:
        logicsDict[filename] = data
        if normalized:
            msg(filename, "was normalized to canonical dict layout in memory")

    for incFile in incrementalFiles:
        basename = os.path.split(incFile)[-1]
        origFile = os.path.join(originalDir, basename)

        msg("Base logic file:", origFile, "| Incremental:", incFile, "| Merge policy: %s" % ("Forced" if forceMerge else "Winner"))
        oriData = logicsDict[origFile]
        incData = logicsDict[incFile]

        compareDestFolderToYaml(originalDir, incFile, incData)
        compareProblemType(oriData, incData)

        origDefault = oriData.get("DefaultSolution") if isinstance(oriData.get("DefaultSolution"), dict) else None
        incDefault = incData.get("DefaultSolution") if isinstance(incData.get("DefaultSolution"), dict) else None
        if origDefault and incDefault:
            syncDefaultParams(oriData, deepcopy(origDefault), deepcopy(incDefault))

        sanitizeSolutions(oriData)
        sanitizeSolutions(incData)
        reNameSolutions(oriData)
        reNameSolutions(incData)

        oriData, numRemoved, numSolutions, numKernels = removeDuplicatedSolutions(oriData)
        msg("Base logic file:", numRemoved, "duplicated solution(s) removed,",
            "sizes: %d, solutions: %d, kernels: %d" % (len(oriData["ExactLogic"]), numSolutions, numKernels))
        incData, numRemoved, numSolutions, numKernels = removeDuplicatedSolutions(incData)
        msg("Inc logic file:", numRemoved, "duplicated solution(s) removed,",
            "sizes: %d, solutions: %d, kernels: %d" % (len(incData["ExactLogic"]), numSolutions, numKernels))

        mergedData, *stats = mergeLogic(oriData, incData, forceMerge, noEff)
        mergedData["MinimumRequiredVersion"] = f"{__version__}"
        if incDefault:
            mergedData["DefaultSolution"] = deepcopy(incDefault)

        msg(stats[0], "size(s) and", stats[1], "solution(s) added,", stats[2], "solution(s) removed.",
            len(mergedData["ExactLogic"]), "sizes and", len(mergedData["Solutions"]), "solutions")

        if mergedData.get("DefaultSolution"):
            removeDefaultInitParams(mergedData)
        normalizeDictLibraryLayout(mergedData)
        if isinstance(mergedData.get("ProblemType"), dict):
            mergedData["ProblemType"] = dict(sorted(mergedData["ProblemType"].items()))

        LibraryIO.writeYAML(
            os.path.join(outputPath, basename),
            mergedData,
            explicit_start=False,
            explicit_end=False,
            sort_keys=False,
        )
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

    if forceMerge in ["none"]:
        forceMerge = True
    elif forceMerge in ["true", "1"]:
        forceMerge = True
    elif forceMerge in ["false", "0"]:
        forceMerge = False

    avoidRegressions(originalDir, incrementalDir, outputPath, forceMerge, no_eff)
