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
from Tensile.LibraryIO import dictBasedArchitectures
from Tensile.SolutionStructs.Naming import getSolutionNameMin
from Tensile.SolutionStructs.Naming import getKernelNameMin
from Tensile.SolutionStructs.Problem import ProblemType, problemTypeToEnum
from Tensile.Common import ParallelMap2

verbosity = 1

class DataAccessor:
    """
    Unified accessor for both list-based and dict-based data formats.
    Provides a consistent interface using string keys regardless of underlying data structure.
    """
    # Mapping from string keys to list indices
    KEY_TO_INDEX = {
        "MinimumRequiredVersion": 0,
        "ScheduleName": 1,
        "ArchitectureName": 2,
        "DeviceNames": 3,
        "ProblemType": 4,
        "Solutions": 5,
        "IndexOrder": 6,
        "ExactLogic": 7,
        "RangeLogic": 8,
        "LibraryType": 11
    }

    def __init__(self, data, isDict=None):
        self.data = data
        self._isDict = isDict if isDict is not None else isinstance(data, dict)

    @property
    def isDict(self):
        return self._isDict

    @property
    def isList(self):
        return not self._isDict

    def get(self, key):
        """Get a field by logical name from list- or dict-format logic data.

        Args:
            key: Logical field name (e.g. ``"Solutions"``, ``"ExactLogic"``).
                For list format, names are mapped via ``KEY_TO_INDEX``.

        Returns:
            Any | None: Field value, or None when the key is unknown or the
            list index is out of range.

        Raises:
            None.
        """
        if self._isDict:
            return self.data.get(key)
        else:
            idx = self.KEY_TO_INDEX.get(key)
            if idx is not None and idx < len(self.data):
                return self.data[idx]
            return None

    def set(self, key, value):
        """Set a field by logical name on list- or dict-format logic data.

        Args:
            key: Logical field name. For list format, must appear in
                ``KEY_TO_INDEX``.
            value: Value to store at that field.

        Returns:
            None.

        Raises:
            ValueError: On list format when *key* is not in ``KEY_TO_INDEX``.
        """
        if self._isDict:
            self.data[key] = value
        else:
            idx = self.KEY_TO_INDEX.get(key)
            if idx is not None:
                # Extend list if necessary
                while len(self.data) <= idx:
                    self.data.append(None)
                self.data[idx] = value
            else:
                raise ValueError(
                    f"Invalid key {key!r}; "
                    f"valid keys: {sorted(self.KEY_TO_INDEX)}"
                )

    def getRaw(self):
        """Return the underlying raw data."""
        return self.data

    def getSolutions(self):
        """Get solutions list."""
        return self.get("Solutions")

    def setSolutions(self, solutions):
        """Set solutions list."""
        self.set("Solutions", solutions)

    def getExactLogic(self):
        """Get exact logic list."""
        return self.get("ExactLogic")

    def setExactLogic(self, logic):
        """Set exact logic list."""
        self.set("ExactLogic", logic)

    def getProblemType(self):
        """Get problem type."""
        return self.get("ProblemType")

    def setProblemType(self, problemType):
        """Set problem type."""
        self.set("ProblemType", problemType)

    def getLibraryType(self):
        """Return library tuning mode (YAML ``LibraryType`` / list index 11).

        Used for dict-based canonical layout and for ``compareDestFolderToYaml``
        (Equality vs GridBased). Dict format: returns ``Equality``, ``GridBased``,
        or ``Range`` when set at top level; ``None`` for FreeSize or Prediction
        (no exact-matching tuning folder). List format: returns the legacy index
        11 string (Equality, GridBased, Range, FreeSize, or Prediction).

        Args:
            self: Accessor bound to list- or dict-format library logic.

        Returns:
            str | None: Tuning mode string, or None when unset or not applicable.

        Raises:
            None.
        """
        if self._isDict:
            lt = self.get("LibraryType")
            if lt in ("Equality", "GridBased", "Range"):
                return lt
            return None
        return self.get("LibraryType")

    def hasDefaultSolution(self):
        """Check if DefaultSolution exists (only for dict format)."""
        return self._isDict and "DefaultSolution" in self.data

    def getDefaultSolution(self):
        """Get default solution (only for dict format)."""
        if self._isDict:
            return self.data.get("DefaultSolution")
        return None

    def setDefaultSolution(self, defaultSol):
        """Set default solution (only for dict format)."""
        if self._isDict:
            self.data["DefaultSolution"] = defaultSol


def createAccessor(data):
    """Factory function to create a DataAccessor with proper initialization."""
    return DataAccessor(data)


# Helper function to detect architecture from loaded data
def getArchitectureFromData(data):
    """Extract architecture name from data (supports both list and dict formats)."""
    accessor = createAccessor(data)
    return accessor.get("ArchitectureName") or ""


def isDictBasedArchitecture(data: Any, dictArchs: list[str] | None = None) -> bool:
    """Check whether an architecture should use dict-based processing.

    Args:
        data: Loaded logic data in list or dict form.
        dictArchs: Optional list of lowercase architecture tags that should use dict
            format. If None, falls back to global ``dictBasedArchitectures``.

    Returns:
        bool: True when architecture matches one of the configured dict-based tags.

    Raises:
        None.
    """
    arch = str(getArchitectureFromData(data))
    archs = dictArchs if dictArchs is not None else dictBasedArchitectures
    return any(name in arch for name in archs) if arch else False


def normalizeDictLibraryLayout(data: dict[str, Any]) -> bool:
    """Normalize on-disk dict-based logic: drop ``Library``, align ``LibraryType``.

    Canonical dict YAML stores the tuning mode only at top-level ``LibraryType``
    (``Equality``, ``GridBased``, ``Range``, ``FreeSize``, or ``Prediction``) with
    exact logic in ``ExactLogic``. If an in-memory ``Library`` block is still
    present (for example after list-to-dict conversion), this copies a recognized
    ``Library["distance"]`` to ``LibraryType`` when needed, then removes ``Library``.
    Only ``Equality``, ``GridBased``, and ``Range`` are promoted from ``Library``
    or from the current top-level ``LibraryType``.

    Args:
        data: Dict-format library logic (mutated in place). Callers must only
            invoke this for dict-based architectures (see ``isDictBasedArchitecture``).

    Returns:
        bool: True when *data* was modified and should be persisted, else False.

    Raises:
        None.
    """
    if not isDictBasedArchitecture(data):
        return False
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
        _solution = deepcopy(solution) # if we don't we will see some subtle errors
        _solution["SolutionIndex"] = index
        solutionPool.append(_solution)
        solDict[solution["SolutionNameMin"]] = _solution
        debug("...A new solution has been added", end="")
    debug("({}) {}".format(index, solutionPool[index]["SolutionNameMin"] if "SolutionNameMin" in solutionPool[index] else "(SolutionName N/A)"))
    return solutionPool, solDict, index

# update dependant parameters if StaggerU == 0
def sanitizeSolutions(accessor):
    """Clear StaggerU-dependent fields when StaggerU is zero.

    Args:
        accessor: DataAccessor wrapping logic data whose Solutions list is
            updated in place.

    Returns:
        None.

    Raises:
        None.
    """
    solList = accessor.getSolutions()
    for sol in solList:
        if sol.get("StaggerU") == 0:
            sol["StaggerUMapping"] = 0
            sol["StaggerUStride"] = 0
            sol["_staggerStrideShift"] = 0

from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.Common import assignParameterWithDefault

def reNameSolutions(accessor):
    """Assign SolutionNameMin/KernelNameMin on every solution in logic data.

    Fills missing solution parameters from ``defaultSolution``, attaches
    ProblemType for naming, then removes temporary naming-only fields. On
    dict-format data with DefaultSolution, default GlobalSplitU is stripped
    unless the solution has a CustomKernelName.

    Args:
        accessor: DataAccessor wrapping logic data whose Solutions list is
            updated in place.

    Returns:
        None.

    Raises:
        None.
    """
    solList = accessor.getSolutions()
    problemType = accessor.getProblemType()

    for sol in solList:
        # Assign solution state from config, filling missing from the defaultSolution
        for key in defaultSolution:
            assignParameterWithDefault(sol, key, sol, defaultSolution)
        sol["ProblemType"] = problemType

        # For dict format (gfx1250), also set GlobalSplitU
        if accessor.isDict and accessor.hasDefaultSolution():
            if "GlobalSplitU" not in sol:
                sol["GlobalSplitU"] = accessor.getDefaultSolution()["GlobalSplitU"]

        sol["SolutionNameMin"] = getSolutionNameMin(sol, splitGSU=False)
        sol["KernelNameMin"] = getKernelNameMin(sol, splitGSU=False)
        del sol["ProblemType"]

        # For dict format (gfx1250), also delete GlobalSplitU
        if accessor.isDict and accessor.hasDefaultSolution():
            defaultGsu = accessor.getDefaultSolution().get("GlobalSplitU")
            if sol.get("GlobalSplitU") == defaultGsu and not sol.get("CustomKernelName"):
                del sol["GlobalSplitU"]

def removeUnusedSolutions(accessor, prefix=""):
    """Unified remove unused solutions using DataAccessor."""
    solutions = accessor.getSolutions()
    exactLogic = accessor.getExactLogic()
    origNumSolutions = len(solutions)

    kernelsInUse = [index for _, [index, _] in exactLogic]
    for i, solution in enumerate(solutions):
        solutionIndex = solution["SolutionIndex"]
        solutions[i]["__InUse__"] = True if solutionIndex in kernelsInUse else False

    # debug prints
    for o in [o for o in solutions if o["__InUse__"] == False]:
        debug("{}Solution ({}) {} is unused".format(
            prefix,
            o["SolutionIndex"],
            o["SolutionNameMin"] if "SolutionNameMin" in o else "(SolutionName N/A)"))

    # filter out dangling kernels
    solutions = [{k: v for k, v in o.items() if k != "__InUse__"}
                 for o in solutions if o["__InUse__"] == True]

    # reindex solutions
    idMap = {}
    for i, solution in enumerate(solutions):
        idMap[solution["SolutionIndex"]] = i
        solutions[i]["SolutionIndex"] = i
    for i, [size, [oldSolIndex, eff]] in enumerate(exactLogic):
        exactLogic[i] = [size, [idMap[oldSolIndex], eff]]

    accessor.setSolutions(solutions)
    accessor.setExactLogic(exactLogic)

    numInvalidRemoved = origNumSolutions - len(solutions)
    return accessor.getRaw(), numInvalidRemoved

def removeDuplicatedSolutions(accessor):
    """Unified remove duplicated solutions using DataAccessor."""
    solutions = accessor.getSolutions()
    exactLogic = accessor.getExactLogic()
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

    for data in exactLogic:
        index = data[1][0]
        data[1][0] = solutionsName[solutions[index]["SolutionNameMin"]]

    accessor.setSolutions(newSolutions)
    numRemoved = origNumSolutions - len(newSolutions)
    return accessor.getRaw(), numRemoved, len(newSolutions), len(kernelsName)


from Tensile import LibraryIO


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
                if k in defaultSolution.keys():
                    if v == defaultSolution[k]:
                        del kernel[k]
        LibraryIO.reorderSolutionsParams(rv)
        data = rv

    return data

from .CustomYamlLoader import load_yaml_stream

def loadData(filename: str) -> list[Any]:
    """Load logic YAML and convert dict-based architectures from list to dict.

    Args:
        filename: Path to YAML logic file.

    Returns:
        list[Any]: ``[filename, data, migrated]`` where *data* is the loaded
        (and possibly converted) logic, and *migrated* is True when a legacy
        list file was converted to dict format for a dict-based architecture,
        or when an existing dict file was rewritten to the canonical layout
        (no ``Library`` block, ``LibraryType`` set to the tuning mode).

    Raises:
        AssertionError: When the YAML stream/document structure is invalid.
        RuntimeError: When the root element is not a sequence or mapping.
        SystemExit: When ``parseLibraryLogicList`` rejects the file via
            ``printExit``.
    """
    data = load_yaml_stream(filename, yaml.CSafeLoader)
    migrated = False

    if isDictBasedArchitecture(data):
        wasList = isinstance(data, list)
        data = convertToDict(data, filename)
        layoutUpdated = normalizeDictLibraryLayout(data)
        migrated = wasList or layoutUpdated

    return [filename, data, migrated]

def compareDestFolderToYaml(originalDir, incFile, accessor):
    """Unified compare destination folder to YAML using DataAccessor."""
    checkFolders = ["Equality", "GridBased"]
    destFolder = originalDir.rstrip('/').split('/')[-1]
    incAttribute = accessor.getLibraryType()
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

def compareProblemType(oriAccessor, incAccessor):
    """Unified compare problem type using DataAccessor."""
    # ProblemType defined in originalFiles
    oriPT = ProblemType(oriAccessor.getProblemType(), False)
    problemTypeToEnum(oriPT)
    oriAccessor.setProblemType(oriPT.state)
    oriProblemType = oriPT.state

    incPT = ProblemType(incAccessor.getProblemType(), False)
    problemTypeToEnum(incPT)
    incAccessor.setProblemType(incPT.state)
    incProblemType = incPT.state

    results = ""
    if oriProblemType != incProblemType:
        for item in oriProblemType:
            if oriProblemType[item] != incProblemType[item]:
                results += f"\t{item}: {oriProblemType[item]} != {incProblemType[item]}\n"
    if results:
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

def syncDefaultParams(origData, origDefaultValues, incDefaultValues):
    # if orig and inc default values are the same, nothing to do
    if origDefaultValues == incDefaultValues:
        return

    # Parameters in solutions that need to be updated.
    # These either had their default values changed, were newly added
    # or are set to the default values (and can be removed)
    # Assume incDefaultValues is more up-to-date
    paramsToUpdate = []

    for p, v in incDefaultValues.items():
        if p not in origDefaultValues.keys() or origDefaultValues[p] != v:
            paramsToUpdate.append(p)

    for soln in origData["Solutions"]:
        for p in paramsToUpdate:
            if p in origDefaultValues.keys() and p not in soln.keys():
                soln[p] = origDefaultValues[p]
            elif p in soln.keys() and soln[p] == incDefaultValues[p]:
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
            if param in defaultSol.keys() and soln[param] == defaultSol[param]:
                del soln[param]

    if "CUCount" in defaultSol.keys():
        defaultSol.pop("CUCount")

def findSolutionWithIndex(solutionData, solIndex):
    # Check solution at the index corresponding to solIndex first
    if solIndex < len(solutionData) and solutionData[solIndex]["SolutionIndex"] == solIndex:
        return solutionData[solIndex]
    else:
        debug("Searching for index...")
        solution = [s for s in solutionData if s["SolutionIndex"]==solIndex]
        assert(len(solution) == 1)
        return solution[0]

def mergeLogic(
    oriAccessor: DataAccessor,
    incAccessor: DataAccessor,
    forceMerge: bool,
    noEff: bool = False,
) -> list[Any]:
    """Merge incremental library logic into base logic using ``DataAccessor``.

    Combines exact-logic entries from the incremental accessor into the base
    accessor: sizes that improve efficiency (or all sizes when *forceMerge*)
    adopt kernels from the incremental file. Unused solutions are trimmed
    before and after the merge.

    Args:
        oriAccessor: Base (original) logic wrapped in a ``DataAccessor``.
        incAccessor: Incremental logic wrapped in a ``DataAccessor``; mutated
            when its exact-logic list is empty (normalized to ``[]``).
        forceMerge: When True, replace entries even if efficiency does not
            improve.
        noEff: When True, store ``0.0`` for efficiency values on merged rows
            instead of the incremental file's efficiency.

    Returns:
        A four-element list ``[mergedData, numSizesAdded, numSolutionsAdded,
        numSolutionsRemoved]`` where *mergedData* is list- or dict-format logic
        (same representation as the base accessor's raw data), and the counters
        summarize changes versus the pre-merge base.

    Raises:
        AssertionError: If ``findSolutionWithIndex`` cannot resolve a solution
            index referenced by incremental exact logic.
    """
    oriSolutions = oriAccessor.getSolutions()
    oriExactLogic = oriAccessor.getExactLogic()
    incSolutions = incAccessor.getSolutions()
    incExactLogic = incAccessor.getExactLogic()

    origNumSizes = len(oriExactLogic)
    origNumSolutions = len(oriSolutions)

    incExactLogic = incExactLogic or []
    incAccessor.setExactLogic(incExactLogic)
    incNumSizes = len(incExactLogic)
    incNumSolutions = len(incSolutions)

    verbose(origNumSizes, "sizes and", origNumSolutions, "solutions in base logic file")
    verbose(incNumSizes, "sizes and", incNumSolutions, "solutions in incremental logic file")

    # trim 8-tuple gemm size format to 4-tuple [m, n, b, k]
    [oriExactLogic, origNumSizes] = fixSizeInconsistencies(oriExactLogic, "base")
    [incExactLogic, incNumSizes] = fixSizeInconsistencies(incExactLogic, "incremental")
    oriAccessor.setExactLogic(oriExactLogic)
    incAccessor.setExactLogic(incExactLogic)

    _, numOrigRemoved = removeUnusedSolutions(oriAccessor, "Base logic file: ")
    _, numIncRemoved = removeUnusedSolutions(incAccessor, "Inc logic file: ")

    # Refresh after removal
    oriSolutions = oriAccessor.getSolutions()
    oriExactLogic = oriAccessor.getExactLogic()
    incSolutions = incAccessor.getSolutions()
    incExactLogic = incAccessor.getExactLogic()

    solutionPool = deepcopy(oriSolutions)
    solDict = {sol["SolutionNameMin"]: sol for sol in oriSolutions}
    solutionMap = deepcopy(oriExactLogic)

    origDict = {tuple(origSize): [i, origEff] for i, [origSize, [origIndex, origEff]] in enumerate(oriExactLogic)}
    for incSize, [incIndex, incEff] in incExactLogic:
        incSolution = findSolutionWithIndex(incSolutions, incIndex)

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

    mergedData = deepcopy(oriAccessor.getRaw())
    mergedAccessor = createAccessor(mergedData)
    mergedAccessor.setSolutions(solutionPool)
    mergedAccessor.setExactLogic(solutionMap)
    mergedData, numReplaced = removeUnusedSolutions(mergedAccessor, "Merged data: ")

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
    for filename, data, migrated in logicsList:
        logicsDict[filename] = data
        if migrated:
            LibraryIO.writeYAML(
                filename, data, explicit_start=False, explicit_end=False, sort_keys=False
            )
            msg("Converted", filename, "to dict format")

    for incFile in incrementalFiles:
        basename = os.path.split(incFile)[-1]
        origFile = os.path.join(originalDir, basename)

        msg("Base logic file:", origFile, "| Incremental:", incFile, "| Merge policy: %s"%("Forced" if forceMerge else "Winner"))
        oriData = logicsDict[origFile]
        incData = logicsDict[incFile]

        # Create accessors for unified data access
        oriAccessor = createAccessor(oriData)
        incAccessor = createAccessor(incData)

        # Check if this architecture should use dict-based processing
        useDictBased = oriAccessor.isDict and incAccessor.isDict

        # Terminate when the destination folder doesn't match Incremental logic yaml
        compareDestFolderToYaml(originalDir, incFile, incAccessor)

        # Terminate when ProblemType of originalFiles and incrementalFiles mismatch
        compareProblemType(oriAccessor, incAccessor)

        if useDictBased:
            # Dict-based architecture: additional dict-specific processing
            if oriAccessor.hasDefaultSolution() and incAccessor.hasDefaultSolution():
                syncDefaultParams(
                    oriData,
                    deepcopy(oriAccessor.getDefaultSolution()),
                    deepcopy(incAccessor.getDefaultSolution()),
                )

        sanitizeSolutions(oriAccessor)
        sanitizeSolutions(incAccessor)

        reNameSolutions(oriAccessor)
        reNameSolutions(incAccessor)

        # Remove duplicated solutions
        oriData, numRemoved, numSolutions, numKernels = removeDuplicatedSolutions(oriAccessor)
        msg("Base logic file:", numRemoved, "duplicated solution(s) removed,",
            "sizes: %d, solutions: %d, kernels: %d" % (len(oriAccessor.getExactLogic()), numSolutions, numKernels))
        incData, numRemoved, numSolutions, numKernels = removeDuplicatedSolutions(incAccessor)
        msg("Inc logic file:", numRemoved, "duplicated solution(s) removed,",
            "sizes: %d, solutions: %d, kernels: %d" % (len(incAccessor.getExactLogic()), numSolutions, numKernels))

        mergedData, *stats = mergeLogic(oriAccessor, incAccessor, forceMerge, noEff)
        mergedAccessor = createAccessor(mergedData)

        if useDictBased:
            # Dict-based architecture: dict-specific post-processing
            mergedData["MinimumRequiredVersion"] = f"{__version__}"
            mergedData["DefaultSolution"] = deepcopy(incAccessor.getDefaultSolution())

            msg(stats[0], "size(s) and", stats[1], "solution(s) added,", stats[2], "solution(s) removed.",
                len(mergedAccessor.getExactLogic()), "sizes and", len(mergedAccessor.getSolutions()), "solutions")

            removeDefaultInitParams(mergedData)
            normalizeDictLibraryLayout(mergedData)
            if isinstance(mergedData.get("ProblemType"), dict):
                mergedData["ProblemType"] = dict(
                    sorted(mergedData["ProblemType"].items())
                )
            LibraryIO.writeYAML(os.path.join(outputPath, basename), mergedData, explicit_start=False, explicit_end=False, sort_keys=False)
        else:
            # Non dict-based architecture: list-specific post-processing
            mergedAccessor.set("MinimumRequiredVersion", {"MinimumRequiredVersion": "%s" % __version__})

            msg(stats[0], "size(s) and", stats[1], "solution(s) added,", stats[2], "solution(s) removed.",
                len(mergedAccessor.getExactLogic()), "sizes and", len(mergedAccessor.getSolutions()), "solutions")

            LibraryIO.writeYAML(os.path.join(outputPath, basename), mergedData, explicit_start=False, explicit_end=False, sort_keys=True)

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
