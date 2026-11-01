################################################################################
#
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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

import hashlib
import functools
import json
import os
import shutil
import yaml
import sys
import time
import itertools

import rocisa

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, TypedDict

from . import SolutionLibrary, LibraryIO
from .KernelWriter import DebugConfig
from .KernelHelperNaming import KernelHelperEnum, HelperKernelObjectsMemo
from .Toolchain.Component import Assembler
from .SolutionStructs.Problem import ProblemType, ProblemSizes
from .SolutionStructs.Solution import Solution
from .Common.TypeValidationErrors import ConfigTypeError
from .SolutionStructs.Validators.MatrixInstruction import matrixInstructionToMIParameters, \
                                                                 validateMIParameters
from .SolutionStructs.Naming import getSolutionNameMin, getKernelNameMin

from .BenchmarkStructs import BenchmarkProcess
from .backends import BackendFactory
from .backends.config import parse_backend_config
from .Contractions import ProblemType as ContractionsProblemType
from .ClientWriter import runClient, writeClientConfig, writeClientConfigIni, getClientExecutablePath
from .KernelWriterAssembly import KernelWriterAssembly
from .TensileCreateLibrary import BenchmarkKernelBuild, benchmarkPoolSize, copyStaticFiles, \
        finishSolutionsAndKernels, generateBenchmarkKernels, helperCompileExecutor, libraryDir, \
        prepareSolutionsAndKernels, tensileLibraryFile
from .CustomKernels import getCustomKernelConfig
from .Toolchain.Assembly import AssemblyToolchain
from .Toolchain.Source import SourceToolchain
from .Common import HR, print1, print2, IsaInfo, IsaVersion, \
        printExit, printWarning, ensurePath, tqdm, state, \
        BENCHMARK_PROBLEMS_DIR, BENCHMARK_DATA_DIR, ParallelMap2
from .Common.Architectures import archNamesByIsa, isaToGfx, gfxToVariants
from .Common.GlobalParameters import globalParameters, startTime
from .Common.TimingInstrumentation import timing_context

_CACHE_FIELDS = {
    "ConstantParams": "constantParams",
    "ForkParams": "forkParams",
    "ParamGroups": "paramGroups",
    "CustomKernels": "customKernels",
    "InternalSupportParams": "internalSupportParams",
    "CustomKernelWildcard": "customKernelWildcard",
}
_KERNEL_NAMING_VERSION = 2

# Deterministic synthetic GFlops value emitted per (problem size, solution) cell in the
# --cpu-only results CSV. Fixed (never random / never timestamped) so the file is
# byte-identical across runs (T8) and the value is large enough that addFromCSV's winner
# selection has a well-defined winner. See the synthetic-perf caveat in
# tensilelite/Tests/unit/characterization/_codegen/GPU-MOCK.md.
_CPU_ONLY_SYNTHETIC_GFLOPS = 1000.0


def _poolSize():
    """Size of this module's process pools: the same as the kernel build's pools, so the worker
    pool is not resized between them (see benchmarkPoolSize)."""
    return benchmarkPoolSize(globalParameters["GenerateSourcesAndExit"])


def _writeSyntheticResultsCSV(resultsFileName, problemSizes, gfxName, numSolutions):
    """Write a deterministic synthetic results CSV in the schema LibraryLogic.addFromCSV
    consumes, for the GPU-less --cpu-only path.

    The real client writes one CSV row per benchmarked problem size; column 0 is the perf
    unit header ("GFlops"), the next ``numIndices`` columns are the problem-size indices,
    and the trailing ``numSolutions`` columns are per-solution GFlops. We mirror exactly
    that layout (data-stub style of ProblemSizesMock in SolutionStructs/Problem.py): the
    first data-row column repeats the gfx arch name (matching the captured real-CSV
    fixtures), the size columns come straight from ``problemSizes.problems[*].sizes``, and
    every solution cell is the fixed ``_CPU_ONLY_SYNTHETIC_GFLOPS`` so the file is
    deterministic / byte-identical across runs.
    """
    problems = list(problemSizes.problems)
    numIndices = len(problems[0].sizes) if problems else 0
    # At least one solution column so addFromCSV's solutionStartIdx is well-defined.
    numSolutions = max(int(numSolutions), 1)

    header = ["GFlops"]
    header += ["Size%d" % i for i in range(numIndices)]
    header += ["Solution_%d" % i for i in range(numSolutions)]

    lines = [",".join(header)]
    for problem in problems:
        row = [gfxName]
        row += [str(int(s)) for s in problem.sizes]
        row += [repr(_CPU_ONLY_SYNTHETIC_GFLOPS)] * numSolutions
        lines.append(",".join(row))

    with open(resultsFileName, "w", newline="") as f:
        f.write("\n".join(lines) + "\n")


def _resultsFileConsumed(hasLibraryLogic: bool, backendName: str) -> bool:
    """Whether anything reads the benchmark client's results CSVs.

    The LibraryLogic step reads them from 2_BenchmarkData, and the Ductile backend
    scores each GA generation from them. Otherwise the client is told to write none.
    """
    return hasLibraryLogic or backendName == "ductile"


# 12 hex chars = 48 bits. Birthday-collision likely around 2^24 (~16M) entries
# in one caches/ dir; tuning sweeps produce <<1k entries, so collision risk is
# negligible. Lookup re-validates by content so a collision becomes a recompile,
# never a wrong-cache hit.
_CACHE_KEY_LEN = 12


def _cacheDataMatches(cacheData, benchmarkStep):
    """Check if cached data matches the current benchmark step parameters."""
    return cacheData.get("KernelNamingVersion") == _KERNEL_NAMING_VERSION and all(
        cacheData[f] == getattr(benchmarkStep, attr) for f, attr in _CACHE_FIELDS.items()
    )


def cmdLineArchsFor(archNames, isaInfoMap) -> List[str]:
    """The compiler targets a build for these ISAs will actually be given.

    A stepping shares its architecture's ISA, so the ISA-derived name would tune
    with the other stepping's compiler target and produce code objects the
    tuned-for silicon cannot load. The requested names win where they cover an
    ISA; the rest fall back to that ISA's own name.
    """
    buildArchNames = archNamesByIsa(archNames or [])
    return [
        var
        for isa in isaInfoMap.keys()
        for var in gfxToVariants(buildArchNames.get(isa) or isaToGfx(isa))
    ]


def _computeCacheKey(benchmarkStep, cmdLineArchs):
    """Compute a deterministic hash from the cache-relevant parameter fields.

    The compiler targets belong in the key because they decide what the cached
    code objects contain while leaving every benchmark parameter untouched: two
    architectures sharing an ISA are tuned by two runs that agree on every field
    here, and without the targets the second would load the first's objects and
    hand the wrong machine code to silicon that cannot run it.
    """
    cacheFields = {f: getattr(benchmarkStep, attr) for f, attr in _CACHE_FIELDS.items()}
    cacheFields["KernelNamingVersion"] = _KERNEL_NAMING_VERSION
    cacheFields["_Architectures"] = ",".join(sorted(cmdLineArchs))
    canonical = json.dumps(cacheFields, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:_CACHE_KEY_LEN]


class CacheEntry(TypedDict):
    CodeObjectFiles: List[str]
    LibraryFile: str


def _readCacheIfValid(cachePath, benchmarkStep, mismatchMessage) -> Optional[CacheEntry]:
    """Return the cache entry from cachePath iff its params match benchmarkStep, else None.

    Returning None triggers a recompile, which is the right thing for any
    cache.yaml that doesn't contain everything --use-cache needs.
    """
    if not os.path.isfile(cachePath):
        return None
    try:
        c = LibraryIO.read(cachePath)
    except (OSError, yaml.YAMLError) as e:
        printWarning(f"Ignoring unreadable cache entry {cachePath}: {e}")
        return None
    try:
        if _cacheDataMatches(c, benchmarkStep):
            return {"CodeObjectFiles": c["CodeObjectFiles"], "LibraryFile": c["LibraryFile"]}
    except KeyError as e:
        printWarning(f"Ignoring incompatible cache entry {cachePath} (missing field {e})")
        return None
    printWarning(mismatchMessage.format(path=cachePath))
    return None


def _loadCacheIfMatches(cacheDir, benchmarkStep) -> Optional[CacheEntry]:
    """Return the cache entry from the hash-keyed cacheDir/cache.yaml iff matching, else None."""
    cachePath = os.path.join(cacheDir, "cache.yaml")
    # Hash matched but content didn't: collision. Loser will be overwritten on the next write.
    return _readCacheIfValid(
        cachePath, benchmarkStep,
        "Cache hash collision at {path}; will overwrite on recompile",
    )


def _resetCacheDir(cacheDir):
    """Wipe and recreate cacheDir so a write fully replaces any prior content."""
    if os.path.exists(cacheDir):
        shutil.rmtree(cacheDir)
    os.makedirs(cacheDir)


def _generate_single_solution(perm, problemType, constantParams, assembler, debugConfig, isaInfoMap,
                              rocIsaData=None):
    """Helper function to generate a single solution from a permutation.

    This function handles standard permutations without group_ parameter expansion.
    For GA individuals that may have group_ parameters, use the GA backend's equivalent.

    ParallelMap2 hands a worker globalParameters and nothing else, so the rocIsa
    singleton here has never been init'd and every capability lookup would read a
    default. Validators that ask the assembly backend what a solution emits need the
    parent's capabilities to answer the same way the emitter will.
    """
    if rocIsaData is not None:
        rocisa.rocIsa.getInstance().setData(rocIsaData)

    solution = {
        "ProblemType": deepcopy(problemType.state),
        "ISA": next(iter(isaInfoMap.keys()))
    }
    solution.update(constantParams)
    solution.update(perm)

    return _build_and_validate_solution(solution, assembler, debugConfig, isaInfoMap)


def _build_and_validate_solution(solution, assembler, debugConfig, isaInfoMap, silent=False):
    """Build and validate a solution from a parameterized dict.
    
    This is the core logic used by both fork and GA solution generation.
    Returns the Solution object if valid, None otherwise.
    """
    try:
        mi = solution["MatrixInstruction"]
        workgroup = solution["WorkGroup"]
        ptype = solution["ProblemType"]
        isa = solution["ISA"]

        if solution["WavefrontSize"] == -1:
            solution["WavefrontSize"] = 32 if isaInfoMap[isa].archCaps["HasWave32"] else 64
        wavefrontSize = solution["WavefrontSize"]

        if len(mi) == 9:
            miParams = matrixInstructionToMIParameters(mi, isa, wavefrontSize, ptype, workgroup, isaInfoMap, solution.get("SourceSwap", False))
            solution.update(miParams)
        elif len(mi) == 0:
            solution["EnableMatrixInstruction"] = False

        if validateMIParameters(solution, isaInfoMap):
            solutionObject = Solution(
                solution,
                debugConfig.splitGSU,
                debugConfig.printSolutionRejectionReason,
                debugConfig.printIndexAssignmentInfo,
                assembler,
                isaInfoMap,
            )
            if solutionObject["Valid"]:
                return solutionObject
            elif not silent and debugConfig.printSolutionRejectionReason:
                print1("rejecting solution " + str(solution))
        elif not silent and debugConfig.printSolutionRejectionReason:
            print1("rejecting solution " + str(solution))
    except ConfigTypeError:
        # Re-raise so this is not swallowed by the generic except below.
        raise
    except Exception as e:
        if not silent:
            print(f"Error processing permutation: {e}")
    return None

def _generateForkedSolutions(problemType, constantParams, forkPermutations, assembler: Assembler,
                             debugConfig: DebugConfig, isaInfoMap: Dict[IsaVersion, IsaInfo]):
    """Creates a list with a Solution object for each parameter combination in forkPermutations using parallel processing"""
    print1("# Enumerating Solutions (Parallel)")

    forkIters = zip(
        forkPermutations,
        itertools.repeat(problemType),
        itertools.repeat(constantParams),
        itertools.repeat(assembler),
        itertools.repeat(debugConfig),
        itertools.repeat(isaInfoMap),
        itertools.repeat(rocisa.rocIsa.getInstance().getData())
    )
    raw_solutions = ParallelMap2(_generate_single_solution, forkIters, "fork solutions", return_as="list",
                                 procs=_poolSize())

    # Filter out None and duplicates
    solutionSet = set()
    solutions = []
    for sol in tqdm(raw_solutions, "Remove duplicate solutions"):
        if sol is not None and sol not in solutionSet:
            solutionSet.add(sol)
            solutions.append(sol)

    return solutions


def _getCustomKernelSolutionObj(
        kernelName,
        internalSupportParams,
        assembler: Assembler,
        debugConfig: DebugConfig,
        isaInfoMap: Dict[IsaVersion, IsaInfo],
        directory=None
    ):
    """Creates the Solution object for a custom kernel"""
    sol = getCustomKernelConfig(kernelName, internalSupportParams, directory)

    mi = sol["MatrixInstruction"]
    isa = next(iter(isaInfoMap.keys()))
    wavefrontSize = sol["WavefrontSize"]
    ptype = sol["ProblemType"]
    workgroup = sol.get("WorkGroup", None)

    if len(mi) == 9:
        miParams = matrixInstructionToMIParameters(mi, isa, wavefrontSize, ptype, workgroup, isaInfoMap, sol.get("SourceSwap", False))
        sol.update(miParams)
    elif len(mi) == 0:
        sol["EnableMatrixInstruction"] = False

    sol = Solution(
               sol,
               debugConfig.printIndexAssignmentInfo,
               debugConfig.printSolutionRejectionReason,
               debugConfig.printIndexAssignmentInfo,
               assembler,
               isaInfoMap,
           )

    return sol


def _hashableProblemTypeKV(k, v):
    """Make (k, v) hashable for set-difference comparisons of ProblemType.

    Lists become tuples; otherwise we fall back to repr(v) for unhashable
    nested values (e.g. dict-valued problem-type fields). Note: repr is
    insertion-order-sensitive for dicts, so equivalent dicts with different
    insertion orders compare as different. Acceptable for the diagnostic
    "differing parameters" message produced below.
    """
    if isinstance(v, list):
        return (k, tuple(v))
    try:
        hash(v)
        return (k, v)
    except TypeError:
        return (k, repr(v))


def _generateCustomKernelSolutions(
        problemType,
        customKernels,
        internalSupportParams,
        failOnMismatch,
        assembler: Assembler,
        debugConfig: DebugConfig,
        isaInfoMap: Dict[str, IsaInfo]
    ):
    """Creates a list with a Solution object for each name in customKernel"""
    solutions = []
    for kernelName in customKernels:
        print1("# Processing custom kernel {}".format(kernelName))
        try:
            solution = _getCustomKernelSolutionObj(kernelName, internalSupportParams, assembler, debugConfig, isaInfoMap)
        except (RuntimeError, KeyError, TypeError) as e:
            printWarning(f"Skipping custom kernel '{kernelName}': missing or invalid custom.config ({e})")
            continue
        # The ActivationType setting in YAML is meaningless in customKernel case.
        # Therefore, we override the customKernel setting with the ActivationType value from ProblemType to avoid false alarms during subsequent problemType checks.
        solution["ProblemType"]["ActivationType"] = problemType["ActivationType"]
        if solution["ProblemType"] != problemType:
            # Raise error if this kernel was specifically requested and problem type doesn't match
            if failOnMismatch:
                benchmarkSet = {_hashableProblemTypeKV(k, v) for k, v in problemType.items()}
                customSet = {_hashableProblemTypeKV(k, v) for k, v in solution["ProblemType"].items()}

                msg = "The problem type in the config file does not match " \
                        "that of the custom kernel, {}.".format(kernelName) \
                        + "\nDiffering parameters:\n" \
                        + "\tConfig values:\n\t" \
                        + str(sorted(benchmarkSet - (customSet & benchmarkSet))) \
                        + "\n\tCustom kernel values:\n\t" \
                        +  str(sorted(customSet - (customSet & benchmarkSet)))
                printExit(msg)
            else:
                print1("# Rejected {}: Problem Type doesn't match".format(kernelName))
        else:
            print1("# Added {} to solutions".format(kernelName))
            if solution["Valid"]:
                solutions.append(solution)
            elif debugConfig.printSolutionRejectionReason:
                print1("rejecting solution " + str(solution))

    return solutions

def _create_solution_from_state(solutionState, problemType,
                                splitGSU, printSolutionRejectionReason,
                                printIndexAssignmentInfo, assembler, isaInfoMap, srcFile):
    """Top-level function for parallel Solution construction from a pool solution state dict.

    Preserves the AssignedDerivedParameters flags from the lib so that the
    Solution constructor uses the parameters as-is without re-applying
    derivation rules.
    """
    try:
        solutionState.pop("SolutionIndex", None)
        solutionState["ProblemType"] = problemType
        return Solution(
            solutionState, splitGSU, printSolutionRejectionReason,
            printIndexAssignmentInfo, assembler, isaInfoMap, srcFile
        )
    except Exception as e:
        print(f"Error creating solution from pool: {e}")
        return None

def _constructAllPoolSolutions(poolEntries, assembler, debugConfig, isaInfoMap):
    """Construct Solution objects from all matched pool entries in a single parallel pass."""
    problemType = ProblemType(poolEntries[0][1]["ProblemType"], debugConfig.printIndexAssignmentInfo)

    allSolStates = []
    allSrcFiles = []
    for poolFile, poolData in poolEntries:
        data = poolData["Solutions"]
        nSols = len(data)
        allSolStates.append(data)
        allSrcFiles.append(itertools.repeat(poolFile, nSols))
        print1("# Pool file {}: {} solutions".format(poolFile, nSols))

    solIters = zip(
        itertools.chain.from_iterable(allSolStates),
        itertools.repeat(problemType),
        itertools.repeat(debugConfig.splitGSU),
        itertools.repeat(debugConfig.printSolutionRejectionReason),
        itertools.repeat(debugConfig.printIndexAssignmentInfo),
        itertools.repeat(assembler),
        itertools.repeat(isaInfoMap),
        itertools.chain.from_iterable(allSrcFiles),
    )
    raw = ParallelMap2(_create_solution_from_state, solIters,
                       "Constructing solutions from {} pool file(s)".format(len(poolEntries)),
                       return_as="list", procs=_poolSize())
    return [s for s in raw if s is not None]

def _parsePoolFile(poolFile):
    """Read and parse a pool file. Returns (ptStr, data, poolFile)."""
    data = LibraryIO.read(poolFile, customizedLoader=True)
    if isinstance(data, list):
        data = LibraryIO.parseLibraryLogicList(data, poolFile)
    ptStr = str(ProblemType(data["ProblemType"], False))
    return (ptStr, poolFile, data)

def _loadSolutionPool(solutionPoolFiles):
    """Parse all pool files and index by ProblemType string.

    Returns {ptStr: [(poolFile, parsedData), ...]}.
    Each file is parsed once; the parsed data is cached for later solution construction.
    """
    poolIndex = {}
    for (ptStr, poolFile, data) in ParallelMap2(_parsePoolFile, zip(solutionPoolFiles), "load solution pool", return_as="list",
                                                procs=_poolSize()):
        print2("# Pool file {}: ProblemType={}, {} solutions".format(poolFile, ptStr, len(data["Solutions"])))
        poolIndex.setdefault(ptStr, []).append((poolFile, data))
    return poolIndex

class BenchmarkFilesBuild(NamedTuple):
    """A benchmark step's files between kernel generation's phases (prepareBenchmarkFiles)."""

    kernels: BenchmarkKernelBuild
    # Writes the rest once the kernels are generated; returns (codeObjectFiles, libraryFileRel).
    finish: Callable[[], Tuple[List[str], str]]


def writeBenchmarkFiles(*args, **kwargs):
    """Write all the files needed for a given benchmarking step.

    Takes prepareBenchmarkFiles' arguments but helperExecutor, and does all of its work: prepare,
    generate the step's kernels, finish. Returns (codeObjectFiles, libraryFileRel).
    """
    with helperCompileExecutor() as helperExecutor:
        build = prepareBenchmarkFiles(*args, helperExecutor=helperExecutor, **kwargs)
        generateBenchmarkKernels([build.kernels])
        return build.finish()


def prepareBenchmarkFiles(
        stepBaseDir,
        solutions,
        problemSizes,
        biasTypeArgs,
        factorDimArgs,
        activationArgs,
        icacheFlushArgs,
        stepName,
        solutionSummationSizes,
        asmToolchain: AssemblyToolchain,
        srcToolchain: SourceToolchain,
        sourcePath: Path,
        debugConfig: DebugConfig,
        gateTypeArgs,
        deviceId: int,
        gfxName: str,
        isaInfoMap: Dict[IsaVersion, IsaInfo],
        probSolMap: dict,
        archNames: Optional[List[str]] = None,
        writeResultsFile: bool = True,
        *,
        helperExecutor,
    ) -> BenchmarkFilesBuild:
    """Write the files needed for a given benchmarking step, up to the generation of its kernels
    (generateBenchmarkKernels, which may take the kernels of several steps at once): the step's
    kernel and helper-kernel lists, with the helper kernels written and their compile queued on
    helperExecutor (helperCompileExecutor). The returned finish function writes the rest.

    writeResultsFile: False makes ClientParameters.ini ask the client for no results CSV.
    """

    ensurePath(sourcePath)
    copyStaticFiles(sourcePath)

    kernels = []
    kernelHelperObjs = []
    kernelHelperNames = set()

    # get unique kernels and kernel helpers
    with timing_context("python_kernel_bench_setup"):
        # Helper objects depend on a few solution fields only: build them once per distinct
        # input; the name-based dedupe below keeps the same helpers in the same order.
        helperKernelObjects = HelperKernelObjectsMemo(KernelHelperEnum.All,
                                                      str(asmToolchain.assembler.path),
                                                      isaInfoMap)
        for solution in tqdm(solutions, "Finding unique solutions"):
            solutionKernels = solution.getKernels()
            for kernel in solutionKernels:
                kernels.append(kernel)

            for ko in helperKernelObjects(solution):
                kname = ko.getKernelName()
                if kname not in kernelHelperNames:
                    kernelHelperObjs.append(ko)
                    kernelHelperNames.add(kname)

        kernelWriterAssembly = KernelWriterAssembly(asmToolchain.assembler, debugConfig)

        cmdLineArchs = cmdLineArchsFor(archNames, isaInfoMap)
    # write solution, kernels and CMake
    problemType = solutions[0]["ProblemType"]
    kernelBuild = prepareSolutionsAndKernels( \
                            sourcePath,
                            asmToolchain,
                            srcToolchain,
                            solutions,
                            kernels,
                            kernelHelperObjs,
                            kernelWriterAssembly,
                            debugConfig.splitGSU,
                            cmdLineArchs,
                            helperExecutor,
                            disableAsmComments=globalParameters["DisableAsmComments"],
                            errorTolerant=True,
                            generateSourcesAndExit=globalParameters["GenerateSourcesAndExit"], # put in debug config
                            compress=False,
                            removeTemporaries=not globalParameters["KeepBuildTmp"],
                        )
    finish = functools.partial(_finishBenchmarkFiles, kernelBuild, stepBaseDir, solutions,
                               problemSizes, biasTypeArgs, factorDimArgs, activationArgs,
                               icacheFlushArgs, stepName, solutionSummationSizes, asmToolchain,
                               sourcePath, debugConfig, gateTypeArgs, deviceId, gfxName,
                               isaInfoMap, probSolMap, writeResultsFile, problemType, cmdLineArchs)
    return BenchmarkFilesBuild(kernelBuild, finish)


def _finishBenchmarkFiles(kernelBuild, stepBaseDir, solutions, problemSizes, biasTypeArgs,
                          factorDimArgs, activationArgs, icacheFlushArgs, stepName,
                          solutionSummationSizes, asmToolchain, sourcePath, debugConfig,
                          gateTypeArgs, deviceId, gfxName, isaInfoMap, probSolMap,
                          writeResultsFile, problemType, cmdLineArchs):
    """The rest of prepareBenchmarkFiles once the step's kernels are generated: assemble and link
    them, then write the library and the client configuration."""
    codeObjectFiles, _ = finishSolutionsAndKernels(kernelBuild)
    # ^ this is where solutions is mutated
    with timing_context("python_kernel_bench_postprocess"):
        with timing_context("python_benchpost_naming"):
            for s in solutions:
                s["SolutionNameMin"] = getSolutionNameMin(s, debugConfig.splitGSU)
                s["KernelNameMin"]   = getKernelNameMin(s, debugConfig.splitGSU)

            # Benchmark builds always target a single base arch; pick its subdir.
            newLibraryDir = ensurePath(libraryDir(sourcePath, cmdLineArchs[0]))
            newLibraryFile = os.path.join(newLibraryDir, "TensileLibrary")
            libraryExt = ".yaml" if globalParameters["LibraryFormat"] == "yaml" else ".dat"
            newLibraryFileFull = newLibraryFile + libraryExt

        with timing_context("python_benchpost_lib_construction"):
            newLibrary = SolutionLibrary.MasterSolutionLibrary.BenchmarkingLibrary(
                             solutions,
                             asmToolchain.assembler,
                             debugConfig.splitGSU,
                             debugConfig.printSolutionRejectionReason,
                             debugConfig.printIndexAssignmentInfo,
                             isaInfoMap,
                         )
            newLibrary.applyNaming(debugConfig.splitGSU)

        with timing_context("python_benchpost_library_write"):
            LibraryIO.write(newLibraryFile, state(newLibrary), globalParameters["LibraryFormat"])

        codeObjectFiles = [os.path.relpath(f, sourcePath) \
                for f in codeObjectFiles]
        libraryFileRel = os.path.relpath(newLibraryFileFull, sourcePath)

        with timing_context("python_benchpost_client_config"):
            if "TileAwareSelection" in problemType and problemType["TileAwareSelection"]:
                maxMacroTile0 = 0
                maxMacroTile1 = 0
                for solution in solutions:
                    macroTile0 = solution["MacroTile0"]
                    macroTile1 = solution["MacroTile1"]
                    if macroTile0 > maxMacroTile0:
                        maxMacroTile0 = macroTile0
                    if macroTile1 > maxMacroTile1:
                        maxMacroTile1 = macroTile1
                idealM = 36 * maxMacroTile0
                idealN = 36 * maxMacroTile1
                idealSizes = []
                if problemType["Batched"]:
                    for idealK in solutionSummationSizes:
                        idealSize = {"Exact": [idealM, idealN, 1, idealK]}
                        idealSizes.append(idealSize)
                else:
                    for idealK in solutionSummationSizes:
                        idealSize = {"Exact": [idealM, idealN, idealK]}
                        idealSizes.append(idealSize)
                idealProblemSizes = ProblemSizes(problemType, idealSizes)
                writeClientConfig(True, solutions, idealProblemSizes, biasTypeArgs, \
                                  factorDimArgs, activationArgs, icacheFlushArgs, stepName, stepBaseDir, \
                                  newLibrary, codeObjectFiles, True, deviceId, gfxName, \
                                  libraryFile=newLibraryFileFull, gateTypeArgs=gateTypeArgs, probSolMap=probSolMap,
                                  sourceDir=str(sourcePath))
            else:
                writeClientConfig(True, solutions, problemSizes, biasTypeArgs, \
                                  factorDimArgs, activationArgs, icacheFlushArgs, stepName, stepBaseDir, \
                                  newLibrary, codeObjectFiles, False, deviceId, gfxName, \
                                  libraryFile=newLibraryFileFull, gateTypeArgs=gateTypeArgs, probSolMap=probSolMap,
                                  sourceDir=str(sourcePath), writeResultsFile=writeResultsFile)

    if len(solutions) == 0:
        printExit("write solutions and kernels results 0 valid soultion.")

    return codeObjectFiles, libraryFileRel


class _StepBatch:
    """Benchmark steps whose kernels are generated in one pool (timing_runs/DECISIONS.md topic 7,
    C). _benchmarkProblemType prepares each step into the batch (solutions, helper kernels written
    and their compile queued on helperExecutor), and main ends each problem size group with the
    function that collects its results. finish() then generates the kernels of all prepared steps
    at once and does the rest of their work in order (assembly, library, client run, results
    copy), as when each step was built on its own.

    A step that would read what an earlier unfinished one writes has to wait for it: its caller
    finishes the batch first (buildsInto, collectsInto)."""

    def __init__(self, helperExecutor):
        self.helperExecutor = helperExecutor
        self._kernelBuilds = []
        # In order: (step finish function, None) or (None, (benchmarkErrors, collect)).
        self._pending = []
        self._cacheDirs = set()
        self._resultsFiles = set()
        # Failed client runs of finished steps whose group is not collected yet.
        self._failedSteps = 0

    def addStep(self, cacheDir, kernelBuild: Optional[BenchmarkKernelBuild],
                finish: Callable[[], Tuple[str, int]]):
        """A prepared step: the directory it builds into, its kernels (None for a --use-cache hit:
        nothing to generate) and the function that finishes it, returning (resultsFileName,
        client return code)."""
        self._cacheDirs.add(cacheDir)
        if kernelBuild is not None:
            self._kernelBuilds.append(kernelBuild)
        self._pending.append((finish, None))

    def endGroup(self, newResultsFileName, benchmarkErrors: int, collect: Callable[[int], None]):
        """The steps added since the last group end make a problem size group whose results go to
        newResultsFileName: collect(errors) runs after them, with errors = benchmarkErrors plus
        their failed client runs."""
        self._resultsFiles.add(newResultsFileName)
        self._pending.append((None, (benchmarkErrors, collect)))

    def buildsInto(self, cacheDir) -> bool:
        """Whether an unfinished step builds into cacheDir."""
        return cacheDir in self._cacheDirs

    def collectsInto(self, newResultsFileName) -> bool:
        """Whether an unfinished group's results go to newResultsFileName."""
        return newResultsFileName in self._resultsFiles

    def finish(self):
        """Generate the kernels of the unfinished steps in one pool, then finish the steps and
        collect their groups, in order. The steps are taken off the batch first: after an error
        in here, finishing again does nothing. Returns whether there was anything to finish."""
        kernelBuilds, self._kernelBuilds = self._kernelBuilds, []
        pending, self._pending = self._pending, []
        hadSteps = bool(pending)
        self._cacheDirs.clear()
        self._resultsFiles.clear()
        if kernelBuilds:
            with timing_context("python_kernel_compilation"):
                generateBenchmarkKernels(kernelBuilds)
        while pending:
            finishStep, groupEnd = pending.pop(0)
            if finishStep is not None:
                _, returncode = finishStep()
                if returncode:
                    self._failedSteps += 1
            else:
                benchmarkErrors, collect = groupEnd
                collect(benchmarkErrors + self._failedSteps)
                self._failedSteps = 0
        return hadSteps


@contextmanager
def _stepBatchFor(batchSteps: bool):
    """The _StepBatch of a BenchmarkProblems run (None unless batchSteps), finished when the block
    ends. If preparing a step fails (an exception or printExit), the steps prepared before it are
    finished first (built, client run, results collected), as when every step was built before
    the next one was prepared; then the error goes on. An error while finishing (in the shared
    codegen pool, or a step's assembly, helper compile or client run) is not finished around: the
    steps whose kernels were not generated, or that come after the failing one, stay unfinished.

    printExit has printed its FATAL line before the steps are finished here: a closing FATAL
    line then points back to it, so a run's log still ends with the reason it stopped."""
    with helperCompileExecutor() as helperExecutor:
        stepBatch = _StepBatch(helperExecutor) if batchSteps else None
        try:
            yield stepBatch
        except SystemExit:
            if stepBatch is not None and stepBatch.finish():
                print("Tensile::FATAL: stopped by the error above, after finishing the steps"
                      " prepared before it")
                sys.stdout.flush()
            raise
        except Exception:
            if stepBatch is not None:
                stepBatch.finish()
            raise
        if stepBatch is not None:
            stepBatch.finish()


def _benchmarkProblemType(backendConfig, problemTypeConfig, problemSizeGroupConfig, problemSizeGroupIdx,
                          outerBenchmarkIdx, configPath, useCache,
                         asmToolchain: AssemblyToolchain, srcToolchain: SourceToolchain, cCompiler: str,
                         buildTmpPath: Path, benchmarkProblemsPath: Path,
                         debugConfig: DebugConfig, deviceId: int,
                         gfxName: str, isaInfoMap: Dict[str, IsaInfo], probSolMap: dict,
                         buildOnly: bool = False,
                         solutionPoolIndex: dict = None,
                         archNames: Optional[List[str]] = None,
                         writeResultsFile: bool = True,
                         stepBatch: Optional["_StepBatch"] = None,
    ):
    """Run the benchmarking for a single entry in the BenchmarkProblems of a Tensile config

    Args:
        buildOnly: If True, generate and build kernels but skip benchmarking.
        solutionPoolIndex: Dict mapping ProblemType string to pool file path.
            Only the matched file's solutions are constructed (on demand).
        writeResultsFile: If False, the client writes no results CSV
            (ClientParameters.ini sets no-results-file).
        stepBatch: Only prepare the steps into this batch (_StepBatch), for a backend that
            does not use the benchmark results: the caller generates the kernels and finishes
            the steps, and the returned fail count covers none of them.
    """
    if solutionPoolIndex is None:
        solutionPoolIndex = {}
    benchmarkTestFails = 0

    print1("")
    print1(HR)
    print1("# Converting Config to BenchmarkProcess Object")
    print1(HR)
    print1("")
    # Pass the YAML location prefix so type-mismatch errors carry a clear
    # path like ``BenchmarkProblems[<outer>][<1+sizeGroup>].ForkParameters.<Key>``.
    keyPathPrefix = f"BenchmarkProblems[{outerBenchmarkIdx}][{1 + problemSizeGroupIdx}]"
    benchmarkProcess = BenchmarkProcess(
        problemTypeConfig, problemSizeGroupConfig, debugConfig.printIndexAssignmentInfo,
        keyPathPrefix=keyPathPrefix, srcFile=configPath, gfxName=gfxName,
    )

    enableTileSelection = benchmarkProcess.problemType["TileAwareSelection"]
    groupName = "{}_{:02d}".format(str(benchmarkProcess.problemType), problemSizeGroupIdx)
    groupNamePath = benchmarkProblemsPath / groupName
    ensurePath(groupNamePath / "Data")

    totalBenchmarkSteps = len(benchmarkProcess)
    resultsFileBaseFinal = None

    print1("# NumBenchmarkSteps: {}".format(totalBenchmarkSteps))
    print1("")
    print1(HR)
    print1("# Done Creating BenchmarkProcess Object")
    print1(HR)

    def runStep(benchmarkStepIdx):
        """One benchmark step. A function of its own, so that the functions it leaves in
        stepBatch keep this step's variables."""
        nonlocal resultsFileBaseFinal
        benchmarkStep = benchmarkProcess[benchmarkStepIdx]
        stepName = str(benchmarkStep)
        shortName = stepName

        print1("\n")
        print1(HR)
        currentTime = time.time()
        elapsedTime = currentTime - startTime
        print1("# Benchmark Step: {} - {} {:.3f}s".format(groupName, stepName, elapsedTime))
        print1("# Num Sizes: {}".format(benchmarkStep.problemSizes.totalProblemSizes))
        print1("# Factor Dim steps: {}".format(benchmarkStep.factorDimArgs.totalProblemSizes))
        print1("# Bias Type steps: {}".format(benchmarkStep.biasTypeArgs.totalProblemSizes))
        print1("# Activation steps: {}".format(benchmarkStep.activationArgs.totalProblemSizes))
        print1("# ICacheFlush steps: {}".format(len(benchmarkStep.icacheFlushArgs)))
        if solutionPoolIndex:
            print1("# Solution Pool: (solutions loaded from library logic file)")
        else:
            print1("# Fork Parameters:")
            for k, v in benchmarkStep.forkParams.items():
                print1("#     {}: {}".format(k, v))
            if benchmarkStep.internalSupportParams:
                print("# InternalSupportParams: {}".format(benchmarkStep.internalSupportParams))

        shortNamePath = ensurePath(groupNamePath / shortName)
        stepBaseDir = shortNamePath
        resultsFileBase = os.path.normpath(shortNamePath / ".." / "Data" / shortName)

        if benchmarkStep.isFinal():
            resultsFileBaseFinal = resultsFileBase
        resultsFileName = resultsFileBase + ".csv"
        solutionsFileName = resultsFileBase + ".yaml"

        backend_name = str(backendConfig.get("Name", "tensile")).lower()

        # check if a solution cache exists and if it matches our solution parameters
        cacheKey = _computeCacheKey(benchmarkStep, cmdLineArchsFor(archNames, isaInfoMap))
        cacheDir = os.path.join(stepBaseDir, "caches", cacheKey)
        sourcePath = Path(cacheDir) / "source"
        if stepBatch is not None and stepBatch.buildsInto(cacheDir):
            # An earlier step of the batch builds into the same directory (the cache key leaves
            # out the problem sizes): finish it before this step resets or reads the directory.
            stepBatch.finish()
        
        with timing_context("python_cache_check"):
            cacheValid = False
            cachedLibraryFile = None
            if useCache:
                cacheEntry = _loadCacheIfMatches(cacheDir, benchmarkStep)
                if cacheEntry is not None:
                    cacheValid = True
                    codeObjectFiles = cacheEntry["CodeObjectFiles"]
                    cachedLibraryFile = cacheEntry["LibraryFile"]
                elif os.path.isdir(os.path.join(stepBaseDir, "caches")):
                    printWarning("Cache data does not match config: redoing solution generation")

        # Pre-compute benchmark runner helper
        def benchmark_runner(solutions, isCached=False, buildOnly=False):
            """Compile and benchmark the given solutions.
            
            This function is backend-agnostic: receives pre-computed solutions
            and handles compilation, library creation, and benchmarking.
            
            With stepBatch, it only prepares the step (solutions checked, helper kernels written
            and their compile queued): the kernels are generated with the other steps of
            stepBatch, and stepBatch's caller runs the rest (finishStep).
            
            Args:
                solutions: List of Solution objects to compile and benchmark
                isCached: If True, indicates that the solutions are cached
                buildOnly: If True, skip benchmarking and only compile
                
            Returns:
                - resultsFileName: Path to benchmark results CSV
                - returncode: Benchmark client return code (0=success)
                None with stepBatch.
            """
            nonlocal benchmarkTestFails
            
            writeFiles = None
            kernelBuild = None
            prevCount = None
            if not isCached:
                 # New compiles always go to the hash-keyed dir, never overwrite legacy in place.
                _resetCacheDir(cacheDir)
                ensurePath(sourcePath)
                # handle no valid solutions
                if not solutions:
                    msg = "Your parameters resulted in 0 valid solutions."
                    if debugConfig.printSolutionRejectionReason:
                        msg += "\nExamine reject and backtrace messages above to see why" \
                                "and where solutions were rejected."
                    else:
                        msg += "\nYou should re-run with \"PrintSolutionRejectionReason: True\"" \
                                "to see why each parameter combination was rejected."
                    if stepBatch is not None:
                        # Finish the earlier steps first, so that the FATAL line comes last.
                        stepBatch.finish()
                    printExit(msg)

                for solution in solutions:
                    print2("#    ({}:{}) {}".format(0, 0, getSolutionNameMin(solution, debugConfig.splitGSU)))
                print2(HR)

                prevCount = len(solutions)
                filesArgs = (stepBaseDir, solutions, \
                        benchmarkStep.problemSizes, benchmarkStep.biasTypeArgs, \
                        benchmarkStep.factorDimArgs, benchmarkStep.activationArgs, \
                        benchmarkStep.icacheFlushArgs, shortName, [], asmToolchain, srcToolchain, \
                        sourcePath, debugConfig, getattr(benchmarkStep, "gateTypeArgs", None), deviceId, gfxName, isaInfoMap, probSolMap, archNames)
                if stepBatch is None:
                    writeFiles = functools.partial(writeBenchmarkFiles, *filesArgs,
                                                   writeResultsFile=writeResultsFile)
                else:
                    with timing_context("python_kernel_compilation"):
                        filesBuild = prepareBenchmarkFiles(*filesArgs, writeResultsFile=writeResultsFile,
                                                           helperExecutor=stepBatch.helperExecutor)
                    kernelBuild = filesBuild.kernels
                    writeFiles = filesBuild.finish

            finish = functools.partial(finishStep, solutions, isCached, buildOnly, writeFiles, prevCount)
            if stepBatch is not None:
                stepBatch.addStep(cacheDir, kernelBuild, finish)
                return None
            resultsFileName, returncode = finish()
            if returncode:
                benchmarkTestFails += 1
            return resultsFileName, returncode

        def finishStep(solutions, isCached, buildOnly, writeFiles, prevCount):
            """benchmark_runner's work from kernel generation on (writeFiles generates the kernels,
            or finishes a stepBatch build). Returns what benchmark_runner returns without
            stepBatch."""
            nonlocal codeObjectFiles, cachedLibraryFile

            if not isCached:
                # write benchmarkFiles (kernel generation and compilation)
                with timing_context("python_kernel_compilation"):
                    codeObjectFiles, libraryFileRel = writeFiles()
                # ^ this mutates solutions

                # write cache data
                with timing_context("python_write_cache"):
                    cachePath = os.path.join(cacheDir, "cache.yaml")
                    cacheData = {
                        "KernelNamingVersion": _KERNEL_NAMING_VERSION,
                        "CodeObjectFiles": codeObjectFiles,
                        "LibraryFile": libraryFileRel,
                        "ConstantParams": benchmarkStep.constantParams,
                        "ForkParams": benchmarkStep.forkParams,
                        "ParamGroups": benchmarkStep.paramGroups,
                        "CustomKernels": benchmarkStep.customKernels,
                        "InternalSupportParams": benchmarkStep.internalSupportParams,
                        "CustomKernelWildcard": benchmarkStep.customKernelWildcard
                    }
                    LibraryIO.writeYAML(cachePath, cacheData)

                print1("# Actual Solutions: {} / {} after KernelWriter\n" \
                        .format(len(solutions), prevCount ))

                # add SolutionIndex and SolutionNameMin into benchmark yaml
                with timing_context("python_solution_indexing"):
                    for i in range(0, len(solutions)):
                        solution = solutions[i]
                        solution["SolutionIndex"] = i
                        solution["SolutionNameMin"] = getSolutionNameMin(solution, debugConfig.splitGSU)
                        solution["KernelNameMin"]   = getKernelNameMin(solution, debugConfig.splitGSU)
            else:
                solutions = None
                print1("# Using cached solution data")

                ssProblemType = ProblemType(problemTypeConfig, debugConfig.printIndexAssignmentInfo)
                conProblemType = ContractionsProblemType.FromOriginalState(ssProblemType)
                outFile = os.path.join(sourcePath, "ClientParameters.ini")

                cachedLibraryFile = tensileLibraryFile(sourcePath, gfxName, globalParameters["LibraryFormat"])
                # writeMsgPack produces .dat.zlib; accept both on disk.
                if not os.path.isfile(cachedLibraryFile) \
                    and not os.path.isfile(str(cachedLibraryFile) + ".zlib"):
                    printExit(
                        f"cache.yaml refers to a library file that no longer "
                        f"exists on disk: {cachedLibraryFile}. The cache directory may "
                        f"have been partially deleted; remove the parent caches/ "
                        f"directory and re-run without --use-cache.")

                writeClientConfigIni(True, benchmarkStep.problemSizes, benchmarkStep.biasTypeArgs,
                                    benchmarkStep.factorDimArgs, benchmarkStep.activationArgs,
                                    benchmarkStep.icacheFlushArgs, conProblemType,
                                    sourcePath, codeObjectFiles,
                                    resultsFileName if writeResultsFile else None,
                                    outFile, deviceId, gfxName, libraryFile=cachedLibraryFile,
                                    gateTypeArgs=getattr(benchmarkStep, "gateTypeArgs", None), probSolMap=probSolMap)

            # I think the size portion of this yaml could be removed,
            # but for now it's needed, so we update it even in the cache case
            with timing_context("python_write_solutions"):
                # Tuning-only: names above already carry its tag; keep the values out of the yaml.
                for s in solutions or []:
                    if isinstance(s, dict):
                        s.pop("_StinkyTofuParameters", None)
                    else:
                        s.dropStinkyTofuParameters()
                LibraryIO.writeSolutions(solutionsFileName, benchmarkStep.problemSizes, benchmarkStep.biasTypeArgs,
                    benchmarkStep.activationArgs, solutions, gateTypeArgs=getattr(benchmarkStep, "gateTypeArgs", None), cache=isCached)

            returncode = 0
            # run benchmarking client
            if buildOnly:
                print1("# Build-only mode: skipping benchmark.")
            elif not os.path.exists(resultsFileName) or globalParameters["ForceRedoBenchmarkProblems"]:
                libraryLogicPath = None
                forBenchmark = True
                configPaths = [str(sourcePath / "ClientParameters.ini")]

                if enableTileSelection:
                    configPaths.append(str(sourcePath / "ClientParameters_Granularity.ini"))
                # A results CSV left by an earlier run into this output directory
                # (reached here only with ForceRedoBenchmarkProblems) would otherwise be
                # copied to 2_BenchmarkData as this run's results.
                if not writeResultsFile and os.path.isfile(resultsFileName):
                    os.remove(resultsFileName)
                returncode = runClient(libraryLogicPath, forBenchmark, enableTileSelection, srcToolchain.compiler, cCompiler, shortNamePath, configPaths=configPaths)

                # --cpu-only plumbing: runClient covered the config/run-script writing then
                # stubbed the device launch (returncode 0). The device never produced a perf
                # CSV, so synthesize a deterministic one here in the addFromCSV schema. We are
                # inside the `not os.path.exists(resultsFileName)` branch, so this honors the
                # existing skip-if-already-benchmarked guard. It stands in for the client's
                # CSV, so it is written only when the client would write one.
                if globalParameters["CpuOnly"] and writeResultsFile:
                    numSolutions = len(solutions) if solutions else 1
                    _writeSyntheticResultsCSV(resultsFileName, benchmarkStep.problemSizes,
                                            gfxName, numSolutions)

                if returncode:
                    printWarning("BenchmarkProblems: Benchmark Process exited with code {}" \
                            .format(returncode))
            else:
                print1("# Already benchmarked; skipping.")

            return resultsFileName, returncode

        backend = BackendFactory.create(backend_name)

        try:
            cfg_name = Path(globalParameters["ConfigPath"][0]).stem
        except (KeyError, IndexError):
            cfg_name = "config" 

        benchmark_config = {
            "benchmarkStep": benchmarkStep,
            "forkParametersEnabled": problemSizeGroupConfig["ForkParameters"],
            "problemType": benchmarkProcess.problemType,
            "assembler": asmToolchain.assembler,
            "debugConfig": debugConfig,
            "isaInfoMap": isaInfoMap,
            "cacheValid": cacheValid,
            "rootPath": Path(benchmarkProblemsPath).parent,
            "sourcePath": sourcePath,
            "benchmarkStepIdx": benchmarkStepIdx,
            "totalBenchmarkSteps": totalBenchmarkSteps,
            "configName": cfg_name,
            # Additional config for TensileBackend (cache + solution pool)
            "solutionPoolIndex": solutionPoolIndex,
        }

        backend.run(
            backendConfig.get("Config", {}),
            benchmark_config,
            benchmark_runner,
            cacheValid=cacheValid,
            buildOnly=buildOnly,
        )

        # End Iteration
        currentTime = time.time()
        elapsedTime = currentTime - startTime
        print1("{}\n# {}\n# {}: End - {:.3f}s\n{}\n" \
                .format(HR, groupName, shortName, elapsedTime, HR))

    for benchmarkStepIdx in range(0, totalBenchmarkSteps):
        runStep(benchmarkStepIdx)

    return (resultsFileBaseFinal, benchmarkTestFails)


def main(
    backend,
    config,
    useCache,
    asmToolchain: AssemblyToolchain,
    srcToolchain: SourceToolchain,
    cCompiler: str,
    outputPath: Path,
    buildTmpPath: Path,
    debugConfig: DebugConfig,
    deviceId: int,
    gfxName: str,
    isaInfoMap: Dict[str, IsaInfo],
    probSolMap: dict,
    buildOnly: bool = False,
    solutionPoolFiles: list = None,
    archNames: Optional[List[str]] = None,
    hasLibraryLogic: bool = True,
):
    """Entry point for the "BenchmarkProblems" section of a Tensile config yaml

    Args:
        backend: Backend configuration from config["Backend"] (defaults to {})
        buildOnly: If True, generate and build kernels but skip benchmarking.
        solutionPoolFiles: If non-empty, load solutions from matching pool files
            instead of generating from ForkParameters.
        archNames: The gfx names this build was asked for, when the caller knows
            them. Needed to tune a stepping, whose ISA cannot name its target.
        hasLibraryLogic: Whether the config has a LibraryLogic section, which reads
            the results CSVs. Without one (and without the Ductile backend) the client
            writes no results CSV, so a rerun into the same output directory with
            ForceRedoBenchmarkProblems=False benchmarks again: the results CSV is the
            "already benchmarked" marker.
    """
    if config is None:
        print(f'No config specified in {globalParameters["ConfigPath"]}, built client only')
        return

    # Parse and validate backend config using shared contract
    backend = parse_backend_config(backend if backend else None)
    backend_name = str(backend.get("Name", "tensile")).lower()
    writeResultsFile = _resultsFileConsumed(hasLibraryLogic, backend_name)

    # Load solution pool index if provided and supported by the backend
    solutionPoolIndex = None
    if solutionPoolFiles:
        if BackendFactory.create(backend_name).supports_solution_pool():
            solutionPoolIndex = _loadSolutionPool(solutionPoolFiles)
        else:
            printWarning(f"Backend '{backend_name}' does not support solution pools; ignoring provided pool files.")
            solutionPoolFiles = None

    benchmarkDataPath = ensurePath(outputPath / BENCHMARK_DATA_DIR)

    totalTestFails = 0
    # Recover the originating YAML path (or first of a list) so input-YAML
    # validation errors can carry a file:line prefix in their message.
    configPath = globalParameters.get("ConfigPath", "")
    if isinstance(configPath, (list, tuple)):
        configPath = configPath[0] if configPath else ""

    def collectResults(resultsFileBaseFinal, newResultsFileName, newSolutionsFileName,
                       newGranularityFileName, benchmarkErrors):
        """A problem size group's benchmark steps are done: count their failures and copy
        their results to 2_BenchmarkData."""
        nonlocal totalTestFails
        totalTestFails += benchmarkErrors

        if buildOnly:
            print1("# Build-only mode: skipping result collection.")
        else:
            print("clientExit={} {} for {}" \
                    .format(totalTestFails, "(ERROR)" if totalTestFails else "(PASS)", \
                    globalParameters["ConfigPath"]) )

            # copy data
            resultsFileBase = resultsFileBaseFinal
            resultsFileName = resultsFileBase + ".csv"
            solutionsFileName = resultsFileBase + ".yaml"
            granularityFileName = resultsFileBase + "_Granularity.csv"
            if os.path.isfile(resultsFileName):
                shutil.copy(resultsFileName, newResultsFileName)
            elif not writeResultsFile and os.path.isfile(newResultsFileName):
                # From an earlier run (ForceRedoBenchmarkProblems): not this run's results.
                os.remove(newResultsFileName)
            shutil.copy(solutionsFileName, newSolutionsFileName)
            if os.path.isfile(granularityFileName):
                shutil.copy(granularityFileName, newGranularityFileName)

    # A backend that does not use the benchmark results (TensileBackend) leaves the steps
    # independent of each other: they are all prepared first, their kernels generated in one
    # pool (timing_runs/DECISIONS.md topic 7, C), then each is finished in order (_StepBatch).
    batchSteps = not BackendFactory.create(backend_name).uses_benchmark_results()

    with _stepBatchFor(batchSteps) as stepBatch:
        for outerIdx, benchmarkProblemTypeConfig in enumerate(config):
            problemTypeConfig = benchmarkProblemTypeConfig[0]
            if len(benchmarkProblemTypeConfig) < 2:
                problemSizeGroupConfigs = [{}]
            else:
                problemSizeGroupConfigs = benchmarkProblemTypeConfig[1:]

            for idx, sizeGroupConfig in enumerate(problemSizeGroupConfigs):
                print2("ProblemTypeConfig: {}".format(problemTypeConfig))
                problemTypeObj = ProblemType(problemTypeConfig, debugConfig.printIndexAssignmentInfo)

                # using a suffix to check the csv version (for later addFromCSV())
                csvSuffix = "_CSVWinner" if globalParameters["CSVExportWinner"] else ""
                # results files will be named
                newResultsFileName = os.path.join(benchmarkDataPath, "{}_{:02d}{}.csv" \
                        .format(str(problemTypeObj), idx, csvSuffix) )
                newSolutionsFileName = os.path.join(benchmarkDataPath, "{}_{:02d}{}.yaml" \
                        .format(str(problemTypeObj), idx, csvSuffix) )
                newGranularityFileName = os.path.join(benchmarkDataPath, "{}_{:02d}{}.gsp" \
                        .format(str(problemTypeObj), idx, csvSuffix) )

                # The check below reads the results of an earlier group of the same name.
                if stepBatch is not None and not globalParameters["ForceRedoBenchmarkProblems"] \
                        and stepBatch.collectsInto(newResultsFileName):
                    stepBatch.finish()

                # skip if possible
                if globalParameters["ForceRedoBenchmarkProblems"] \
                        or not os.path.exists(newResultsFileName):

                    # benchmark problem size group
                    benchmarkProblemsPath = ensurePath(outputPath / BENCHMARK_PROBLEMS_DIR)
                    (resultsFileBaseFinal, benchmarkErrors) = \
                            _benchmarkProblemType(
                                backendConfig=backend,
                                problemTypeConfig=problemTypeConfig,
                                problemSizeGroupConfig=sizeGroupConfig,
                                problemSizeGroupIdx=idx,
                                outerBenchmarkIdx=outerIdx,
                                configPath=configPath,
                                useCache=useCache,
                                asmToolchain=asmToolchain,
                                srcToolchain=srcToolchain,
                                cCompiler=cCompiler,
                                buildTmpPath=buildTmpPath,
                                benchmarkProblemsPath=benchmarkProblemsPath,
                                debugConfig=debugConfig,
                                deviceId=deviceId,
                                gfxName=gfxName,
                                isaInfoMap=isaInfoMap,
                                probSolMap=probSolMap,
                                buildOnly=buildOnly,
                                solutionPoolIndex=solutionPoolIndex,
                                archNames=archNames,
                                writeResultsFile=writeResultsFile,
                                stepBatch=stepBatch,
                            )
                    collect = functools.partial(collectResults, resultsFileBaseFinal,
                                                newResultsFileName, newSolutionsFileName,
                                                newGranularityFileName)
                    if stepBatch is None:
                        collect(benchmarkErrors)
                    else:
                        stepBatch.endGroup(newResultsFileName, benchmarkErrors, collect)
                else:
                    print1("# {}_{:02d} already benchmarked; skipping." \
                            .format(str(problemTypeObj), idx) )

    if globalParameters["ExitOnFails"] and totalTestFails:
        sys.exit(1)
