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

import rocisa

import copy
import functools
import glob
import itertools
import os
import shutil
import subprocess
import sys
import pickle
import zlib
from pathlib import Path
from timeit import default_timer as timer
from typing import Collection, Dict, List, NamedTuple, Optional, Union

from Tensile import LibraryIO
from Tensile.Common import (
    CHeader,
    DebugConfig,
    ensurePath,
    HR,
    IsaVersion,
    ParallelMap2,
    print1,
    print2,
    printWarning,
    printExit,
    state,
    tqdm,
    setVerbosity,
    getVerbosity,
)
from Tensile.Common.Architectures import archNamesByIsa, architectureMap, baseArchName, gfxToIsa, isaCollisionFreeGroups, isaToGfx, splitArchsFromPredicates, filterLogicFilesByPredicates, expandAllArchitectures, steppingArchOf
from Tensile.Common.Capabilities import applyArchCapOverrides, makeIsaInfoMap
from Tensile.Common.GlobalParameters import assignGlobalParameters, globalParameters
from Tensile.Common.TimingInstrumentation import timing_context
from Tensile.SolutionStructs.Naming import getKernelFileBase, getKeyNoInternalArgs, getKernelNameMin

from Tensile.CustomYamlLoader import load_logic_gfx_arch, archMatch
from Tensile.KernelHelperNaming import kernelObjectNameCallables, initHelperKernelObjects
from Tensile.KernelWriterAssembly import KernelWriterAssembly
from Tensile.KernelWriterBase import (
    KERNEL_HELPER_FILENAME_CPP,
    KERNEL_HELPER_FILENAME_H,
)
from Tensile.resources import copy_static_headers
from Tensile.SolutionLibrary import MasterSolutionLibrary, PlaceholderLibrary
from Tensile.SolutionStructs import Solution
from Tensile.SolutionStructs.Solution import (
    raiseIfTypeMismatches,
    mergeTypeMismatchCollector,
    resetTypeMismatchCollector,
)
from Tensile.verify_stinky_comment_vs_elf_text import verify_stinky_paths
from Tensile.Toolchain.Assembly import makeAssemblyToolchain, buildAssemblyCodeObjectFiles
from Tensile.Toolchain.Source import makeSourceToolchain, buildSourceCodeObjectFiles
from Tensile.Toolchain.Validators import (
    ToolchainDefaults,
    validateToolchain,
)
from Tensile.Toolchain.Component import Assembler
from Tensile.Utilities.Decorators.Profile import profile
from Tensile.Utilities.Decorators.Timing import timing

from .ParseArguments import parseArguments


def libraryRoot(outputPath: Union[str, Path]) -> Path:
    """The library/ root directory under outputPath.

    Used as the dispatch root by builders that fan kernels out into per-base
    subdirectories at write time.
    """
    return Path(outputPath) / "library"


def libraryDir(outputPath: Union[str, Path], arch: str) -> Path:
    """The per-base-arch library subdirectory: <outputPath>/library/<base>/.

    Target features (xnack+/xnack-, sramecc, etc.) are stripped from the path —
    variants of one base co-locate in one directory, disambiguated by kernel
    filename suffix. Layout matches the runtime probe in tensile_host.cpp which
    strips at the first colon before looking up the subdirectory.
    """
    return libraryRoot(outputPath) / baseArchName(arch)


def _baseArchs(archs: Collection[str]) -> List[str]:
    """Unique base archs (predicates and qualifiers stripped), sorted for determinism."""
    return sorted({baseArchName(a) for a in archs})


# Set on the children of a fan-out. Only they share their scratch parent with a
# run they must not disturb, so only they have to leave siblings alone; every
# other run owns build_tmp outright and clears it as it always did.
_GROUP_BUILD_ENV = "TENSILE_GROUP_BUILD"


def buildTmpRoot(outputPath: Union[str, Path]) -> Path:
    """The scratch parent every run writing into one output directory shares.

    Derived from outputPath alone so cleanup can name it without going through
    a run's own scratch path, which a malformed OutputPath could otherwise make
    resolve to somewhere rmtree must never reach.
    """
    return Path(outputPath) / "build_tmp"


def buildTmpDir(outputPath: Union[str, Path], archs: Collection[str]) -> Path:
    """This run's scratch directory under buildTmpRoot(outputPath).

    Covering a stepping and the architecture it steps from takes two runs, since
    the two spell one ISA and a run names its target by the ISA. Pointed at one
    output directory, everything they write outside library/ collides: kernel
    basenames come from the ISA, so both name their .s and .o identically while
    the machine code inside differs.

    Only a run asked for a stepping is named apart, so architecture sets that
    predate steppings keep the directory they had. One suffix suffices: the two
    never share a group, and steppings of different base architectures share no
    ISA, so they arrive together in one group.

    Only names the architecture table knows become path components. A spec is
    otherwise free text -- gfxToIsa reads the leading gfx digits and ignores the
    rest -- so "gfx1250/.." would smuggle a segment into a name handed to rmtree.
    """
    steppings = sorted(
        {
            baseArchName(a)
            for a in archs
            if baseArchName(a) in architectureMap and steppingArchOf(a) is not None
        }
    )
    # A stem can be empty (Path("/").stem), and "<root>" / "" is <root> itself --
    # a run would then take the shared parent as its own scratch directory.
    stem = Path(outputPath).stem.upper() or "SCRATCH"
    return buildTmpRoot(outputPath) / (f"{stem}-{steppings[0]}" if steppings else stem)


def tensileLibraryFile(outputPath: Union[str, Path], arch: str, library_format: str = "msgpack") -> Path:
    """The canonical TensileLibrary path for one base arch under outputPath.

    Composes ``<outputPath>/library/<base>/TensileLibrary.<ext>`` where ``ext``
    is ``.yaml`` for the YAML format and ``.dat`` for msgpack. The base arch
    is derived from ``arch`` via the same colon-strip rule as ``libraryDir``,
    so cooked variants like ``gfx942:sramecc+:xnack+`` resolve to the same
    file as the bare ``gfx942`` arch.

    This is the file that ``writeClientConfigIni``'s ``libraryFile`` argument
    must point to under the per-base layout. Callers (BenchmarkProblems'
    cache-hit branch, ClientWriter's benchmark-parameters helper) reach for
    it from different parts of the pipeline; the helper keeps the
    "library/<base>/TensileLibrary.<ext>" naming convention in one place so
    future format/extension changes touch a single call site.
    """
    ext = ".yaml" if library_format == "yaml" else ".dat"
    return libraryDir(outputPath, arch) / f"TensileLibrary{ext}"


class KernelCodeGenResult(NamedTuple):
    err: int
    src: Union[str, bytes]
    header: Optional[str]
    name: str
    targetObjFilename: str
    isa: IsaVersion
    wavefrontSize: int
    cuoccupancy: int
    pgr: int
    mathclk: int

class KernelMinResult(NamedTuple):
    err: int
    cuoccupancy: int
    pgr: int
    mathclk: int


def _stinky_asm_verify_wanted(isa: IsaVersion) -> bool:
    """Return True if asm/.o Stinky size check should run for this kernel.

    Requires ``CheckASMCodeSize`` and gfx1250. When True, callers should avoid joblib for
    write+assemble so logs stay on one process.
    """
    return bool(globalParameters["CheckASMCodeSize"]) and isaToGfx(isa) == "gfx1250"


def _stinky_out(msg: str) -> None:
    """Emit one user-visible log line for Stinky verify.

    Writes to stderr via ``os.write(2, ...)``. Under pytest-xdist, worker stdout may be
    hidden; stderr often still appears in the terminal.
    """
    try:
        os.write(2, (msg + "\n").encode("utf-8", errors="replace"))
    except OSError:
        pass


def _verify_stinky_asm_comment_vs_elf_text(s_path: Path, o_path: Path, kernel_base: str) -> None:
    """After assembling ``s_path`` → ``o_path``, verify Stinky vs ELF ``.text``.

    Call only when ``_stinky_asm_verify_wanted(isa)`` is True. Uses ``verify_stinky_comment_vs_elf_text``
    (``readelf`` / ``llvm-readelf``; ``ROCM_PATH``, ``LLVM_BIN``, or ``PATH``). Forwards messages
    through ``_stinky_out``. Exits via ``printExit`` on mismatch (1) or tool/readelf error (2).

    Args:
        s_path: Path to the generated ``.s`` file.
        o_path: Path to the assembled ``.o`` file.
        kernel_base: Short name for messages (usually the asm stem).
    """
    _stinky_out(f"CheckASMCodeSize: running verify for {kernel_base}")
    try:
        code, out_s, err_s = verify_stinky_paths(s_path, o_path)
    except Exception as ex:
        printExit(f"CheckASMCodeSize: could not run verifier for {kernel_base}: {ex}")
    if out_s:
        for line in out_s.splitlines():
            _stinky_out(line)
    if err_s:
        for line in err_s.splitlines():
            _stinky_out(line)
    if code == 2:
        printExit(f"CheckASMCodeSize: verifier error for {kernel_base}")
    if code == 1:
        printExit(
            f"CheckASMCodeSize: STINKY_TOTAL_INST_BYTES vs ELF .text mismatch for {kernel_base}"
        )
    if code == 0:
        out = (out_s or "") + (err_s or "")
        matched = "OK STINKY" in out
        if matched:
            _stinky_out(
                f"CheckASMCodeSize: OK STINKY_TOTAL_INST_BYTES vs ELF .text match for {kernel_base}"
            )


def memCompress(obj):
    return zlib.compress(pickle.dumps(obj))

def memDecompress(byt):
    return pickle.loads(zlib.decompress(byt))

def processKernelSource(kernelWriterAssembly, data, outOptions, splitGSU, kernel, compress = False) -> KernelCodeGenResult:
    """
    Generate source for a single kernel.
    Returns (error, source, header, kernelName).
    """
    kernelWriter = kernelWriterAssembly
    kernelWriter.setRocIsa(data, outOptions)
    asmFilename = getKernelFileBase(splitGSU, kernel)
    err, src = kernelWriter.getSourceFileString(kernel)
    if compress:
        src = memCompress(src)
    header = kernelWriter.getHeaderFileString(kernel)
    objFilename = kernel._state.get("codeObjectFile", None)
    pgr = int(kernel["PrefetchGlobalRead"])
    cuocc = kernel["CUOccupancy"]
    if cuocc <= 0 and getVerbosity() >= 2:
        print2(
            f"[codegen] CUOccupancy={cuocc} (<=0) after codegen for kernel {asmFilename}; "
            f"runtime will clamp to 1."
        )
    return KernelCodeGenResult(
        err, src, header, asmFilename, objFilename, tuple(kernel["ISA"]), \
        kernel["WavefrontSize"], cuocc, \
        pgr, kernel["MathClocksUnrolledLoop"]
    )

def _checkInvalidSolutionsAndKernels(errorTolerant, result, kernel):
    if result.err != 0:
        if not errorTolerant:
            print(
                "\nKernel generation failed for kernel: {}".format(
                    kernel["SolutionIndex"]
                )
            )
            print(kernel["SolutionNameMin"])
        return True
    return False

def _checkInvalidSolutions(splitGSU, removeKernelNames, solutions):
    invalids = []
    for solution in solutions:
        solutionKernels = solution.getKernels()
        for kernel in solutionKernels:
            kName = getKeyNoInternalArgs(kernel, splitGSU)
            if kName in removeKernelNames:
                invalids.append(True)
                break
        invalids.append(False)
    return invalids

def removeInvalidSolutionsAndKernels(results, kernels, solutions, errorTolerant, printLevel: bool, splitGSU: bool):
    removeKernelsAndResultsFlag = ParallelMap2(functools.partial(_checkInvalidSolutionsAndKernels, errorTolerant),
                                               zip(results, kernels), "check invalid kernels and results", return_as="list")

    if any(removeKernelsAndResultsFlag) and not errorTolerant:
        printExit("** kernel generation failure **")

    removeKernelNames = {getKeyNoInternalArgs(kernel, splitGSU) for invalid, kernel in zip(removeKernelsAndResultsFlag, kernels) if invalid}
    kernels[:] = [kernel for invalid, kernel in zip(removeKernelsAndResultsFlag, kernels) if not invalid]

    removeSolutionsFlag = []
    for solution in (
        tqdm(solutions, "Finding invalid solutions")
        if printLevel > 1
        else solutions
    ):
        solutionKernels = solution.getKernels()
        flag = False
        for kernel in solutionKernels:
            kName = getKeyNoInternalArgs(kernel, splitGSU)
            if kName in removeKernelNames:
                flag = True
                break
        removeSolutionsFlag.append(flag)

    solutions[:] = [solut for invalid, solut in zip(removeSolutionsFlag, solutions) if not invalid]
    results[:] = [rel for invalid, rel in zip(removeKernelsAndResultsFlag, results) if not invalid]

def passPostKernelInfoToSolution(results, kernels, solutions, splitGSU: bool):
    resultDict = {}
    for kernIdx, r in enumerate(results):
        kName = getKernelNameMin(kernels[kernIdx], splitGSU)
        resultDict["%s"%kName] = r
    for solution in solutions:
        solutionKernels = solution.getKernels()
        for kernel in solutionKernels:
            kName = getKernelNameMin(kernel, splitGSU)
            result = resultDict["%s"%kName]
            solution._state["CUOccupancy"] = result.cuoccupancy
            solution._state["PrefetchGlobalRead"] = result.pgr
            solution._state["MathClocksUnrolledLoop"] = result.mathclk

def passPostKernelInfoToLibrary(results, kernels, masterLibraries, splitGSU: bool):
    resultDict = {}
    for kernIdx, r in enumerate(results):
        kName = getKernelFileBase(splitGSU, kernels[kernIdx])
        resultDict["%s"%kName] = r
    for archName, masterLibrary in masterLibraries.items():
        for solIdx, sol in masterLibrary.solutions.items():
            solutionKernels = sol.originalSolution.getKernels()
            for kernel in solutionKernels:
                kName = getKernelFileBase(splitGSU, kernel)
                try:
                    result = resultDict["%s"%kName]
                    sol.sizeMapping.CUOccupancy = result.cuoccupancy
                    sol.sizeMapping.MathClocksUnrolledLoop = result.mathclk
                    sol.sizeMapping.PrefetchGlobalRead = sol.originalSolution._state['PrefetchGlobalRead']
                    sol.sizeMapping.NonTemporalA = sol.originalSolution._state['NonTemporalA']
                    sol.sizeMapping.NonTemporalB = sol.originalSolution._state['NonTemporalB']
                    sol.sizeMapping.adaptiveGemmNTAB = sol.originalSolution._state.get('AdaptiveGemmNTAB', 0)
                    sol.sizeMapping.NonTemporalD = sol.originalSolution._state['NonTemporalD']
                    sol.sizeMapping.WaveSeparateGlobalReadA = sol.originalSolution._state['WaveSeparateGlobalReadA']
                    sol.sizeMapping.WaveSeparateGlobalReadB = sol.originalSolution._state['WaveSeparateGlobalReadB']
                    sol.sizeMapping.UnrollLoopSwapGlobalReadOrder = sol.originalSolution._state['UnrollLoopSwapGlobalReadOrder']
                    sol.sizeMapping.DirectToVgprA = bool(sol.originalSolution._state['DirectToVgprA'])
                    sol.sizeMapping.DirectToVgprB = bool(sol.originalSolution._state['DirectToVgprB'])
                except KeyError:
                    print(f"\n{'='*80}")
                    print(f"ERROR: KeyError in masterLibrary.solutions")
                    print(f"Architecture: {archName}")
                    print(f"Solution Index: {solIdx}")
                    print(f"Solution source file: {getattr(sol, 'srcName', 'Unknown')}")
                    print(f"Solution library logic index: {getattr(sol, 'libraryLogicIndex', 'Unknown')}")
                    print(f"Missing kernel name: {kName}")
                    print(f"{'='*80}\n")
                    raise
        masterLibrary.lazyLibraries = dict(sorted(masterLibrary.lazyLibraries.items()))
        for name, lib in masterLibrary.lazyLibraries.items():
            for solIdx, sol in lib.solutions.items():
                solutionKernels = sol.originalSolution.getKernels()
                for kernel in solutionKernels:
                    kName = getKernelFileBase(splitGSU, kernel)
                    try:
                        result = resultDict["%s"%kName]
                        sol.sizeMapping.CUOccupancy = result.cuoccupancy
                        sol.sizeMapping.MathClocksUnrolledLoop = result.mathclk
                        sol.sizeMapping.PrefetchGlobalRead = sol.originalSolution._state['PrefetchGlobalRead']
                        sol.sizeMapping.NonTemporalA = sol.originalSolution._state['NonTemporalA']
                        sol.sizeMapping.NonTemporalB = sol.originalSolution._state['NonTemporalB']
                        sol.sizeMapping.adaptiveGemmNTAB = sol.originalSolution._state.get('AdaptiveGemmNTAB', 0)
                        sol.sizeMapping.NonTemporalD = sol.originalSolution._state['NonTemporalD']
                        sol.sizeMapping.WaveSeparateGlobalReadA = sol.originalSolution._state['WaveSeparateGlobalReadA']
                        sol.sizeMapping.WaveSeparateGlobalReadB = sol.originalSolution._state['WaveSeparateGlobalReadB']
                        sol.sizeMapping.UnrollLoopSwapGlobalReadOrder = sol.originalSolution._state['UnrollLoopSwapGlobalReadOrder']
                        sol.sizeMapping.DirectToVgprA = bool(sol.originalSolution._state['DirectToVgprA'])
                        sol.sizeMapping.DirectToVgprB = bool(sol.originalSolution._state['DirectToVgprB'])
                    except KeyError:
                        print(f"\n{'='*80}")
                        print(f"ERROR: KeyError in lazyLibrary")
                        print(f"Architecture: {archName}")
                        print(f"LazyLibrary name: {name}")
                        print(f"Solution Index: {solIdx}")
                        print(f"Solution source file: {getattr(sol, 'srcName', 'Unknown')}")
                        print(f"Solution library logic index: {getattr(sol, 'libraryLogicIndex', 'Unknown')}")
                        print(f"Missing kernel name: {kName}")
                        print(f"Total kernels in this solution: {len(solutionKernels)}")
                        print(f"{'='*80}\n")
                        raise

def writeAssembly(asmPath: Union[Path, str], result: KernelCodeGenResult):
    if result.err:
        printExit(f"Failed to build kernel {result.name} because it has error code {result.err}")
    path = Path(asmPath) / f"{result.name}.s"
    isa = result.isa
    wfsize = result.wavefrontSize
    with open(path, "w", encoding="utf-8") as f:
        src = result.src
        if isinstance(src, bytes):
            src = memDecompress(src)
        f.write(src)

    minResult = KernelMinResult(result.err, result.cuoccupancy, result.pgr, result.mathclk)
    return path, isa, wfsize, minResult

def writeHelpers(
    outputPath, kernelHelperObjs, KERNEL_HELPER_FILENAME_CPP, KERNEL_HELPER_FILENAME_H
):
    kernelSourceFilename = os.path.join(os.path.normcase(outputPath), KERNEL_HELPER_FILENAME_CPP)
    kernelHeaderFilename = os.path.join(os.path.normcase(outputPath), KERNEL_HELPER_FILENAME_H)

    with open(kernelHeaderFilename, "w", encoding="utf-8") as kernelHeaderFile, open(
        kernelSourceFilename, "w", encoding="utf-8"
    ) as kernelSourceFile:
        kernelSourceFile.write(CHeader)
        kernelHeaderFile.write(CHeader)
        kernelSourceFile.write('#include "Kernels.h"\n')
        kernelHeaderFile.write("#pragma once\n")
        kernelHeaderFile.write("#include <hip/hip_runtime.h>\n")
        kernelHeaderFile.write("#include <hip/hip_ext.h>\n\n")
        kernelHeaderFile.write('#include "KernelHeader.h"\n\n')
        HeaderText = ""
        for ko in kernelHelperObjs:
            kernelName = ko.getKernelName()
            (err, src) = ko.getSourceFileString()

            kernelSourceFile.write(src)
            if err:
                print("*** warning: invalid kernel#%u" % kernelName)
            HeaderText += ko.getHeaderFileString()
        kernelHeaderFile.write(HeaderText)


def writeSolutionsAndKernels(
    outputPath,
    asmToolchain,
    srcToolchain,
    solutions,
    kernels,
    kernelHelperObjs,
    kernelWriterAssembly,
    splitGSU: bool,
    cmdlineArchs: List[str],
    disableAsmComments: bool=False,
    errorTolerant: bool=False,
    generateSourcesAndExit: bool=False,
    compress: bool=True,
    removeTemporaries: bool=True,
):
    if globalParameters["PythonProfile"]:
        globalParameters["CpuThreads"] = 0
        printWarning("Python profiling is enabled. CpuThreads set to 0.")
        import yappi
        yappi.start()

    codeObjectFiles = []

    archNames = archNamesByIsa(cmdlineArchs)

    with timing_context("python_kernel_setup"):
        outputPath = Path(outputPath)
        # Builders create <destRoot>/<arch>/ as they write, so this is not about
        # racing them; it is so an architecture that emits nothing still leaves a
        # subdirectory behind rather than a gap in the tree.
        destRoot = ensurePath(libraryRoot(outputPath))
        for base in _baseArchs(cmdlineArchs):
            ensurePath(libraryDir(outputPath, base))
        # Through buildTmpDir like every other scratch user, so the "one namer of
        # the scratch directory" rule holds by construction. Tuning never fans
        # out -- it rejects an ISA it cannot name rather than splitting -- so
        # this path only ever sees one group, and the name it gets back is the
        # one this directory always had unless a stepping was asked for.
        buildTmpPath = ensurePath(buildTmpDir(outputPath, cmdlineArchs))
        assemblyTmpPath = ensurePath(
            buildTmpPath / "assembly"
        )  # Temp path for generated assembly files (.s)
        objectTmpPath = ensurePath(
            buildTmpPath / "code_object_tmp"
        )  # Temp path for HSA code object files (.hsaco)

        asmKernels = [k for k in kernels if k["KernelLanguage"] == "Assembly"]

        visited = set()
        duplicates = 0
        for k in asmKernels:
            base = getKernelFileBase(splitGSU, k)
            k.duplicate = True if base in visited else False
            if not k.duplicate:
                k["BaseName"] = base
            duplicates += k.duplicate
            print2(f"Duplicate: {base}")
            visited.add(base)
        print1(f"Number of duplicate kernels: {duplicates}")

        outOptions = rocisa.rocIsa.getInstance().getOutputOptions()
        outOptions.outputNoComment = disableAsmComments

        numAsmKernels = len(asmKernels)
        numKernels = len(asmKernels)
        assert numKernels == numAsmKernels, "Only assembly kernels are supported in TensileLite"
        asmIter = zip(
            itertools.repeat(kernelWriterAssembly),
            itertools.repeat(rocisa.rocIsa.getInstance().getData()),
            itertools.repeat(outOptions),
            itertools.repeat(splitGSU),
            asmKernels
        )
        memcompress = numAsmKernels > 10000
    with timing_context("python_kernel_codegen"):
        asmResults = ParallelMap2(functools.partial(processKernelSource, compress=memcompress), asmIter, "Generating assembly kernels", return_as="list")
    with timing_context("python_kernel_validate"):
        removeInvalidSolutionsAndKernels(
            asmResults, asmKernels, solutions, errorTolerant, getVerbosity(), splitGSU
        )
        passPostKernelInfoToSolution(
            asmResults, asmKernels, solutions, splitGSU
        )

    def assemble(ret):
        p, isa, wavefrontsize, _ = ret
        o_path = p.with_suffix(".o")
        try:
            asmToolchain.assembler(archNames.get(isa) or isaToGfx(isa), wavefrontsize, str(p), str(o_path))
        except RuntimeError as e:
            printWarning(f"Failed to assemble {p}: {e}")
            return
        if _stinky_asm_verify_wanted(isa):
            _verify_stinky_asm_comment_vs_elf_text(p, o_path, p.stem)
        if removeTemporaries:
            p.unlink()

    unaryWriteAssembly = functools.partial(writeAssembly, assemblyTmpPath)
    compose = lambda *F: functools.reduce(lambda f, g: lambda x: f(g(x)), F)
    with timing_context("python_kernel_write_assemble"):
        ret = ParallelMap2(
            compose(assemble, unaryWriteAssembly),
            asmResults,
            "Writing assembly kernels",
            return_as="list",
            multiArg=False,
        )

    # Remove solutions whose kernels failed to assemble (no .o file produced)
    failedBases = {k["BaseName"] for k in asmKernels
                   if not k.duplicate and not (assemblyTmpPath / (k["BaseName"] + ".o")).exists()}
    if failedBases:
        solutions[:] = [s for s in solutions
                        if getKernelFileBase(splitGSU, s.getKernels()[0]) not in failedBases]
        asmKernels[:] = [k for k in asmKernels if k.get("BaseName", None) not in failedBases]

    with timing_context("python_kernel_write_helpers"):
        writeHelpers(outputPath, kernelHelperObjs, KERNEL_HELPER_FILENAME_CPP, KERNEL_HELPER_FILENAME_H)
    srcKernelFile = Path(outputPath) / "Kernels.cpp"

    if globalParameters["PythonProfile"]:
        yappi.stop()
        yappi.get_func_stats().save("yappi_results.profile", type="callgrind")
        with open("yappi_results.txt", "w") as f:
            yappi.get_func_stats().print_all(out=f)
        if globalParameters["CpuThreads"] != 0:
            with open("yappi_thread_stats.txt", "w") as f:
                yappi.get_thread_stats().print_all(out=f)

    if not generateSourcesAndExit:
        with timing_context("python_kernel_build_co"):
            codeObjectFiles += buildAssemblyCodeObjectFiles(
                asmToolchain.linker,
                asmToolchain.bundler,
                [k for k in asmKernels if not k.duplicate],
                destRoot,
                assemblyTmpPath,
                compress,
                archNames=archNames,
            )

        with timing_context("python_kernel_build_src_co"):
            buildSourceCodeObjectFiles(
                srcToolchain.compiler,
                srcToolchain.bundler,
                destRoot,
                objectTmpPath,
                outputPath,
                srcKernelFile,
                cmdlineArchs,
            )

    if removeTemporaries and not generateSourcesAndExit:
        # Same rule as the other scratch cleanup: a fan-out child shares the
        # parent with a sibling still writing into it and may only take its own
        # subdirectory, while every other run owns the tree and clears it whole.
        # The parent is named from outputPath rather than from this run's own
        # scratch path, so no leaf name can move what gets removed.
        scratchRoot = buildTmpRoot(outputPath)
        if buildTmpPath.is_dir():
            if os.environ.get(_GROUP_BUILD_ENV):
                shutil.rmtree(buildTmpPath)
                try:
                    scratchRoot.rmdir()
                except OSError:
                    pass
            else:
                shutil.rmtree(scratchRoot)

    return codeObjectFiles, numKernels


def writeSolutionsAndKernelsTCL(
    outputPath,
    asmToolchain,
    srcToolchain,
    solutions,
    kernels,
    kernelHelperObjs,
    kernelWriterAssembly,
    cmdlineArchs: List[str],
    disableAsmComments: bool=False,
    compress: bool=True,
    removeTemporaries: bool=True,
):
    archNames = archNamesByIsa(cmdlineArchs)
    outputPath = Path(outputPath)
    # Builders create <destRoot>/<arch>/ as they write, so this is not about
    # racing them. The master and mapping writes later iterate the full requested
    # list, so every architecture needs its subdirectory even if it emitted no
    # kernels at all.
    destRoot = ensurePath(libraryRoot(outputPath))
    for base in _baseArchs(cmdlineArchs):
        ensurePath(libraryDir(outputPath, base))
    buildTmpPath = ensurePath(buildTmpDir(outputPath, cmdlineArchs))
    assemblyTmpPath = ensurePath(
        buildTmpPath / "assembly"
    )  # Temp path for generated assembly files (.s)
    objectTmpPath = ensurePath(
        buildTmpPath / "code_object_tmp"
    )  # Temp path for HSA code object files (.hsaco)
    # The source kernels compile against these, so they belong beside the
    # Kernels.cpp written below rather than in the output root a concurrent run
    # also writes. HelperKernelCache keys on their contents read from this same
    # directory, so the three have to travel together.
    copyStaticFiles(buildTmpPath)

    asmKernels = [k for k in kernels if k["KernelLanguage"] == "Assembly"]

    visited = set()
    duplicates = 0
    splitGSU = False
    for k in asmKernels:
        base = getKernelFileBase(splitGSU, k)
        k["BaseName"] = base
        k.duplicate = True if base in visited else False
        duplicates += k.duplicate
        print2(f"Duplicate: {base}")
        visited.add(base)
    print1(f"Number of duplicate kernels: {duplicates}")

    uniqueAsmKernels = [k for k in asmKernels if not k.duplicate]

    def assemble(ret, removeTemporaries: bool):
        asmPath, isa, wavefrontsize, result = ret
        o_path = asmPath.with_suffix(".o")
        asmToolchain.assembler(archNames.get(isa) or isaToGfx(isa), wavefrontsize, str(asmPath), str(o_path))
        if _stinky_asm_verify_wanted(isa):
            _verify_stinky_asm_comment_vs_elf_text(asmPath, o_path, asmPath.stem)
        if removeTemporaries:
            asmPath.unlink()
        return result

    unaryAssemble = functools.partial(assemble, removeTemporaries=removeTemporaries)

    outOptions = rocisa.rocIsa.getInstance().getOutputOptions()
    outOptions.outputNoComment = not disableAsmComments

    memcompress = len(uniqueAsmKernels) > 10000
    unaryProcessKernelSource = functools.partial(
        processKernelSource,
        kernelWriterAssembly,
        rocisa.rocIsa.getInstance().getData(),
        outOptions,
        splitGSU,
        compress = memcompress,
    )

    unaryWriteAssembly = functools.partial(writeAssembly, assemblyTmpPath)
    def compose(assemble, unaryWriteAssembly, unaryProcessKernelSource):
        def composed_function(kernel):
            processed_kernel = unaryProcessKernelSource(kernel)
            written_kernel = unaryWriteAssembly(processed_kernel)
            assembled_kernel = assemble(written_kernel)
            return processed_kernel
        return composed_function

    results = ParallelMap2(
        compose(unaryAssemble, unaryWriteAssembly, unaryProcessKernelSource),
        uniqueAsmKernels,
        "Generating assembly kernels",
        multiArg=False,
        return_as="list"
    )

    buildAssemblyCodeObjectFiles(
        asmToolchain.linker,
        asmToolchain.bundler,
        asmKernels,
        destRoot,
        assemblyTmpPath,
        compress,
        archNames=archNames,
    )

    # The basename has to stay "Kernels": the emitted code object is named from
    # it, and the runtime opens Kernels.so-000-<arch>.hsaco by that name.
    writeHelpers(buildTmpPath, kernelHelperObjs, KERNEL_HELPER_FILENAME_CPP, KERNEL_HELPER_FILENAME_H)
    srcKernelFile = buildTmpPath / "Kernels.cpp"

    buildSourceCodeObjectFiles(
        srcToolchain.compiler,
        srcToolchain.bundler,
        destRoot,
        objectTmpPath,
        buildTmpPath,
        srcKernelFile,
        cmdlineArchs,
    )

    return len(uniqueAsmKernels), uniqueAsmKernels, results


@timing
def copyStaticFiles(outputPath):
    return copy_static_headers(outputPath)


@timing
def generateKernelObjectsFromSolutions(solutions):
    kernels = []
    kernelNames = set()
    for solution in solutions:
        solutionKernels = solution.getKernels()
        for kernel in solutionKernels:
            kName = getKeyNoInternalArgs(kernel, False)
            if kName not in kernelNames:
                kernels.append(kernel)
                kernelNames.add(kName)
    return kernels


def generateKernelHelperObjects(solutions: List[Solution], cxxCompiler: str, isaInfoMap):
    """
    Generates a unique list of kernel helpers.

    Kernel helpers are used to generate hip source code kernels called
    before/after gemm kernels. This function creates a minimal list of
    kernel helpers required to support the solutions reuested in a build.
    The list of kernel helpers is then used to write Kernels.cpp/h to
    disk. To ensure the ActivationEnumHeaders are written first, the
    list is sorted such that those kernel helpers appear first.

    Args:
        solutions: a list of solutions to process.
        cxxCompiler: the full path to the cxxCompiler.

    Returns:
        List of kernel helpers.
    """
    khos = []
    visited = set()
    for solution in solutions:
        for kernelHelperType, callable in kernelObjectNameCallables():
            buildMask = []
            names = callable(solution)
            if names:
                sortByEnum = lambda x: ("Enum" in x, names.index(x))
                names = sorted(names, key=sortByEnum, reverse=True)
                for name in names:
                    if name not in visited:
                        visited.add(name)
                        buildMask.append(True)
                    else:
                        buildMask.append(False)
                if any(buildMask):
                    kho = initHelperKernelObjects(solution, kernelHelperType, cxxCompiler, isaInfoMap)
                    kho = list(itertools.compress(kho, buildMask))
                    if kho:
                        khos.extend(kho)
    # fromkeys, not set: these are written into Kernels.cpp in this order, and
    # KernelWriterBase hashes its string form, so a set would order them by a
    # salted string hash.
    khos = list(dict.fromkeys(khos))
    sortByEnum = lambda x: ("Enum" in x.getKernelName(), khos.index(x))
    return sorted(khos, key=sortByEnum, reverse=True) # Ensure that we write Enum kernel helpers are first in list


def _renameFallbackPlaceholders(node, arch: str) -> None:
    """Walk a library tree, appending "_<arch>" to fallback PlaceholderLibrary names.

    Mutates `filenamePrefix` on every PlaceholderLibrary leaf whose existing
    prefix already encodes a fallback (i.e. came from the merged-in fallback
    master library and therefore ends with "_fallback") and which has not yet
    been arch-suffixed. Idempotent: a prefix already ending in "_<arch>"
    is left alone so a second pass cannot double-suffix.
    """
    if node is None:
        return
    if isinstance(node, PlaceholderLibrary):
        if "_fallback" in node.filenamePrefix and not node.filenamePrefix.endswith("_" + arch):
            node.filenamePrefix = node.filenamePrefix + "_" + arch
        return
    rows = getattr(node, "rows", None)
    if rows:
        for row in rows:
            _renameFallbackPlaceholders(row.get("library"), arch)
    mapping = getattr(node, "mapping", None)
    if mapping:
        for child in mapping.values():
            _renameFallbackPlaceholders(child, arch)


def renameFallbacksPerArch(masterLibraries) -> None:
    """Make merged-in fallback lazy-library filenames arch-specific.

    `MasterSolutionLibrary.merge` aliases the same fallback lazy library across
    every per-arch master that absorbs it (keys collide on the un-suffixed
    "_fallback" name). That alias means the per-arch *_fallback.dat files are
    written with overlapping filenames carrying different solution-index spaces,
    and the per-arch Mapping write loop's `name.endswith("_<arch>")` filter drops
    every fallback entry — runtime then can't resolve fallback-served dtypes.

    Per-arch deep-copy here splits the alias and arch-suffixes both the
    `lazyLibraries` dict keys and the matching PlaceholderLibrary nodes inside
    the master library tree, so:
      - on-disk filenames diverge (no overlay collision),
      - the per-arch Mapping filter matches "_fallback_<arch>" naturally, and
      - each arch keeps its own re-indexed copy of the fallback solutions.
    """
    for arch in list(masterLibraries.keys()):
        master = copy.deepcopy(masterLibraries[arch])
        masterLibraries[arch] = master
        renamed = {}
        for name, lib in master.lazyLibraries.items():
            if "_fallback" in name and not name.endswith("_" + arch):
                renamed[name + "_" + arch] = lib
            else:
                renamed[name] = lib
        master.lazyLibraries = renamed
        _renameFallbackPlaceholders(master.library, arch)


@timing
def generateLogicDataAndSolutions(logicFiles, args, assembler: Assembler, isaInfoMap):

    if ";" in args["Architecture"]:
        archs = args["Architecture"].split(";")  # user arg list format
    else:
        archs = args["Architecture"].split("_")  # workaround for cmake list in list issue

    solutions = []
    masterLibraries = {}
    nextSolIndex = 0
    splitGSU = False
    printSolutionRejectionReason = True
    printIndexAssignmentInfo = False

    fIter = zip(
        logicFiles,
        itertools.repeat(assembler),
        itertools.repeat(splitGSU),
        itertools.repeat(printSolutionRejectionReason),
        itertools.repeat(printIndexAssignmentInfo),
        itertools.repeat(isaInfoMap),
        itertools.repeat(args["LazyLibraryLoading"]),
    )

    def libraryIter(lib: MasterSolutionLibrary):
        if len(lib.solutions):
            for i, s in enumerate(lib.solutions.items()):
                yield (i, *s)
        else:
            for _, lazyLib in lib.lazyLibraries.items():
                yield from libraryIter(lazyLib)

    def mergeTypeMismatchSnapshot(
        aggregate: dict,
        snapshot: dict,
    ) -> None:
        """Merge one parseLibraryLogicFile() mismatch snapshot into aggregate."""
        for key, entry in snapshot.items():
            target = aggregate.setdefault(key, {"count": 0, "values": set(), "files": set()})
            target["count"] += entry["count"]
            target["values"].update(entry["values"])
            target["files"].update(entry["files"])

    # parseLibraryLogicData() uses Solution.py's module-level type mismatch
    # collector as a per-file scratch buffer: it resets the collector at parse
    # start, builds the ProblemType/Solution objects, and returns a snapshot in
    # the LibraryLogic tuple. ParallelMap2 can execute those parses in worker
    # processes or in this process (notably when CpuThreads resolves to 1). In
    # the in-process path, the scratch buffer is the same global collector this
    # function can see, and after parsing it still contains the most recently
    # parsed file's mismatches.
    #
    # Keep the run-level aggregate detached from that global collector while
    # consuming parse results. Merging returned snapshots directly into the live
    # collector would mix aggregate state with parser scratch state: a later
    # parse can reset previously merged aggregate state, and the final file's
    # snapshot can be double-counted. Once every logic file has been parsed,
    # replace the global collector with this clean aggregate so
    # raiseIfTypeMismatches() can format the existing fatal aggregate error.
    typeMismatchAggregate: dict = {}
    parsedLibraries = ParallelMap2(
        LibraryIO.parseLibraryLogicFile, fIter, "Loading Logics...", return_as="generator"
    )
    for library in parsedLibraries:
        _, architectureName, _, _, _, newLibrary, typeMismatches = library
        mergeTypeMismatchSnapshot(typeMismatchAggregate, typeMismatches)

        if architectureName == "":
            continue

        if architectureName in masterLibraries:
            nextSolIndex = masterLibraries[architectureName].merge(newLibrary, nextSolIndex)
        else:
            masterLibraries[architectureName] = newLibrary
            masterLibraries[architectureName].version = args["CodeObjectVersion"]

    # After all YAML files have been parsed and Solution objects created,
    # fail on any type mismatches that were collected.
    resetTypeMismatchCollector()
    mergeTypeMismatchCollector(typeMismatchAggregate)
    raiseIfTypeMismatches()

    # Sort masterLibraries to make global soln index values deterministic
    solnReIndex = 0
    masterLibraries = dict(sorted(masterLibraries.items()))
    for _, masterLibrary in masterLibraries.items():
        for _, sol in masterLibrary.solutions.items():
            sol.index = solnReIndex
            solnReIndex += 1
        # Sort masterLibrary to make global soln index values deterministic
        masterLibrary.lazyLibraries = dict(sorted(masterLibrary.lazyLibraries.items()))
        for name, lib in masterLibrary.lazyLibraries.items():
            # Sort solns by the lib logic file they were generated from
            lib.solutions = {
                k: lib.solutions[k]
                for k in sorted(lib.solutions, key=lambda idx: lib.solutions[idx].srcName)
            }
            for _, sol in lib.solutions.items():
                sol.index = solnReIndex
                solnReIndex += 1

    if args["GenSolTable"]:
        matchTable = {}
        # Match yaml file solutions to solution index
        for _, masterLibrary in masterLibraries.items():
            for _, _, s in libraryIter(masterLibrary):
                matchTable[s.index] = [s.srcName, s.libraryLogicIndex]
        LibraryIO.write("MatchTable", matchTable)

    fallbackAdded = "fallback" in masterLibraries.keys()
    if fallbackAdded:
        for key, value in masterLibraries.items():
            if key != "fallback":
                value.merge(masterLibraries["fallback"])
        masterLibraries.pop("fallback")
    if fallbackAdded:
        # Must run AFTER merge (so per-arch masters carry their own fallback
        # entries) and BEFORE the codeObjectFile-assignment loop below (which
        # snapshots the dict key as the on-disk filename for each solution).
        renameFallbacksPerArch(masterLibraries)
    solIndex = []
    for _, masterLibrary in masterLibraries.items():
        for _, sol in masterLibrary.solutions.items():
            solutions.append(sol.originalSolution)
            solIndex.append(sol.index)
        for name, lib in masterLibrary.lazyLibraries.items():
            for _, sol in lib.solutions.items():
                sol.originalSolution._state["codeObjectFile"] = name
                solutions.append(sol.originalSolution)
                solIndex.append(sol.index)

    # Get the solution index and it's codeObjectFile name
    codeObjectFilesIndex = {}
    for solution, index in zip(solutions, solIndex):
        if "codeObjectFile" in solution._state and solution._state["codeObjectFile"] is not None:
            if solution._state["codeObjectFile"] in codeObjectFilesIndex:
                codeObjectFilesIndex[solution._state["codeObjectFile"]] = min(index, codeObjectFilesIndex[solution._state["codeObjectFile"]])
            else:
                codeObjectFilesIndex[solution._state["codeObjectFile"]] = index

    # Reorder to int: name format
    codeObjectFilesIndex = {v: k for k, v in codeObjectFilesIndex.items()}
    # Reorder to maintain ascending order by index
    codeObjectFilesIndex = dict(sorted(codeObjectFilesIndex.items()))

    # remove duplicates while preserving order
    numSoln = len(solutions)
    solutions = dict.fromkeys(solutions).keys()

    print1(f"Number of solutions parsed: {numSoln}")
    print1(f"Number of unique solutions: {len(solutions)}")

    return solutions, masterLibraries, codeObjectFilesIndex


################################################################################
# Tensile Create Library
################################################################################
def _cpuCount() -> int:
    if os.name == "nt":
        # Matches CPUThreadCount: the Windows scheduler caps waitable handles.
        return min(os.cpu_count() or 1, 61)
    return len(os.sched_getaffinity(0))


def _jobsPerGroup(requested: int, groups: List[List[str]]) -> List[int]:
    """This run's requested parallelism, shared out among the groups.

    The groups build concurrently, so passing the count through unchanged would
    spend it per group rather than across them.

    Shared in proportion to each group's architecture count, not evenly: a second
    group exists because one stepping collided, leaving it holding one
    architecture and the other holding the rest. An even split would run the
    large group at half speed and idle half the machine once the small one ends.

    ``requested`` is the parsed ``CpuThreads``, not a re-scan of argv, which
    would have to enumerate the spellings argparse accepts (``-j=8``, any
    unambiguous abbreviation of ``--jobs``) to avoid silently dropping them.
    """
    # 0 disables threading and has to survive the split intact; -1 is the
    # documented "use every CPU" default.
    if requested == 0:
        return [0] * len(groups)
    total = _cpuCount() if requested < 0 else min(_cpuCount(), requested)
    archCount = sum(len(g) for g in groups) or 1
    return [max(1, total * len(g) // archCount) for g in groups]


def _argvForArchitectures(argv: List[str], specs: List[str], jobs: int) -> List[str]:
    """This invocation's arguments, retargeted at specs.

    Everything else is passed through untouched, so a group is built exactly as
    the caller asked, only narrower.

    The canonical spellings are dropped rather than left for the appended ones
    to override, so the child's command line does not read as asking for two
    architectures. argparse also accepts abbreviations (``--arch``) and ``-j=8``,
    which this does not catch; those survive and are settled by last-wins, which
    is correct because both options are plain ``store``.
    """
    kept = []
    dropValue = False
    for token in argv:
        if dropValue:
            dropValue = False
        elif token in ("--architecture", "--jobs", "-j"):
            dropValue = True
        elif token.startswith("--architecture=") or token.startswith("--jobs="):
            continue
        elif token.startswith("-j") and token[2:].lstrip("-").isdigit():
            continue
        else:
            kept.append(token)
    kept.append("--architecture=" + ";".join(specs))
    kept.append(f"--jobs={jobs}")
    return kept


def _childEnvironment() -> Dict[str, str]:
    """This process's environment, with Tensile importable by the children.

    Tensile/bin/TensileCreateLibrary puts the source root on sys.path when the
    package is not installed. A child launched with -m inherits the environment
    but not that sys.path edit, so it has to be passed as PYTHONPATH.
    """
    packageRoot = str(Path(__file__).resolve().parents[2])
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        packageRoot + os.pathsep + existing if existing else packageRoot
    )
    env[_GROUP_BUILD_ENV] = "1"
    return env


def _buildGroupsSeparately(groups: List[List[str]], requestedJobs: int) -> None:
    """Builds each group in its own process, concurrently.

    A run names its target by the ISA it parses out of the gfx name, so a
    stepping and the architecture it steps from cannot share one; see
    archNamesByIsa. Separate processes rather than a loop here because a run
    settles arch-dependent state process-wide -- the capability map is keyed by
    ISA and patched in place, and StinkyTofu's cost table is named by a global.

    The output directory is shared: each run keeps its scratch to itself
    (buildTmpDir) and their per-architecture outputs are named apart.
    """
    print1(
        f"# Architectures requested need {len(groups)} builds to cover: "
        + ", ".join("[" + ";".join(g) + "]" for g in groups)
    )
    argv = sys.argv[1:]
    jobs = _jobsPerGroup(requestedJobs, groups)
    env = _childEnvironment()
    # Spawned inside the try so that a failure to start the second one still
    # reaches the cleanup below: the first is already running, and a spawn can
    # fail for reasons that have nothing to do with the build (fork under
    # memory pressure).
    procs = []
    try:
        for group, groupJobs in zip(groups, jobs):
            procs.append((
                group,
                subprocess.Popen(
                    [sys.executable, "-m", "Tensile.TensileCreateLibrary"]
                    + _argvForArchitectures(argv, group, groupJobs),
                    env=env,
                ),
            ))
        failed = [group for group, proc in procs if proc.wait() != 0]
    finally:
        # An interrupted parent would otherwise leave these writing into the
        # output directory after whatever killed it has given up on the build.
        for _, proc in procs:
            if proc.poll() is None:
                proc.terminate()
            # Reaped rather than just signalled, or the parent exits with the
            # children still occupying process table entries.
            proc.wait()
    if failed:
        printExit(
            "Failed to build architectures: "
            + ", ".join(";".join(g) for g in failed)
        )


@profile
def run():
    start = timer()
    print1("")
    print1(HR)
    print1("# Tensile Create Library")
    print2(HR)
    print2("")

    arguments = parseArguments()
    setVerbosity(arguments["PrintLevel"])
    outputPath = Path(ensurePath(os.path.abspath(arguments["OutputPath"])))
    cxxCompiler, _, offloadBundler, _, _ = validateToolchain(
        arguments["CxxCompiler"],
        arguments["CCompiler"],
        arguments["OffloadBundler"],
        arguments["Assembler"],
        ToolchainDefaults.HIP_CONFIG,
    )

    if ";" in arguments["Architecture"]:
        archs = arguments["Architecture"].split(";")
    else:
        archs = arguments["Architecture"].split("_")

    # More than one group only when a stepping was asked for beside the
    # architecture it steps from, which no single run can name. Everything else
    # takes the one path below, unchanged.
    groups = isaCollisionFreeGroups(archs)
    if len(groups) > 1:
        _buildGroupsSeparately(groups, arguments["CpuThreads"])
        return

    archs = expandAllArchitectures(archs)
    archs, requestedPredicateMap = splitArchsFromPredicates(archs)

    targetIsas = [gfxToIsa(a) for a in archs]
    isaInfoMap = makeIsaInfoMap(targetIsas, cxxCompiler)
    applyArchCapOverrides(isaInfoMap, archs)

    assignGlobalParameters(arguments, isaInfoMap)

    # StinkyTofu selects its cost table by name, and two steppings share one ISA,
    # so the ISA-derived name would land on the base arch's table. Hand it the
    # requested name instead; "" means this build asked for none.
    #
    # One name, because the global is single-valued -- not because grouping says
    # so. Collision-free grouping only keeps two architectures sharing an ISA
    # apart, so steppings of two different base architectures would share a
    # group. gfx1250-strict is the only stepping registered today, so that pair
    # cannot arise; a second one has to revisit this.
    steppings = [baseArchName(a) for a in archs if steppingArchOf(a)]
    globalParameters["StinkyTofuArchName"] = steppings[0] if steppings else ""

    asmToolchain = makeAssemblyToolchain(
        cxxCompiler,
        offloadBundler,
        arguments["CodeObjectVersion"],
        arguments["BuildIdKind"],
        arguments["AsmDebug"],
    )
    srcToolchain = makeSourceToolchain(
        cxxCompiler,
        offloadBundler,
        arguments["AsanBuild"],
        arguments["BuildIdKind"],
        save_temps=False
    )

    print1(asmToolchain.assembler)
    print1(asmToolchain.bundler)

    if not os.path.exists(arguments["LogicPath"]):
        printExit(f"LogicPath {arguments['LogicPath']} doesn't exist")

    logicExtFormat = ".yaml"
    if arguments["LogicFormat"] == "yaml":
        pass
    elif arguments["LogicFormat"] == "json":
        logicExtFormat = ".json"
    else:
        printExit(f"Unrecognized LogicFormat: {arguments['LogicFormat']}")

    def validLogicFile(p: Path):
        if p.suffix != logicExtFormat:
            return False
        # archs came through expandAllArchitectures, which replaces "all" with
        # the architectures it covers, so there is no keyword left to honour.
        return archMatch(load_logic_gfx_arch(p), archs)

    globPattern = os.path.join(
        arguments["LogicPath"], f"**/{arguments['LogicFilter']}{logicExtFormat}"
    )
    print1(f"# LogicFilter:       {globPattern}")
    # Sorted: glob yields readdir order, which differs between checkouts and
    # machines, and this order sets the solution indices written to the library.
    logicFiles = sorted(glob.iglob(globPattern, recursive=True))

    logicFiles = [file for file in logicFiles if validLogicFile(Path(file))]

    print1(f"# Experimental:      {arguments['Experimental']}")
    if not arguments["Experimental"]:
        logicFiles = [
            file for file in logicFiles if "experimental" not in map(str.lower, Path(file).parts)
        ]

    print1("# Archs: " + ', '.join(archs))
    if requestedPredicateMap:
        print1("# Predicates:\n" + "\n".join(f"#   {arch}: {', '.join(v) if v else 'all variants'}" for arch, v in requestedPredicateMap.items()))
        numPrior = len(logicFiles)
        logicFiles = filterLogicFilesByPredicates(logicFiles, requestedPredicateMap)
        print1(f"# Filtered {numPrior - len(logicFiles)} logic files not matching requested predicates")

    print1(f"# LibraryLogicFiles: {len(logicFiles)}")

    if not logicFiles:
        # Still exits 0 and still writes the subtree: a build that legitimately
        # filters down to nothing is not this function's to reject. What this
        # buys is a named cause, because the failure otherwise surfaces much
        # later as the runtime failing to read TensileLibrary_lazy_<arch>.dat,
        # and the only trace here is a zero that looks like every other zero.
        printWarning(
            f"No logic files matched {', '.join(archs)} under {arguments['LogicPath']}; "
            "the library will have no master and no Mapping, and every matmul will "
            "fail at runtime."
        )

    for logicFile in logicFiles:
        print2("#   %s" % logicFile)

    start_glds = timer()
    solutions, masterLibraries, libraryMapping = generateLogicDataAndSolutions(
        logicFiles, arguments, asmToolchain.assembler, isaInfoMap
    )
    stop_glds = timer()
    print(f"Time to load yaml files (s): {(stop_glds-start_glds):3.2f}")


    kernels = generateKernelObjectsFromSolutions(solutions)
    kernelHelperObjs = generateKernelHelperObjects(kernels, str(asmToolchain.assembler.path), isaInfoMap)
    kernelWriterAssembly = KernelWriterAssembly(asmToolchain.assembler, DebugConfig())

    # Resolved before the writer runs: archs is narrowed to the supported subset
    # below, and dropping the stepping there would rename this run's scratch
    # directory, leaving what the writer actually filled behind.
    buildTmpPath = buildTmpDir(outputPath, archs)

    start_wsk = timer()
    numKernels, uniqueKernels, kernelInfo = writeSolutionsAndKernelsTCL(
        outputPath,
        asmToolchain,
        srcToolchain,
        solutions,
        kernels,
        kernelHelperObjs,
        kernelWriterAssembly,
        archs,
        arguments["DisableAsmComments"],
        compress=arguments["UseCompression"],
        removeTemporaries=not arguments["KeepBuildTmp"],
    )
    stop_wsk = timer()
    print(f"Time to generate kernels (s): {(stop_wsk-start_wsk):3.2f}")

    # The names that were requested, not names rebuilt from their ISAs: a
    # stepping shares its base architecture's ISA, so rebuilding would hand back
    # the base's name and every per-arch write below would address the wrong
    # architecture. Qualifiers are dropped because the writes name directories
    # and files, which carry the bare architecture; deduplicated for the same
    # reason, so gfx942:xnack+ and gfx942:xnack- do not write gfx942 twice.
    archs = list(
        dict.fromkeys(
            baseArchName(a)
            for a in archs
            if isaInfoMap[gfxToIsa(a)].asmCaps["SupportedISA"]
        )
    )
    splitGSU = False

    start_pki = timer()
    passPostKernelInfoToLibrary(kernelInfo, uniqueKernels, masterLibraries, splitGSU)
    stop_pki = timer()
    print(f"Time to pass kernel info to library (s): {(stop_pki-start_pki):3.2f}")

    solDict = {}
    for solution in solutions:
        solutionKernels = solution.getKernels()
        for kernel in solutionKernels:
            kName = getKeyNoInternalArgs(kernel, False)
            if kName not in solDict:
                solDict["%s"%kName] = kernel

    # The per-arch subdirectories these loops write into were created by
    # writeSolutionsAndKernelsTCL above, from the same requested names; LibraryIO.write
    # does not create them.
    #
    # Split libraryMapping per arch and write one mapping file per arch into
    # that arch's per-base subdirectory. Every value ends in "_<arch>" because
    # tuned entries carry the arch natively and renameFallbacksPerArch
    # arch-suffixed every fallback entry before this point. Filtering on that
    # suffix keeps each arch's Mapping complete while letting builds produce
    # non-colliding mapping artifacts that survive overlay-style installs.
    for archName in archs:
        archMapping = {
            idx: name
            for idx, name in libraryMapping.items()
            if name.endswith("_" + archName)
        }
        if archMapping:
            archDir = libraryDir(outputPath, archName)
            archMappingFile = os.path.join(
                archDir, "TensileLiteLibrary_lazy_" + archName + "_Mapping"
            )
            LibraryIO.write(archMappingFile, archMapping, "msgpack")

    start_msl = timer()
    for archName, newMasterLibrary in masterLibraries.items():
        if archName in archs:
            archDir = libraryDir(outputPath, archName)
            def writeMsl(name, lib, archDir=archDir):
                filename = os.path.join(archDir, name)
                lib.applyNaming(splitGSU)
                LibraryIO.write(filename, state(lib), arguments["LibraryFormat"])

            if arguments["LazyLibraryLoading"]:
                masterFile = os.path.join(archDir, "TensileLibrary_lazy_" + archName)
            else:
                masterFile = os.path.join(archDir, "TensileLibrary_" + archName)
            newMasterLibrary.applyNaming(splitGSU)
            LibraryIO.write(masterFile, state(newMasterLibrary), arguments["LibraryFormat"])

            ParallelMap2(writeMsl,
                         newMasterLibrary.lazyLibraries.items(),
                         "Writing master solution libraries",
                         return_as="list")
    stop_msl = timer()
    print(f"Time to write master solution libraries (s): {(stop_msl-start_msl):3.2f}")

    if not arguments["KeepBuildTmp"]:
        # Left over from an older layout. With an OutputPath ending in "library"
        # it resolves onto the shared parent of this run's scratch, where a
        # concurrent group build may still be assembling into its own
        # subdirectory, so it is skipped rather than emptied.
        # Named from outputPath rather than from this run's scratch path, so the
        # architecture list cannot move what gets removed.
        scratchRoot = buildTmpRoot(outputPath)
        legacyBuildTmp = Path(arguments["OutputPath"]).parent / "library" / "build_tmp"
        if legacyBuildTmp.is_dir() and legacyBuildTmp.resolve() != scratchRoot.resolve():
            shutil.rmtree(legacyBuildTmp)
        if buildTmpPath.is_dir():
            if os.environ.get(_GROUP_BUILD_ENV):
                # A sibling group is building into its own subdirectory of the
                # same parent right now, so only this run's scratch may go. The
                # parent is left for whichever of us finishes last.
                shutil.rmtree(buildTmpPath)
                try:
                    scratchRoot.rmdir()
                except OSError:
                    pass
            else:
                # Nothing else is writing here, so take the whole tree -- which
                # also reclaims scratch left by an earlier build that named its
                # own directory differently.
                shutil.rmtree(scratchRoot)
        else:
            printWarning(f"Cannot remove build_tmp")

    print("# Tensile Library Writer DONE")
    print(HR)
    print("")

    stop = timer()

    print(f"Total time (s): {(stop-start):3.2f}")
    print(f"Total kernels processed: {numKernels}")
    print(f"Kernels processed per second: {(numKernels/(stop-start)):3.2f}")
    print(f"KernelHelperObjs: {len(kernelHelperObjs)}")
