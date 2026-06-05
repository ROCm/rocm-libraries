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
import pickle
import zlib
from pathlib import Path
from timeit import default_timer as timer
from typing import Collection, List, NamedTuple, Optional, Union

from Tensile import SOURCE_PATH, LibraryIO
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
    printWarning,
    state,
    tqdm,
    setVerbosity,
    getVerbosity,
)
from Tensile.Common.Architectures import gfxToIsa, isaToGfx, SUPPORTED_GFX, splitArchsFromPredicates, filterLogicFilesByPredicates
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Common.GlobalParameters import assignGlobalParameters, globalParameters
from Tensile.Common.TimingInstrumentation import timing_context
from Tensile.SolutionStructs.Naming import getKernelFileBase, getKeyNoInternalArgs, getKernelNameMin

from Tensile.CustomYamlLoader import load_logic_gfx_arch
from Tensile.KernelHelperNaming import kernelObjectNameCallables, initHelperKernelObjects
from Tensile.KernelWriterAssembly import KernelWriterAssembly
from Tensile.KernelWriterBase import (
    KERNEL_HELPER_FILENAME_CPP,
    KERNEL_HELPER_FILENAME_H,
)
from Tensile.SolutionLibrary import MasterSolutionLibrary, PlaceholderLibrary
from Tensile.SolutionStructs import Solution
from Tensile.SolutionStructs.Solution import mergeTypeMismatchCollector, printTypeMismatchSummary
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
    return libraryRoot(outputPath) / arch.split(":")[0]


def _baseArchs(archs: Collection[str]) -> List[str]:
    """Unique base archs (xnack/sramecc stripped), sorted for determinism."""
    return sorted({a.split(":")[0] for a in archs})


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
    return KernelCodeGenResult(
        err, src, header, asmFilename, objFilename, tuple(kernel["ISA"]), \
        kernel["WavefrontSize"], kernel["CUOccupancy"], \
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

    with timing_context("python_kernel_setup"):
        outputPath = Path(outputPath)
        # Builders fan out into <destRoot>/<base-arch>/ at the moment of write.
        # Pre-create the per-base subdirs so concurrent emit doesn't race mkdir.
        destRoot = ensurePath(libraryRoot(outputPath))
        for base in _baseArchs(cmdlineArchs):
            ensurePath(libraryDir(outputPath, base))
        buildTmpPath = ensurePath(outputPath / "build_tmp" / outputPath.stem.upper())  #
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
        asmToolchain.assembler(isaToGfx(isa), wavefrontsize, str(p), str(o_path))
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
                asmKernels,
                destRoot,
                assemblyTmpPath,
                compress,
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
        buildTmp = outputPath / "build_tmp"
        if buildTmp.exists() and buildTmp.is_dir():
            shutil.rmtree(buildTmp)

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
    removeTemporaries: bool=True
):
    outputPath = Path(outputPath)
    # Builders fan out into <destRoot>/<base-arch>/ at the moment of write.
    # Pre-create the per-base subdirs so concurrent emit doesn't race mkdir.
    destRoot = ensurePath(libraryRoot(outputPath))
    for base in _baseArchs(cmdlineArchs):
        ensurePath(libraryDir(outputPath, base))
    buildTmpPath = ensurePath(outputPath / "build_tmp" / outputPath.stem.upper())
    assemblyTmpPath = ensurePath(
        buildTmpPath / "assembly"
    )  # Temp path for generated assembly files (.s)
    objectTmpPath = ensurePath(
        buildTmpPath / "code_object_tmp"
    )  # Temp path for HSA code object files (.hsaco)

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
        asmToolchain.assembler(isaToGfx(isa), wavefrontsize, str(asmPath), str(o_path))
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
    )

    writeHelpers(outputPath, kernelHelperObjs, KERNEL_HELPER_FILENAME_CPP, KERNEL_HELPER_FILENAME_H)
    srcKernelFile = Path(outputPath) / "Kernels.cpp"

    buildSourceCodeObjectFiles(
        srcToolchain.compiler,
        srcToolchain.bundler,
        destRoot,
        objectTmpPath,
        outputPath,
        srcKernelFile,
        cmdlineArchs,
    )

    return len(uniqueAsmKernels), uniqueAsmKernels, results


@timing
def copyStaticFiles(outputPath):
    libraryStaticFiles = [
        "TensileTypes.h",
        "tensile_bfloat16.h",
        "tensile_float8_bfloat8.h",
        "KernelHeader.h",
        "ReductionTemplate.h",
        "memory_gfx.h",
    ]

    for fileName in libraryStaticFiles:
        shutil.copy(os.path.join(SOURCE_PATH, fileName), outputPath)

    return libraryStaticFiles


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
    khos = list(set(khos))
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

    for library in ParallelMap2(
        LibraryIO.parseLibraryLogicFile, fIter, "Loading Logics...", return_as="generator_unordered"
    ):
        _, architectureName, _, _, _, newLibrary, typeMismatches = library
        mergeTypeMismatchCollector(typeMismatches)

        if architectureName == "":
            continue

        if architectureName in masterLibraries:
            nextSolIndex = masterLibraries[architectureName].merge(newLibrary, nextSolIndex)
        else:
            masterLibraries[architectureName] = newLibrary
            masterLibraries[architectureName].version = args["CodeObjectVersion"]

    # After all YAML files have been parsed and Solution objects created,
    # print a summary of any type mismatches that were collected.
    printTypeMismatchSummary(len(logicFiles))

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
    archs = SUPPORTED_GFX if "all" in archs else archs
    archs, requestedPredicateMap = splitArchsFromPredicates(archs)

    targetIsas = [gfxToIsa(a) for a in archs]
    isaInfoMap = makeIsaInfoMap(targetIsas, cxxCompiler)
    assignGlobalParameters(arguments, isaInfoMap)

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

    def archMatch(arch: str, archs: List[str]):
        return (arch in archs) or any(a.startswith(arch) for a in archs)

    def validLogicFile(p: Path):
        return p.suffix == logicExtFormat and (
            "all" in archs or archMatch(load_logic_gfx_arch(p), archs)
        )

    globPattern = os.path.join(
        arguments["LogicPath"], f"**/{arguments['LogicFilter']}{logicExtFormat}"
    )
    print1(f"# LogicFilter:       {globPattern}")
    logicFiles = [
        file for file in glob.iglob(globPattern, recursive=True)
    ]

    logicFiles = [file for file in logicFiles if validLogicFile(Path(file))]

    print1(f"# Experimental:      {arguments['Experimental']}")
    if not arguments["Experimental"]:
        logicFiles = [
            file for file in logicFiles if "experimental" not in map(str.lower, Path(file).parts)
        ]

    print1("# Archs: " + ' ,'.join(archs))
    if requestedPredicateMap:
        print1("# Predicates:\n" + "\n".join(f"#   {arch}: {', '.join(v) if v else 'all variants'}" for arch, v in requestedPredicateMap.items()))
        numPrior = len(logicFiles)
        logicFiles = filterLogicFilesByPredicates(logicFiles, requestedPredicateMap)
        print1(f"# Filtered {numPrior - len(logicFiles)} logic files not matching requested predicates")

    print1(f"# LibraryLogicFiles: {len(logicFiles)}")

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

    copyStaticFiles(outputPath)

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

    archs = [ # is this really different than the other archs above?
        isaToGfx(arch)
        for arch in targetIsas
        if isaInfoMap[arch].asmCaps["SupportedISA"]
    ]
    # Per-base subdirs are created here (idempotent if writeSolutionsAndKernels*
    # already created them above). Each per-arch write below routes to its own
    # libraryDir(outputPath, archName).
    for base in _baseArchs(archs):
        ensurePath(libraryDir(outputPath, base))
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
        buildTmp = Path(arguments["OutputPath"]).parent / "library" / "build_tmp"
        if buildTmp.exists() and buildTmp.is_dir():
            shutil.rmtree(buildTmp)
        buildTmp = Path(arguments["OutputPath"]) / "build_tmp"
        if buildTmp.exists() and buildTmp.is_dir():
            shutil.rmtree(buildTmp)
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
